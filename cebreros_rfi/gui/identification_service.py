#!/usr/bin/env python3
"""
Service layer for the Streamlit GUI.

Known-frequency candidates:
- every candidate whose lookup_status is different from "not_found"

Unknown-frequency candidates:
- only candidates whose lookup_status is exactly "not_found"
"""

from datetime import timezone
from typing import Dict, List

from skyfield.api import load

from cebreros_rfi.src.case2_rfi_celestrak import (
    angular_separation_deg,
    build_station,
)
from cebreros_rfi.src.horizons_target_track import fetch_target_track
from cebreros_rfi.src.local_candidate_catalog import (
    get_local_catalog_metadata,
    load_local_candidate_catalog,
)
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad

import numpy as np
import pandas as pd


UTC = timezone.utc

STATION_CONFIG = {
    "CEB": {
        "lat_deg": 40.4526889,
        "lon_deg": -4.36755,
        "elevation_m": 794.0,
        "allowed_bands": {"X", "KA"},
    }
}

DEFAULT_STEP_SIZE = "60s"
PRESELECTION_SEP_DEG = 10.0
FINAL_SEP_DEG = 5.0
TOP_N_KNOWN_OUTPUT = 10
TOP_N_UNKNOWN_OUTPUT = 10
MAX_REASONABLE_RANGE_KM = 100000.0
MAX_RF_LOOKUPS = 120


class IdentificationServiceError(RuntimeError):
    pass


def classify_band_match(candidate_bands: List[str], allowed_bands: set) -> str:
    normalized = set([str(item).upper() for item in candidate_bands if item])
    normalized = normalized.intersection({"X", "KA"})

    if not normalized:
        return "UNKNOWN"

    if normalized.intersection(allowed_bands):
        return "MATCH"

    return "MISMATCH"


def compute_band_penalty(band_match: str) -> float:
    if band_match == "MATCH":
        return 0.0
    if band_match == "UNKNOWN":
        return 1.0
    return 5.0


def is_known_frequency_candidate(lookup_status: str) -> bool:
    """
    New rule requested:
    - known = everything except not_found
    - unknown = only not_found
    """
    return str(lookup_status or "").strip().lower() != "not_found"


def preselect_candidates(
    target_track_df: pd.DataFrame,
    satellites,
    station,
) -> List[dict]:
    ts = load.timescale()

    timestamps = list(target_track_df["UTC"])
    times = ts.from_datetimes([value.to_pydatetime() for value in timestamps])

    target_az = np.array(target_track_df["AZ_target_deg"].values, dtype=float)
    target_el = np.array(target_track_df["EL_target_deg"].values, dtype=float)

    results = []

    for satellite, groups, raw_row in satellites:
        difference = satellite - station
        topocentric = difference.at(times)
        alt, az, distance = topocentric.altaz()

        sat_el = np.asarray(alt.degrees, dtype=float)
        sat_az = np.asarray(az.degrees, dtype=float)
        sat_range_km = np.asarray(distance.km, dtype=float)

        visible_mask = sat_el > 0.0
        if not np.any(visible_mask):
            continue

        separation = angular_separation_deg(target_az, target_el, sat_az, sat_el)
        valid_mask = visible_mask & np.isfinite(separation)
        if not np.any(valid_mask):
            continue

        min_idx = int(np.argmin(np.where(valid_mask, separation, np.inf)))
        min_sep = float(separation[min_idx])
        min_range_km = float(sat_range_km[min_idx])

        if min_sep > PRESELECTION_SEP_DEG:
            continue

        if not np.isfinite(min_range_km) or min_range_km > MAX_REASONABLE_RANGE_KM:
            continue

        close_mask = valid_mask & (separation <= FINAL_SEP_DEG)
        close_samples = int(np.count_nonzero(close_mask))
        visible_samples = int(np.count_nonzero(visible_mask))
        closest_time_utc = timestamps[min_idx].to_pydatetime()

        results.append(
            {
                "object_name": str(raw_row.get("OBJECT_NAME", "")).strip() or "UNKNOWN",
                "norad_cat_id": str(raw_row.get("NORAD_CAT_ID", "")).strip(),
                "groups": ",".join(groups),
                "min_sep_deg": min_sep,
                "closest_time_utc": closest_time_utc.isoformat(),
                "visible_samples": visible_samples,
                "close_samples": close_samples,
                "sat_az_deg": float(sat_az[min_idx]),
                "sat_el_deg": float(sat_el[min_idx]),
                "sat_range_km": min_range_km,
                "target_az_deg": float(target_az[min_idx]),
                "target_el_deg": float(target_el[min_idx]),
            }
        )

    results.sort(key=lambda item: (item["min_sep_deg"], -item["close_samples"], item["closest_time_utc"]))
    return results


def enrich_candidate(item: dict, allowed_bands: set) -> dict:
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])

    satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
    satnogs_freqs_mhz = satnogs_info.get("frequencies_mhz", [])
    all_satnogs_bands = satnogs_info.get("all_bands", ["UNKNOWN"])
    all_satnogs_freqs_mhz = satnogs_info.get("all_frequencies_mhz", [])
    lookup_status = satnogs_info.get("lookup_status", "unknown")

    band_match = classify_band_match(satnogs_bands, allowed_bands)
    known_frequency = is_known_frequency_candidate(lookup_status)

    score = (
        compute_band_penalty(band_match)
        + item["min_sep_deg"]
        - 0.05 * item["close_samples"]
    )

    enriched = dict(item)
    enriched.update(
        {
            "satnogs_bands": satnogs_bands,
            "satnogs_freqs_mhz": satnogs_freqs_mhz,
            "all_satnogs_bands": all_satnogs_bands,
            "all_satnogs_freqs_mhz": all_satnogs_freqs_mhz,
            "band_match": band_match,
            "score": score,
            "satnogs_url": satnogs_info.get("satnogs_url"),
            "lookup_status": lookup_status,
            "has_known_frequency": known_frequency,
            "manual_review_candidate": not known_frequency,
        }
    )
    return enriched


def build_rankings(preselected_results: List[dict], allowed_bands: set):
    all_results = []
    known_results = []
    unknown_results = []
    lookups_done = 0

    for item in preselected_results:
        if lookups_done >= MAX_RF_LOOKUPS:
            break

        candidate = enrich_candidate(item, allowed_bands)
        all_results.append(candidate)
        lookups_done += 1

        if candidate["has_known_frequency"]:
            known_results.append(candidate)
        else:
            unknown_results.append(candidate)

        if (
            len(known_results) >= TOP_N_KNOWN_OUTPUT
            and len(unknown_results) >= TOP_N_UNKNOWN_OUTPUT
        ):
            break

    all_results.sort(key=lambda item: (item["score"], item["min_sep_deg"], -item["close_samples"], item["closest_time_utc"]))
    known_results.sort(key=lambda item: (item["score"], item["min_sep_deg"], -item["close_samples"], item["closest_time_utc"]))
    unknown_results.sort(key=lambda item: (item["score"], item["min_sep_deg"], -item["close_samples"], item["closest_time_utc"]))

    return all_results, known_results, unknown_results, lookups_done


def run_identification(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
) -> Dict:
    station_id = str(station_id).strip().upper()
    mission_id = normalize_mission_name(mission_id)

    if station_id not in STATION_CONFIG:
        raise IdentificationServiceError("Unsupported station ID: {0}".format(station_id))

    station_cfg = STATION_CONFIG[station_id]
    allowed_bands = set(station_cfg["allowed_bands"])

    catalog_meta = get_local_catalog_metadata()
    if not catalog_meta["exists"]:
        raise IdentificationServiceError(
            "Local ACTIVE catalog not found. Run update_candidate_catalog.py first."
        )

    target_track_df = fetch_target_track(
        mission_id=mission_id,
        station_id=station_id,
        start_time_utc=start_utc,
        stop_time_utc=end_utc,
        step_size=DEFAULT_STEP_SIZE,
    )

    ts = load.timescale()
    satellites = load_local_candidate_catalog(ts)

    station = build_station(
        station_cfg["lat_deg"],
        station_cfg["lon_deg"],
        station_cfg["elevation_m"],
    )

    preselected = preselect_candidates(
        target_track_df=target_track_df,
        satellites=satellites,
        station=station,
    )

    all_results, known_results, unknown_results, lookups_done = build_rankings(
        preselected_results=preselected,
        allowed_bands=allowed_bands,
    )

    return {
        "station_id": station_id,
        "mission_id": mission_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "catalog_meta": catalog_meta,
        "preselected_count": len(preselected),
        "lookups_done": lookups_done,
        "known_results": known_results,
        "unknown_results": unknown_results,
        "all_results": all_results,
    }