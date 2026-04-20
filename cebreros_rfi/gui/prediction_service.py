#!/usr/bin/env python3
"""
Prediction service for:
- manual future intervals
- future schedule XML placeholder / best-effort parser

Prediction v1:
- geometry-first
- frequency-aware
- history-assisted
- outputs a heuristic probability (0-100)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
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
from cebreros_rfi.src.feedback_db import get_candidate_history_stats


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
TOP_N_OUTPUT = 20
MAX_REASONABLE_RANGE_KM = 100000.0
MAX_RF_LOOKUPS = 120


class PredictionServiceError(RuntimeError):
    pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def probability_label(probability: float) -> str:
    if probability >= 75:
        return "HIGH"
    if probability >= 45:
        return "MEDIUM"
    return "LOW"


def compute_probability(
    min_sep_deg: float,
    close_samples: int,
    lookup_status: str,
    history_stats: Dict,
) -> float:
    geometry_component = max(0.0, 70.0 - 8.0 * float(min_sep_deg))
    duration_component = min(15.0, float(close_samples) * 1.5)

    lookup_status = str(lookup_status or "").strip().lower()
    if lookup_status == "ok":
        frequency_component = 15.0
    elif lookup_status == "no_ceb_band_match":
        frequency_component = 8.0
    elif lookup_status == "no_frequency_found":
        frequency_component = 4.0
    elif lookup_status == "lookup_error":
        frequency_component = 1.0
    else:
        frequency_component = 0.0

    confirmed = int(history_stats.get("confirmed", 0))
    rejected = int(history_stats.get("rejected", 0))
    uncertain = int(history_stats.get("uncertain", 0))

    history_component = min(18.0, confirmed * 4.0) - min(12.0, rejected * 3.0) + min(4.0, uncertain * 1.0)

    probability = geometry_component + duration_component + frequency_component + history_component
    return round(clamp(probability, 0.0, 100.0), 1)


def compute_possible_rfi_slot(close_indices: np.ndarray, timestamps: List[pd.Timestamp], closest_idx: int):
    if close_indices.size == 0:
        instant = timestamps[closest_idx].to_pydatetime().isoformat()
        return instant, instant

    start_idx = int(close_indices.min())
    end_idx = int(close_indices.max())
    return (
        timestamps[start_idx].to_pydatetime().isoformat(),
        timestamps[end_idx].to_pydatetime().isoformat(),
    )


def preselect_candidates(target_track_df: pd.DataFrame, satellites, station) -> List[dict]:
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
        close_indices = np.where(close_mask)[0]

        close_samples = int(np.count_nonzero(close_mask))
        visible_samples = int(np.count_nonzero(visible_mask))
        closest_time_utc = timestamps[min_idx].to_pydatetime().isoformat()
        slot_start_utc, slot_end_utc = compute_possible_rfi_slot(close_indices, timestamps, min_idx)

        results.append(
            {
                "object_name": str(raw_row.get("OBJECT_NAME", "")).strip() or "UNKNOWN",
                "norad_cat_id": str(raw_row.get("NORAD_CAT_ID", "")).strip(),
                "groups": ",".join(groups),
                "min_sep_deg": min_sep,
                "closest_time_utc": closest_time_utc,
                "visible_samples": visible_samples,
                "close_samples": close_samples,
                "sat_az_deg": float(sat_az[min_idx]),
                "sat_el_deg": float(sat_el[min_idx]),
                "sat_range_km": min_range_km,
                "target_az_deg": float(target_az[min_idx]),
                "target_el_deg": float(target_el[min_idx]),
                "possible_rfi_start_utc": slot_start_utc,
                "possible_rfi_end_utc": slot_end_utc,
            }
        )

    results.sort(key=lambda item: (item["min_sep_deg"], -item["close_samples"], item["closest_time_utc"]))
    return results


def enrich_prediction_candidate(item: dict) -> dict:
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])
    history_stats = get_candidate_history_stats(item["norad_cat_id"])

    lookup_status = satnogs_info.get("lookup_status", "unknown")
    probability = compute_probability(
        min_sep_deg=item["min_sep_deg"],
        close_samples=item["close_samples"],
        lookup_status=lookup_status,
        history_stats=history_stats,
    )

    enriched = dict(item)
    enriched.update(
        {
            "lookup_status": lookup_status,
            "all_satnogs_freqs_mhz": satnogs_info.get("all_frequencies_mhz", []),
            "all_satnogs_bands": satnogs_info.get("all_bands", ["UNKNOWN"]),
            "cebreros_freqs_mhz": satnogs_info.get("frequencies_mhz", []),
            "cebreros_bands": satnogs_info.get("bands", ["UNKNOWN"]),
            "probability": probability,
            "probability_label": probability_label(probability),
            "history_confirmed": history_stats.get("confirmed", 0),
            "history_rejected": history_stats.get("rejected", 0),
            "history_uncertain": history_stats.get("uncertain", 0),
        }
    )
    return enriched


def run_prediction_interval(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
) -> Dict:
    station_id = str(station_id).strip().upper()
    mission_id = normalize_mission_name(mission_id)

    if station_id not in STATION_CONFIG:
        raise PredictionServiceError("Unsupported station ID: {0}".format(station_id))

    catalog_meta = get_local_catalog_metadata()
    if not catalog_meta["exists"]:
        raise PredictionServiceError(
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

    station_cfg = STATION_CONFIG[station_id]
    station = build_station(
        station_cfg["lat_deg"],
        station_cfg["lon_deg"],
        station_cfg["elevation_m"],
    )

    preselected = preselect_candidates(target_track_df, satellites, station)

    results = []
    lookups_done = 0

    for item in preselected:
        if lookups_done >= MAX_RF_LOOKUPS:
            break
        results.append(enrich_prediction_candidate(item))
        lookups_done += 1

    results.sort(
        key=lambda item: (
            -item["probability"],
            item["min_sep_deg"],
            -item["close_samples"],
            item["closest_time_utc"],
        )
    )

    return {
        "station_id": station_id,
        "mission_id": mission_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "catalog_meta": catalog_meta,
        "preselected_count": len(preselected),
        "lookups_done": lookups_done,
        "results": results[:TOP_N_OUTPUT],
    }


def parse_schedule_xml(xml_path: str) -> List[Dict]:
    """
    Best-effort parser prepared for future ESA Scheduling XML integration.

    Expected output:
    [
        {"mission_id": "...", "start_utc": "...", "end_utc": "..."}
    ]

    This parser tries common tag/attribute names and can be adapted later.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise PredictionServiceError("XML file not found: {0}".format(xml_path))

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    jobs = []

    candidate_nodes = []
    for tag_name in ("pass", "allocation", "track", "activity", "event"):
        candidate_nodes.extend(root.findall(".//{0}".format(tag_name)))
        candidate_nodes.extend(root.findall(".//{{*}}{0}".format(tag_name)))

    def first_present(d, keys):
        for key in keys:
            value = d.get(key)
            if value:
                return value
        return None

    for node in candidate_nodes:
        attrs = dict(node.attrib)

        text_fields = {}
        for child in list(node):
            tag = child.tag.split("}")[-1].lower()
            text_fields[tag] = (child.text or "").strip()

        source = {}
        source.update(attrs)
        source.update(text_fields)

        mission_id = first_present(source, ["mission_id", "mission", "spacecraft", "name"])
        start_utc = first_present(source, ["start_utc", "start", "begin", "starttime"])
        end_utc = first_present(source, ["end_utc", "end", "stop", "endtime"])

        if mission_id and start_utc and end_utc:
            jobs.append(
                {
                    "mission_id": normalize_mission_name(mission_id),
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                }
            )

    if not jobs:
        raise PredictionServiceError(
            "No prediction jobs could be extracted from the XML file. "
            "Adapt parse_schedule_xml() once the ESA Scheduling XML structure is confirmed."
        )

    return jobs


def run_prediction_schedule(
    station_id: str,
    xml_path: str,
) -> Dict:
    jobs = parse_schedule_xml(xml_path)

    batch_results = []
    for job in jobs:
        batch_results.append(
            run_prediction_interval(
                station_id=station_id,
                mission_id=job["mission_id"],
                start_utc=job["start_utc"],
                end_utc=job["end_utc"],
            )
        )

    return {
        "station_id": station_id,
        "job_count": len(batch_results),
        "jobs": batch_results,
    }