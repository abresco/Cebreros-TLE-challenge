#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Shared helpers for candidate preselection and ranking.

These helpers keep the geometry and RF classification rules in one place so
identification and prediction stay aligned.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from skyfield.api import load

from cebreros_rfi.src.config_loader import get_supported_bands
from cebreros_rfi.src.core.geometry_utils import angular_separation_deg


SUPPORTED_CEBREROS_BANDS = frozenset(get_supported_bands())


def classify_band_match(candidate_bands: List[str], allowed_bands: set) -> str:
    """
    Compare candidate RF bands against the station receive bands.
    """
    normalized = {
        str(item).upper()
        for item in candidate_bands
        if item
    }.intersection(SUPPORTED_CEBREROS_BANDS)

    if not normalized:
        return "UNKNOWN"

    if normalized.intersection(allowed_bands):
        return "MATCH"

    return "MISMATCH"


def compute_band_penalty(band_match: str) -> float:
    """
    Lower values are better for ranking.
    """
    if band_match == "MATCH":
        return 0.0
    if band_match == "UNKNOWN":
        return 1.0
    return 5.0


def has_known_frequency(lookup_status: str) -> bool:
    """
    Treat every lookup except an explicit "not_found" as known RF context.
    """
    return str(lookup_status or "").strip().lower() != "not_found"


def candidate_sort_key(candidate: Dict) -> Tuple:
    """
    Shared tie-break order for geometry-first candidate lists.
    """
    return (
        candidate["min_sep_deg"],
        -candidate["close_samples"],
        candidate["closest_time_utc"],
    )


def compute_possible_rfi_slot(
    close_indices: np.ndarray,
    timestamps: Sequence,
    closest_idx: int,
) -> Tuple[str, str]:
    """
    Return the full close-approach window or a single instant if there is only
    one valid sample.
    """
    if close_indices.size == 0:
        instant = timestamps[closest_idx].to_pydatetime().isoformat()
        return instant, instant

    start_idx = int(close_indices.min())
    end_idx = int(close_indices.max())
    return (
        timestamps[start_idx].to_pydatetime().isoformat(),
        timestamps[end_idx].to_pydatetime().isoformat(),
    )


def preselect_geometric_candidates(
    target_track_df,
    satellites,
    station,
    preselection_sep_deg: float,
    final_sep_deg: float,
    max_reasonable_range_km: float,
    include_possible_rfi_slot: bool = False,
) -> List[dict]:
    """
    Run the shared geometry filter used by identification and prediction.
    """
    ts = load.timescale()

    timestamps = list(target_track_df["UTC"])
    times = ts.from_datetimes([value.to_pydatetime() for value in timestamps])

    target_az = np.asarray(target_track_df["AZ_target_deg"].values, dtype=float)
    target_el = np.asarray(target_track_df["EL_target_deg"].values, dtype=float)

    results = []

    for satellite, groups, raw_row in satellites:
        topocentric = (satellite - station).at(times)
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

        if min_sep > preselection_sep_deg:
            continue

        if not np.isfinite(min_range_km) or min_range_km > max_reasonable_range_km:
            continue

        close_mask = valid_mask & (separation <= final_sep_deg)
        close_indices = np.where(close_mask)[0]

        candidate = {
            "object_name": str(raw_row.get("OBJECT_NAME", "")).strip() or "UNKNOWN",
            "norad_cat_id": str(raw_row.get("NORAD_CAT_ID", "")).strip(),
            "groups": ",".join(groups),
            "min_sep_deg": min_sep,
            "closest_time_utc": timestamps[min_idx].to_pydatetime().isoformat(),
            "visible_samples": int(np.count_nonzero(visible_mask)),
            "close_samples": int(np.count_nonzero(close_mask)),
            "sat_az_deg": float(sat_az[min_idx]),
            "sat_el_deg": float(sat_el[min_idx]),
            "sat_range_km": min_range_km,
            "target_az_deg": float(target_az[min_idx]),
            "target_el_deg": float(target_el[min_idx]),
        }

        if include_possible_rfi_slot:
            slot_start_utc, slot_end_utc = compute_possible_rfi_slot(
                close_indices=close_indices,
                timestamps=timestamps,
                closest_idx=min_idx,
            )
            candidate["possible_rfi_start_utc"] = slot_start_utc
            candidate["possible_rfi_end_utc"] = slot_end_utc

        results.append(candidate)

    results.sort(key=candidate_sort_key)
    return results
