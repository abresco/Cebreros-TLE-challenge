#!/usr/bin/env python3
"""
Service layer for the Streamlit identification workflow.

Known-frequency candidates:
- every candidate whose lookup_status is different from "not_found"

Unknown-frequency candidates:
- only candidates whose lookup_status is exactly "not_found"
"""

from typing import Dict, List

from cebreros_rfi.src.core.candidate_helpers import (
    candidate_sort_key,
    classify_band_match,
    compute_band_penalty,
    has_known_frequency,
    preselect_geometric_candidates,
)
from cebreros_rfi.src.core.operational_context import (
    DEFAULT_STEP_SIZE,
    FINAL_SEP_DEG,
    MAX_REASONABLE_RANGE_KM,
    PRESELECTION_SEP_DEG,
    build_operational_context,
)
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad


TOP_N_KNOWN_OUTPUT = 10
TOP_N_UNKNOWN_OUTPUT = 10
MAX_RF_LOOKUPS = 120


class IdentificationServiceError(RuntimeError):
    pass


def enrich_candidate(item: dict, allowed_bands: set) -> dict:
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])

    satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
    lookup_status = satnogs_info.get("lookup_status", "unknown")
    known_frequency = has_known_frequency(lookup_status)

    enriched = dict(item)
    enriched.update(
        {
            "satnogs_bands": satnogs_bands,
            "satnogs_freqs_mhz": satnogs_info.get("frequencies_mhz", []),
            "all_satnogs_bands": satnogs_info.get("all_bands", ["UNKNOWN"]),
            "all_satnogs_freqs_mhz": satnogs_info.get("all_frequencies_mhz", []),
            "band_match": classify_band_match(satnogs_bands, allowed_bands),
            "satnogs_url": satnogs_info.get("satnogs_url"),
            "lookup_status": lookup_status,
            "has_known_frequency": known_frequency,
            "manual_review_candidate": not known_frequency,
        }
    )
    enriched["score"] = (
        compute_band_penalty(enriched["band_match"])
        + enriched["min_sep_deg"]
        - 0.05 * enriched["close_samples"]
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

        enough_known = len(known_results) >= TOP_N_KNOWN_OUTPUT
        enough_unknown = len(unknown_results) >= TOP_N_UNKNOWN_OUTPUT
        if enough_known and enough_unknown:
            break

    ranking_key = lambda item: (item["score"],) + candidate_sort_key(item)
    all_results.sort(key=ranking_key)
    known_results.sort(key=ranking_key)
    unknown_results.sort(key=ranking_key)

    return all_results, known_results, unknown_results, lookups_done


def run_identification(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
) -> Dict:
    try:
        context = build_operational_context(
            station_id=station_id,
            mission_id=mission_id,
            start_utc=start_utc,
            end_utc=end_utc,
            step_size=DEFAULT_STEP_SIZE,
        )
    except Exception as exc:
        raise IdentificationServiceError(str(exc))

    preselected = preselect_geometric_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
        preselection_sep_deg=PRESELECTION_SEP_DEG,
        final_sep_deg=FINAL_SEP_DEG,
        max_reasonable_range_km=MAX_REASONABLE_RANGE_KM,
    )

    all_results, known_results, unknown_results, lookups_done = build_rankings(
        preselected_results=preselected,
        allowed_bands=context.allowed_bands,
    )

    return {
        "station_id": context.station_id,
        "mission_id": context.mission_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "catalog_meta": context.catalog_meta,
        "preselected_count": len(preselected),
        "lookups_done": lookups_done,
        "known_results": known_results,
        "unknown_results": unknown_results,
        "all_results": all_results,
    }
