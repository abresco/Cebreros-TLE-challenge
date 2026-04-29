#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Service layer for the Streamlit identification workflow.

Known-frequency candidates:
- every candidate whose lookup_status is different from "not_found"

Unknown-frequency candidates:
- only candidates whose lookup_status is exactly "not_found"
"""

from typing import Dict, List

from cebreros_rfi.src.config_loader import get_identification_config
from cebreros_rfi.src.core.candidate_helpers import (
    candidate_sort_key,
    classify_band_match,
    compute_band_penalty,
    has_known_frequency,
    preselect_geometric_candidates,
)
from cebreros_rfi.src.core.operational_context import (
    build_operational_context,
)
from cebreros_rfi.src.input_validation import (
    InputValidationError,
    validate_identification_interval,
    validate_mission_id,
    validate_station_id,
)
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad


# Description:
#   Error raised for user-facing Identification service failures.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class IdentificationServiceError(RuntimeError):
    pass


# Description:
#   Load Identification runtime settings from the YAML configuration.
# input:-
#   None.
# output:-
#   None.
# return:-
#   Dict with Identification thresholds, lookup limits, and output limits.
def get_runtime_config() -> Dict:
    return get_identification_config()


# Description:
#   Add SatNOGS RF metadata and ranking fields to one geometry candidate.
# input:-
#   item: geometry-only candidate produced by preselection.
#   allowed_bands: RF bands configured for the selected station.
# output:-
#   Reads SatNOGS cache/API.
# return:-
#   Dict containing original geometry plus RF context and ranking score.
def enrich_candidate(item: dict, allowed_bands: set) -> dict:
    # SatNOGS lookup is intentionally done after geometric filtering because it
    # is network/cache backed and much more expensive than local geometry.
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])

    satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
    lookup_status = satnogs_info.get("lookup_status", "unknown")
    known_frequency = has_known_frequency(lookup_status)

    # Copy the geometry payload so callers can keep using the original object.
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
    # Lower score is better: prefer station-band matches, smaller angular
    # separation, and more persistent close samples.
    enriched["score"] = (
        compute_band_penalty(enriched["band_match"])
        + enriched["min_sep_deg"]
        - 0.05 * enriched["close_samples"]
    )
    return enriched


# Description:
#   Build known-frequency and unknown-frequency rankings for Identification.
# input:-
#   preselected_results: geometry candidates sorted by closeness.
#   allowed_bands: RF bands configured for the selected station.
# output:-
#   Performs up to the configured number of SatNOGS lookups.
# return:-
#   Tuple of all results, known-frequency results, unknown-frequency results,
#   and number of RF lookups performed.
def build_rankings(preselected_results: List[dict], allowed_bands: set):
    config = get_runtime_config()
    max_rf_lookups = int(config["max_rf_lookups"])
    top_n_known_output = int(config["top_n_known_output"])
    top_n_unknown_output = int(config["top_n_unknown_output"])

    all_results = []
    known_results = []
    unknown_results = []
    lookups_done = 0

    for item in preselected_results:
        # Stop before exhausting external/cache lookups beyond the configured
        # limit for an interactive Identification run.
        if lookups_done >= max_rf_lookups:
            break

        candidate = enrich_candidate(item, allowed_bands)
        all_results.append(candidate)
        lookups_done += 1

        if candidate["has_known_frequency"]:
            known_results.append(candidate)
        else:
            unknown_results.append(candidate)

        # Once both visible tables are full, extra enrichment would not change
        # the default GUI output.
        enough_known = len(known_results) >= top_n_known_output
        enough_unknown = len(unknown_results) >= top_n_unknown_output
        if enough_known and enough_unknown:
            break

    ranking_key = lambda item: (item["score"],) + candidate_sort_key(item)
    all_results.sort(key=ranking_key)
    known_results.sort(key=ranking_key)
    unknown_results.sort(key=ranking_key)

    return all_results, known_results, unknown_results, lookups_done


# Description:
#   Run the complete manual Identification workflow for one past interval.
# input:-
#   station_id: station selected by the user.
#   mission_id: victim mission selected by the user.
#   start_utc: interval start in YYYY-MM-DD HH:MM:SS.
#   end_utc: interval end in YYYY-MM-DD HH:MM:SS.
# output:-
#   May refresh/load catalog cache, query Horizons, and read SatNOGS.
# return:-
#   Dict with metadata, preselection counts, lookup counts, and ranked results.
def run_identification(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
) -> Dict:
    config = get_runtime_config()

    try:
        # Validate before loading catalogs or calling Horizons so bad GUI input
        # fails quickly with an operator-friendly message.
        normalized_station_id = validate_station_id(station_id)
        normalized_mission_id = validate_mission_id(mission_id)
        validate_identification_interval(start_utc, end_utc)

        context = build_operational_context(
            station_id=normalized_station_id,
            mission_id=normalized_mission_id,
            start_utc=start_utc,
            end_utc=end_utc,
            step_size=str(config["step_size"]),
        )
    except InputValidationError as exc:
        raise IdentificationServiceError(str(exc)) from exc
    except Exception as exc:
        raise IdentificationServiceError(str(exc))

    # Geometry preselection keeps SatNOGS/API enrichment limited to plausible
    # angular coincidences near the victim mission track.
    preselected = preselect_geometric_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
        preselection_sep_deg=float(config["preselection_sep_deg"]),
        final_sep_deg=float(config["final_sep_deg"]),
        max_reasonable_range_km=float(config["max_reasonable_range_km"]),
    )

    all_results, known_results, unknown_results, lookups_done = build_rankings(
        preselected_results=preselected,
        allowed_bands=context.allowed_bands,
    )

    # Preserve the response shape consumed by the Streamlit rendering code.
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
