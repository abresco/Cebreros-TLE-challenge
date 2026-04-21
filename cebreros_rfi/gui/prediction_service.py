#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Prediction service for:
- manual future intervals
- schedule CSV batch processing

Prediction v1:
- geometry-first
- persistence-aware
- history-assisted
- outputs a heuristic probability (0-100)
"""

from typing import Dict, List, Tuple

from cebreros_rfi.src.core.candidate_helpers import preselect_geometric_candidates
from cebreros_rfi.src.core.operational_context import (
    DEFAULT_STEP_SIZE,
    FINAL_SEP_DEG,
    MAX_REASONABLE_RANGE_KM,
    PRESELECTION_SEP_DEG,
    build_operational_context,
)
from cebreros_rfi.src.feedback_db import (
    HistoryStats,
    get_candidate_history_stats,
    get_candidate_history_stats_for_mission,
)
from cebreros_rfi.src.horizons_target_track import normalize_station_id
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad
from cebreros_rfi.src.schedule_parser import parse_schedule_csv


TOP_N_OUTPUT = 20
MAX_RF_LOOKUPS = 20


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


def compute_history_component(history_stats: HistoryStats) -> float:
    confirmed = int(history_stats.get("confirmed", 0))
    rejected = int(history_stats.get("rejected", 0))
    uncertain = int(history_stats.get("uncertain", 0))

    return (
        min(20.0, confirmed * 4.0)
        - min(12.0, rejected * 3.0)
        + min(4.0, uncertain * 1.0)
    )


def select_effective_history(
    mission_history_stats: HistoryStats,
    global_history_stats: HistoryStats,
) -> Tuple[str, HistoryStats]:
    if int(mission_history_stats.get("total", 0)) > 0:
        return "mission", mission_history_stats
    if int(global_history_stats.get("total", 0)) > 0:
        return "global", global_history_stats
    return "none", {
        "confirmed": 0,
        "rejected": 0,
        "uncertain": 0,
        "total": 0,
    }


def compute_probability(
    min_sep_deg: float,
    close_samples: int,
    mission_history_stats: HistoryStats,
    global_history_stats: HistoryStats,
) -> float:
    geometry_component = max(0.0, 50.0 - 5.0 * float(min_sep_deg))
    duration_component = min(30.0, float(close_samples) * 3.0)

    if int(mission_history_stats.get("total", 0)) > 0:
        history_component = compute_history_component(mission_history_stats)
    elif int(global_history_stats.get("total", 0)) > 0:
        history_component = 0.6 * compute_history_component(global_history_stats)
    else:
        history_component = 0.0

    probability = geometry_component + duration_component + history_component
    return round(clamp(probability, 0.0, 100.0), 1)


def enrich_prediction_candidate(item: dict, victim_mission_id: str) -> dict:
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])
    mission_history_stats = get_candidate_history_stats_for_mission(
        mission_id=victim_mission_id,
        norad_cat_id=item["norad_cat_id"],
    )
    global_history_stats = get_candidate_history_stats(item["norad_cat_id"])
    history_source, effective_history = select_effective_history(
        mission_history_stats=mission_history_stats,
        global_history_stats=global_history_stats,
    )

    enriched = dict(item)
    enriched.update(
        {
            "victim_mission_id": victim_mission_id,
            "lookup_status": satnogs_info.get("lookup_status", "unknown"),
            "all_satnogs_freqs_mhz": satnogs_info.get("all_frequencies_mhz", []),
            "all_satnogs_bands": satnogs_info.get("all_bands", ["UNKNOWN"]),
            "cebreros_freqs_mhz": satnogs_info.get("frequencies_mhz", []),
            "cebreros_bands": satnogs_info.get("bands", ["UNKNOWN"]),
            "probability": compute_probability(
                min_sep_deg=item["min_sep_deg"],
                close_samples=item["close_samples"],
                mission_history_stats=mission_history_stats,
                global_history_stats=global_history_stats,
            ),
            "history_source": history_source,
            "history_confirmed": effective_history.get("confirmed", 0),
            "history_rejected": effective_history.get("rejected", 0),
            "history_uncertain": effective_history.get("uncertain", 0),
            "history_total": effective_history.get("total", 0),
            "mission_history_confirmed": mission_history_stats.get("confirmed", 0),
            "mission_history_rejected": mission_history_stats.get("rejected", 0),
            "mission_history_uncertain": mission_history_stats.get("uncertain", 0),
            "mission_history_total": mission_history_stats.get("total", 0),
            "global_history_confirmed": global_history_stats.get("confirmed", 0),
            "global_history_rejected": global_history_stats.get("rejected", 0),
            "global_history_uncertain": global_history_stats.get("uncertain", 0),
            "global_history_total": global_history_stats.get("total", 0),
        }
    )
    enriched["probability_label"] = probability_label(enriched["probability"])
    return enriched


def run_prediction_interval(
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
        raise PredictionServiceError(str(exc))

    preselected = preselect_geometric_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
        preselection_sep_deg=PRESELECTION_SEP_DEG,
        final_sep_deg=FINAL_SEP_DEG,
        max_reasonable_range_km=MAX_REASONABLE_RANGE_KM,
        include_possible_rfi_slot=True,
    )

    results = []
    lookups_done = 0

    for item in preselected:
        if lookups_done >= MAX_RF_LOOKUPS:
            break
        results.append(
            enrich_prediction_candidate(
                item=item,
                victim_mission_id=context.mission_id,
            )
        )
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
        "station_id": context.station_id,
        "mission_id": context.mission_id,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "catalog_meta": context.catalog_meta,
        "preselected_count": len(preselected),
        "lookups_done": lookups_done,
        "results": results[:TOP_N_OUTPUT],
    }


def load_prediction_jobs_from_schedule_csv(
    csv_path: str,
    station_filter: str,
) -> Dict:
    return parse_schedule_csv(
        csv_path=csv_path,
        station_filter=station_filter,
    )


def run_prediction_jobs(
    station_id: str,
    jobs: List[Dict],
) -> Dict:
    normalized_station_id = normalize_station_id(station_id)

    batch_results = []
    for job in jobs:
        batch_results.append(
            {
                "job_metadata": dict(job),
                "prediction": run_prediction_interval(
                    station_id=normalized_station_id,
                    mission_id=normalize_mission_name(job["mission_id"]),
                    start_utc=job["start_utc"],
                    end_utc=job["end_utc"],
                ),
            }
        )

    return {
        "station_id": normalized_station_id,
        "job_count": len(batch_results),
        "jobs": batch_results,
    }
