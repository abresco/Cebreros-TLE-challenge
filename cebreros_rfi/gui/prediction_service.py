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

from cebreros_rfi.src.config_loader import (
    get_prediction_config,
    get_prediction_scoring_config,
)
from cebreros_rfi.src.core.candidate_helpers import preselect_geometric_candidates
from cebreros_rfi.src.core.operational_context import (
    build_operational_context,
)
from cebreros_rfi.src.feedback_db import (
    HistoryStats,
    get_candidate_history_stats,
    get_candidate_history_stats_for_mission,
)
from cebreros_rfi.src.input_validation import (
    InputValidationError,
    validate_mission_id,
    validate_prediction_interval,
    validate_station_id,
)
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad
from cebreros_rfi.src.schedule_parser import ScheduleParserError, parse_schedule_csv


# Description:
#   Error raised for user-facing Prediction service failures.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class PredictionServiceError(RuntimeError):
    pass


# Description:
#   Load Prediction runtime settings from the YAML configuration.
# input:-
#   None.
# output:-
#   None.
# return:-
#   Dict with Prediction thresholds, lookup limits, and scoring config.
def get_runtime_config() -> Dict:
    return get_prediction_config()


# Description:
#   Load the heuristic probability scoring weights from YAML.
# input:-
#   None.
# output:-
#   None.
# return:-
#   Dict with geometry, duration, and feedback scoring weights.
def get_scoring_runtime_config() -> Dict:
    return get_prediction_scoring_config()


# Description:
#   Clamp a numeric value to a closed interval.
# input:-
#   value: number to clamp.
#   low: minimum accepted value.
#   high: maximum accepted value.
# output:-
#   None.
# return:-
#   The value limited to the range [low, high].
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Description:
#   Convert a numeric prediction score into a coarse operator label.
# input:-
#   probability: heuristic score from 0 to 100.
# output:-
#   None.
# return:-
#   "HIGH", "MEDIUM", or "LOW".
def probability_label(probability: float) -> str:
    if probability >= 75:
        return "HIGH"
    if probability >= 45:
        return "MEDIUM"
    return "LOW"


# Description:
#   Convert feedback label counts into a scoring contribution.
# input:-
#   history_stats: counts for confirmed, rejected, uncertain, and total labels.
#   scoring_cfg: configured scoring gains, penalties, and caps.
# output:-
#   None.
# return:-
#   Signed numeric contribution added to the Prediction score.
def compute_history_component(history_stats: HistoryStats, scoring_cfg: Dict) -> float:
    confirmed = int(history_stats.get("confirmed", 0))
    rejected = int(history_stats.get("rejected", 0))
    uncertain = int(history_stats.get("uncertain", 0))

    return (
        min(float(scoring_cfg["history_confirmed_cap"]), confirmed * float(scoring_cfg["history_confirmed_gain"]))
        - min(float(scoring_cfg["history_rejected_cap"]), rejected * float(scoring_cfg["history_rejected_penalty"]))
        + min(float(scoring_cfg["history_uncertain_cap"]), uncertain * float(scoring_cfg["history_uncertain_gain"]))
    )


# Description:
#   Choose which feedback history source should affect scoring.
# input:-
#   mission_history_stats: labels for this victim mission and candidate.
#   global_history_stats: labels for this candidate across all missions.
# output:-
#   None.
# return:-
#   Tuple containing the selected source name and selected stats dictionary.
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


# Description:
#   Compute the bounded heuristic Prediction probability for one candidate.
# input:-
#   min_sep_deg: closest angular separation from the victim mission track.
#   close_samples: number of samples inside the final close-approach threshold.
#   mission_history_stats: mission-specific feedback counts.
#   global_history_stats: global feedback counts for this candidate.
# output:-
#   None.
# return:-
#   Rounded score between 0.0 and 100.0.
def compute_probability(
    min_sep_deg: float,
    close_samples: int,
    mission_history_stats: HistoryStats,
    global_history_stats: HistoryStats,
) -> float:
    scoring_cfg = get_scoring_runtime_config()

    # Geometry starts high and is penalized as the closest angular separation
    # grows. Negative geometry scores are clipped to zero.
    geometry_component = max(
        0.0,
        float(scoring_cfg["geometry_base"])
        - float(scoring_cfg["geometry_penalty_per_deg"]) * float(min_sep_deg),
    )
    # Persistence rewards candidates that stay close for multiple samples.
    duration_component = min(
        float(scoring_cfg["duration_cap"]),
        float(close_samples) * float(scoring_cfg["duration_gain_per_sample"]),
    )

    # Mission-specific history is stronger evidence; global history is only
    # used when no victim-mission evidence exists for the candidate.
    if int(mission_history_stats.get("total", 0)) > 0:
        history_component = compute_history_component(mission_history_stats, scoring_cfg)
    elif int(global_history_stats.get("total", 0)) > 0:
        history_component = float(scoring_cfg["global_history_weight"]) * compute_history_component(
            global_history_stats,
            scoring_cfg,
        )
    else:
        history_component = 0.0

    probability = geometry_component + duration_component + history_component
    return round(clamp(probability, 0.0, 100.0), 1)


# Description:
#   Add RF metadata, feedback history, and probability fields to a candidate.
# input:-
#   item: geometry-only candidate produced by preselection.
#   victim_mission_id: normalized victim mission ID for mission-specific history.
# output:-
#   Reads SatNOGS cache/API and feedback DB.
# return:-
#   Dict containing original geometry plus Prediction-specific fields.
def enrich_prediction_candidate(item: dict, victim_mission_id: str) -> dict:
    # RF metadata is contextual for operators and does not directly increase the
    # current probability score.
    satnogs_info = lookup_bands_by_norad(item["norad_cat_id"])
    # Read both history scopes so the UI can show which evidence was used.
    mission_history_stats = get_candidate_history_stats_for_mission(
        mission_id=victim_mission_id,
        norad_cat_id=item["norad_cat_id"],
    )
    global_history_stats = get_candidate_history_stats(item["norad_cat_id"])
    history_source, effective_history = select_effective_history(
        mission_history_stats=mission_history_stats,
        global_history_stats=global_history_stats,
    )

    # Keep the original geometry fields and add prediction-specific evidence.
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


# Description:
#   Run Prediction for one manually supplied future interval.
# input:-
#   station_id: station selected by the user.
#   mission_id: victim mission selected by the user.
#   start_utc: interval start in YYYY-MM-DD HH:MM:SS.
#   end_utc: interval end in YYYY-MM-DD HH:MM:SS.
# output:-
#   May refresh/load catalog cache, query Horizons, read SatNOGS, and read DB.
# return:-
#   Dict with metadata, preselection counts, lookup counts, and ranked results.
def run_prediction_interval(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
) -> Dict:
    config = get_runtime_config()

    try:
        # Invalid user input should fail before catalog refresh, Horizons, or
        # SatNOGS work begins.
        normalized_station_id = validate_station_id(station_id)
        normalized_mission_id = validate_mission_id(mission_id)
        validate_prediction_interval(start_utc, end_utc)

        context = build_operational_context(
            station_id=normalized_station_id,
            mission_id=normalized_mission_id,
            start_utc=start_utc,
            end_utc=end_utc,
            step_size=str(config["step_size"]),
        )
    except InputValidationError as exc:
        raise PredictionServiceError(str(exc)) from exc
    except Exception as exc:
        raise PredictionServiceError(str(exc))

    # This is a broad geometry filter before the more expensive RF lookup and
    # feedback-assisted probability scoring.
    preselected = preselect_geometric_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
        preselection_sep_deg=float(config["preselection_sep_deg"]),
        final_sep_deg=float(config["final_sep_deg"]),
        max_reasonable_range_km=float(config["max_reasonable_range_km"]),
        include_possible_rfi_slot=True,
    )

    results = []
    lookups_done = 0
    max_rf_lookups = int(config["max_rf_lookups"])
    top_n_output = int(config["top_n_output"])

    for item in preselected:
        # Limit enrichment because each candidate may require RF/cache and DB
        # work. Geometry sort order decides which candidates are considered.
        if lookups_done >= max_rf_lookups:
            break
        results.append(
            enrich_prediction_candidate(
                item=item,
                victim_mission_id=context.mission_id,
            )
        )
        lookups_done += 1

    # Higher probability wins, then closer geometry and longer persistence.
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
        "results": results[:top_n_output],
    }


# Description:
#   Parse a schedule CSV into selectable Prediction jobs.
# input:-
#   csv_path: path to the uploaded or local CSV file.
#   station_filter: station selected by the user for filtering schedule rows.
# output:-
#   Reads the CSV file.
# return:-
#   Dict with usable jobs and a parsing summary.
def load_prediction_jobs_from_schedule_csv(
    csv_path: str,
    station_filter: str,
) -> Dict:
    try:
        return parse_schedule_csv(
            csv_path=csv_path,
            station_filter=station_filter,
        )
    except ScheduleParserError as exc:
        raise PredictionServiceError(str(exc)) from exc


# Description:
#   Run a batch of already parsed schedule jobs for one station.
# input:-
#   station_id: station selected for the batch.
#   jobs: list of job dictionaries returned by the schedule parser.
# output:-
#   Runs Prediction for each selected job and may perform network/cache/DB work.
# return:-
#   Dict with station ID, job count, and per-job Prediction results.
def run_prediction_jobs(
    station_id: str,
    jobs: List[Dict],
) -> Dict:
    try:
        normalized_station_id = validate_station_id(station_id)
    except InputValidationError as exc:
        raise PredictionServiceError(str(exc)) from exc

    batch_results = []
    for job in jobs:
        # Schedule rows already store canonical mission/time values, but mission
        # normalization is kept here as a final guard for programmatic callers.
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
