#!/usr/bin/env python3
"""
Prediction service for:
- manual future intervals
- future schedule XML placeholder / best-effort parser

Prediction v1:
- geometry-first
- persistence-aware
- history-assisted
- outputs a heuristic probability (0-100)

Important:
Prediction uses both:
1. mission-specific history (victim mission + interferer NORAD)
2. global history (interferer NORAD)

Mission-specific history has priority.

Probability no longer includes a direct RF metadata bonus.
RF / lookup information is still exposed as context in the outputs.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

from cebreros_rfi.src.core.candidate_helpers import preselect_geometric_candidates
from cebreros_rfi.src.core.operational_context import (
    DEFAULT_STEP_SIZE,
    FINAL_SEP_DEG,
    MAX_REASONABLE_RANGE_KM,
    PRESELECTION_SEP_DEG,
    build_operational_context,
)
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad
from cebreros_rfi.src.feedback_db import (
    HistoryStats,
    get_candidate_history_stats,
    get_candidate_history_stats_for_mission,
)
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
    """
    Use mission history when available, otherwise fall back to global history.
    """
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
    """
    Probability model rationale:

    1. Geometry:
       Smaller separation should strongly increase probability, but not dominate everything.
       We use:
           geometry_component = max(0, 50 - 5 * min_sep_deg)

       Examples:
       - 0.5 deg -> 47.5
       - 1.0 deg -> 45.0
       - 2.0 deg -> 40.0
       - 5.0 deg -> 25.0
       - 10.0 deg -> 0.0

    2. Persistence / duration:
       Repeated close samples should matter a lot more than before.
       We use:
           duration_component = min(30, 3 * close_samples)

       Examples:
       - 1 sample  -> 3
       - 5 samples -> 15
       - 10 samples -> 30

    3. History:
       Mission-specific history has priority.
       If no mission-specific history exists, fall back to 60% of the global-history signal.

    No direct RF bonus is included in probability.
    RF metadata remains visible in outputs, but not as score inflation.
    """
    geometry_component = max(0.0, 50.0 - 5.0 * float(min_sep_deg))
    duration_component = min(30.0, float(close_samples) * 3.0)

    mission_total = int(mission_history_stats.get("total", 0))
    global_total = int(global_history_stats.get("total", 0))

    if mission_total > 0:
        history_component = compute_history_component(mission_history_stats)
    elif global_total > 0:
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

    probability = compute_probability(
        min_sep_deg=item["min_sep_deg"],
        close_samples=item["close_samples"],
        mission_history_stats=mission_history_stats,
        global_history_stats=global_history_stats,
    )

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
            "probability": probability,
            "probability_label": probability_label(probability),
            # Flat aliases keep the consumer contract simple while the detailed
            # mission/global counters remain available for inspection.
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
        results.append(enrich_prediction_candidate(item, victim_mission_id=context.mission_id))
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


def parse_schedule_xml(xml_path: str) -> List[Dict]:
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
