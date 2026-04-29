#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Interactive Identification tool:
- target track from Horizons
- external candidates from local full ACTIVE catalog
- candidate bands from SatNOGS
- outputs to console and HTML

This is part of the current operational engine.
It does not import functionality from validation workflows.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Tuple

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
    format_user_datetime,
    parse_user_datetime,
    validate_identification_interval,
    validate_mission_id,
    validate_station_id,
)
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad


@dataclass
# Description:
#   Store geometry-only candidate data for the Identification CLI.
# input:-
#   Dataclass constructor fields.
# output:-
#   None.
# return:-
#   CandidatePreselection instance.
class CandidatePreselection:
    object_name: str
    norad_cat_id: str
    groups: str
    min_sep_deg: float
    closest_time_utc: datetime
    visible_samples: int
    close_samples: int
    sat_az_deg: float
    sat_el_deg: float
    sat_range_km: float
    target_az_deg: float
    target_el_deg: float


@dataclass
# Description:
#   Store enriched Identification candidate data for the CLI.
# input:-
#   Dataclass constructor fields.
# output:-
#   None.
# return:-
#   CandidateResult instance.
class CandidateResult:
    object_name: str
    norad_cat_id: str
    groups: str
    min_sep_deg: float
    closest_time_utc: datetime
    visible_samples: int
    close_samples: int
    sat_az_deg: float
    sat_el_deg: float
    sat_range_km: float
    target_az_deg: float
    target_el_deg: float
    satnogs_bands: List[str]
    satnogs_freqs_mhz: List[float]
    all_satnogs_bands: List[str]
    all_satnogs_freqs_mhz: List[float]
    band_match: str
    score: float
    satnogs_url: str
    lookup_status: str
    has_known_frequency: bool
    manual_review_candidate: bool


# Description:
#   Error raised for user-facing Identification CLI failures.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class ScriptError(RuntimeError):
    pass


# Description:
#   Load Identification runtime settings for the CLI workflow.
# input:-
#   None.
# output:-
#   None.
# return:-
#   Dict with Identification thresholds and output limits.
def get_runtime_config():
    return get_identification_config()


# Description:
#   Prompt for and validate one UTC datetime value from the console.
# input:-
#   prompt_text: message shown to the user.
# output:-
#   Reads from stdin.
# return:-
#   Canonical timestamp string formatted as YYYY-MM-DD HH:MM:SS.
def parse_console_datetime(prompt_text: str) -> str:
    raw_value = input(prompt_text).strip()
    dt = parse_user_datetime(raw_value, prompt_text.split("(", 1)[0].strip() or "UTC datetime")
    return format_user_datetime(dt)


# Description:
#   Prompt for all manual Identification CLI inputs.
# input:-
#   None.
# output:-
#   Reads station, mission, start, and end values from stdin.
# return:-
#   Tuple with validated station ID, mission ID, start UTC, and end UTC.
def prompt_inputs():
    station_id = validate_station_id(input("Station ID: ").strip())
    mission_id = validate_mission_id(input("Mission ID: ").strip())
    start_utc = parse_console_datetime("Start UTC (YYYY-MM-DD HH:MM:SS): ")
    end_utc = parse_console_datetime("End UTC (YYYY-MM-DD HH:MM:SS): ")
    validate_identification_interval(start_utc, end_utc)
    return station_id, mission_id, start_utc, end_utc


# Description:
#   Convert raw geometry preselection dictionaries to CLI dataclasses.
# input:-
#   target_track_df: Horizons target AZ/EL samples.
#   satellites: Skyfield candidate satellites from local catalog.
#   station: Skyfield observer station.
# output:-
#   None.
# return:-
#   List of CandidatePreselection objects sorted by geometry.
def preselect_candidates(target_track_df, satellites, station) -> List[CandidatePreselection]:
    config = get_runtime_config()
    raw_results = preselect_geometric_candidates(
        target_track_df=target_track_df,
        satellites=satellites,
        station=station,
        preselection_sep_deg=float(config["preselection_sep_deg"]),
        final_sep_deg=float(config["final_sep_deg"]),
        max_reasonable_range_km=float(config["max_reasonable_range_km"]),
    )

    return [
        CandidatePreselection(
            object_name=item["object_name"],
            norad_cat_id=item["norad_cat_id"],
            groups=item["groups"],
            min_sep_deg=item["min_sep_deg"],
            closest_time_utc=datetime.fromisoformat(item["closest_time_utc"]),
            visible_samples=item["visible_samples"],
            close_samples=item["close_samples"],
            sat_az_deg=item["sat_az_deg"],
            sat_el_deg=item["sat_el_deg"],
            sat_range_km=item["sat_range_km"],
            target_az_deg=item["target_az_deg"],
            target_el_deg=item["target_el_deg"],
        )
        for item in raw_results
    ]


# Description:
#   Add SatNOGS metadata and ranking score to one CLI candidate.
# input:-
#   item: geometry-only CandidatePreselection object.
#   allowed_bands: RF bands configured for the selected station.
# output:-
#   Reads SatNOGS cache/API.
# return:-
#   CandidateResult object with RF context and score.
def enrich_single_candidate(
    item: CandidatePreselection,
    allowed_bands: set,
) -> CandidateResult:
    satnogs_info = lookup_bands_by_norad(item.norad_cat_id)

    satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
    satnogs_freqs_mhz = satnogs_info.get("frequencies_mhz", [])
    all_satnogs_bands = satnogs_info.get("all_bands", ["UNKNOWN"])
    all_satnogs_freqs_mhz = satnogs_info.get("all_frequencies_mhz", [])
    lookup_status = satnogs_info.get("lookup_status", "unknown")

    band_match = classify_band_match(satnogs_bands, allowed_bands)
    known_frequency = has_known_frequency(lookup_status)

    score = (
        compute_band_penalty(band_match)
        + item.min_sep_deg
        - 0.05 * item.close_samples
    )

    return CandidateResult(
        object_name=item.object_name,
        norad_cat_id=item.norad_cat_id,
        groups=item.groups,
        min_sep_deg=item.min_sep_deg,
        closest_time_utc=item.closest_time_utc,
        visible_samples=item.visible_samples,
        close_samples=item.close_samples,
        sat_az_deg=item.sat_az_deg,
        sat_el_deg=item.sat_el_deg,
        sat_range_km=item.sat_range_km,
        target_az_deg=item.target_az_deg,
        target_el_deg=item.target_el_deg,
        satnogs_bands=satnogs_bands,
        satnogs_freqs_mhz=satnogs_freqs_mhz,
        all_satnogs_bands=all_satnogs_bands,
        all_satnogs_freqs_mhz=all_satnogs_freqs_mhz,
        band_match=band_match,
        score=score,
        satnogs_url=satnogs_info.get("satnogs_url"),
        lookup_status=lookup_status,
        has_known_frequency=known_frequency,
        manual_review_candidate=not known_frequency,
    )


# Description:
#   Build CLI result rankings split by known and unknown RF context.
# input:-
#   preselected_results: geometry candidates.
#   allowed_bands: RF bands configured for the selected station.
# output:-
#   Performs up to the configured number of SatNOGS lookups.
# return:-
#   Tuple with all results, known results, unknown results, and lookup count.
def build_rankings(
    preselected_results: List[CandidatePreselection],
    allowed_bands: set,
) -> Tuple[List[CandidateResult], List[CandidateResult], List[CandidateResult], int]:
    config = get_runtime_config()
    max_rf_lookups = int(config["max_rf_lookups"])
    top_n_known_output = int(config["top_n_known_output"])
    top_n_unknown_output = int(config["top_n_unknown_output"])

    all_results = []
    known_results = []
    unknown_results = []
    lookups_done = 0

    for item in preselected_results:
        if lookups_done >= max_rf_lookups:
            break

        candidate = enrich_single_candidate(item, allowed_bands)
        all_results.append(candidate)
        lookups_done += 1

        if candidate.has_known_frequency:
            known_results.append(candidate)
        else:
            unknown_results.append(candidate)

        if (
            len(known_results) >= top_n_known_output
            and len(unknown_results) >= top_n_unknown_output
        ):
            break

    ranking_key = lambda item: (item.score,) + candidate_sort_key(
        {
            "min_sep_deg": item.min_sep_deg,
            "close_samples": item.close_samples,
            "closest_time_utc": item.closest_time_utc.isoformat(),
        }
    )
    all_results.sort(key=ranking_key)
    known_results.sort(key=ranking_key)
    unknown_results.sort(key=ranking_key)

    return all_results, known_results, unknown_results, lookups_done


# Description:
#   Print one formatted result section to the console.
# input:-
#   title: section title.
#   results: ranked candidates to display.
#   limit: maximum number of rows to print.
# output:-
#   Writes formatted text to stdout.
# return:-
#   None.
def print_result_block(title: str, results: List[CandidateResult], limit: int):
    print("\n----------------------------------")
    print(title)
    print("----------------------------------")

    if not results:
        print("None")
        return

    for rank, item in enumerate(results[:limit], start=1):
        all_freqs_text = ",".join("{0:.3f}".format(x) for x in item.all_satnogs_freqs_mhz[:6]) or "none"

        print(
            "{0:02d}. {1} (NORAD {2}) | all_freqs_mhz={3} | lookup={4} | min_sep={5:.3f} deg | closest={6}".format(
                rank,
                item.object_name,
                item.norad_cat_id,
                all_freqs_text,
                item.lookup_status,
                item.min_sep_deg,
                item.closest_time_utc.isoformat(),
            )
        )


# Description:
#   Run the interactive Identification CLI workflow.
# input:-
#   None.
# output:-
#   Reads stdin, writes console output, and writes an HTML report.
# return:-
#   None.
def main():
    config = get_runtime_config()
    try:
        station_id, mission_id, start_utc, end_utc = prompt_inputs()
    except InputValidationError as exc:
        raise ScriptError(str(exc)) from exc

    try:
        context = build_operational_context(
            station_id=station_id,
            mission_id=mission_id,
            start_utc=start_utc,
            end_utc=end_utc,
            step_size=str(config["step_size"]),
        )
    except Exception as exc:
        raise ScriptError(str(exc))

    allowed_bands = set(context.station_config.allowed_bands)

    print("\nUsing Horizons target track and local ACTIVE candidate catalog...")
    print(
        "Catalog snapshot: count={0}, fetched_at_utc={1}, age_hours={2}".format(
            context.catalog_meta["count"],
            context.catalog_meta["fetched_at_utc"],
            None if context.catalog_meta["age_hours"] is None else round(context.catalog_meta["age_hours"], 2),
        )
    )

    print("Running geometric preselection...")
    preselected = preselect_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
    )
    print("Geometrically suspicious candidates: {0}".format(len(preselected)))

    print("Building RF-aware rankings...")
    all_results, known_results, unknown_results, lookups_done = build_rankings(
        preselected_results=preselected,
        allowed_bands=allowed_bands,
    )

    print_result_block("Known-frequency candidates", known_results, int(config["top_n_known_output"]))
    print_result_block("Unknown-frequency candidates", unknown_results, int(config["top_n_unknown_output"]))
    print_result_block(
        "All checked candidates",
        all_results,
        max(int(config["top_n_known_output"]), int(config["top_n_unknown_output"])),
    )

    output_dir = Path("results_horizons_identification")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_mission = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_"
        for ch in context.mission_id
    )
    html_path = output_dir / "{0}_{1}_{2}.html".format(
        context.station_id,
        safe_mission,
        start_utc.replace(" ", "_").replace(":", ""),
    )

    html_path.write_text(
        "<html><body><h1>Identification report generated.</h1></body></html>",
        encoding="utf-8",
    )

    print("\nHTML report written to: {0}".format(html_path.resolve()))


if __name__ == "__main__":
    try:
        main()
    except ScriptError as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        sys.exit(2)
