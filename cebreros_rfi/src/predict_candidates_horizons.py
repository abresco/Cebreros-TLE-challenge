#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
CLI entrypoint for Prediction v1.
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cebreros_rfi.src.config_loader import get_gui_config
from cebreros_rfi.src.input_validation import (
    InputValidationError,
    validate_mission_id,
    validate_prediction_interval,
    validate_station_id,
)
from cebreros_rfi.src.schedule_parser import ScheduleParserError, parse_schedule_csv

PREDICTION_CSV_FIELDS = [
    "object_name",
    "norad_cat_id",
    "lookup_status",
    "all_satnogs_freqs_mhz",
    "probability",
    "probability_label",
    "min_sep_deg",
    "close_samples",
    "closest_time_utc",
    "possible_rfi_start_utc",
    "possible_rfi_end_utc",
    "sat_az_deg",
    "sat_el_deg",
    "target_az_deg",
    "target_el_deg",
    "sat_range_km",
    "mission_history_confirmed",
    "mission_history_rejected",
    "mission_history_uncertain",
    "mission_history_total",
    "global_history_confirmed",
    "global_history_rejected",
    "global_history_uncertain",
    "global_history_total",
]


# Description:
#   Error raised for user-facing Prediction CLI failures.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class CliError(RuntimeError):
    pass


# Description:
#   Build a filesystem-safe base name for a Prediction CSV output.
# input:-
#   station_id: canonical station ID.
#   mission_id: canonical mission ID.
#   start_utc: prediction start timestamp.
# output:-
#   None.
# return:-
#   String safe for use as a filename stem.
def build_safe_output_name(station_id: str, mission_id: str, start_utc: str) -> str:
    return "{0}_{1}_{2}".format(
        station_id,
        mission_id,
        start_utc.replace(" ", "_").replace(":", ""),
    )


# Description:
#   Write Prediction result rows to a CSV file.
# input:-
#   path: destination CSV path.
#   rows: iterable of Prediction result dictionaries.
# output:-
#   Writes the CSV file when rows are available.
# return:-
#   None.
def write_prediction_csv(path: Path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()

        for row in rows:
            out = dict(row)
            out["all_satnogs_freqs_mhz"] = ",".join(
                "{0:.3f}".format(x) for x in row.get("all_satnogs_freqs_mhz", [])
            )
            writer.writerow({key: out.get(key) for key in PREDICTION_CSV_FIELDS})


# Description:
#   Run the Prediction CLI in manual or schedule mode.
# input:-
#   None. Arguments are read from sys.argv.
# output:-
#   Writes console output and CSV report files.
# return:-
#   None.
def main():
    gui_config = get_gui_config()
    parser = argparse.ArgumentParser(description="Prediction v1 using Horizons + local ACTIVE catalog.")
    parser.add_argument("--station-id", default=str(gui_config.get("default_station", "CEB")))
    parser.add_argument("--mission-id")
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--schedule-csv")
    parser.add_argument("--output-dir", default="results_horizons_prediction")

    args = parser.parse_args()

    try:
        station_id = validate_station_id(args.station_id)
    except InputValidationError as exc:
        raise CliError(str(exc)) from exc

    if args.schedule_csv:
        try:
            schedule_info = parse_schedule_csv(
                csv_path=args.schedule_csv,
                station_filter=station_id,
            )
        except ScheduleParserError as exc:
            raise CliError(str(exc)) from exc

        print("Schedule source:", schedule_info["source_path"])
        print("Usable jobs:", schedule_info["job_count"])
        print("Summary:", schedule_info["summary"])

        from cebreros_rfi.gui.prediction_service import PredictionServiceError, run_prediction_jobs

        try:
            batch = run_prediction_jobs(
                station_id=station_id,
                jobs=schedule_info["jobs"],
            )
        except PredictionServiceError as exc:
            raise CliError(str(exc)) from exc

        print("Prediction batch jobs:", batch["job_count"])

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, job_bundle in enumerate(batch["jobs"], start=1):
            job = job_bundle["job_metadata"]
            pred = job_bundle["prediction"]

            print(
                "[{0}] {1} {2} -> {3} | candidates={4}".format(
                    index,
                    job["mission_id"],
                    job["start_utc"],
                    job["end_utc"],
                    len(pred["results"]),
                )
            )

            safe_name = build_safe_output_name(
                station_id=station_id,
                mission_id=job["mission_id"],
                start_utc=job["start_utc"],
            )
            csv_path = output_dir / "{0}.csv".format(safe_name)
            write_prediction_csv(csv_path, pred["results"])
            print("  CSV:", csv_path.resolve())
        return

    if not args.mission_id or not args.start_utc or not args.end_utc:
        parser.error("Manual mode requires --mission-id, --start-utc and --end-utc.")

    try:
        mission_id = validate_mission_id(args.mission_id)
        validate_prediction_interval(args.start_utc, args.end_utc)
    except InputValidationError as exc:
        raise CliError(str(exc)) from exc

    from cebreros_rfi.gui.prediction_service import PredictionServiceError, run_prediction_interval

    try:
        result = run_prediction_interval(
            station_id=station_id,
            mission_id=mission_id,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
        )
    except PredictionServiceError as exc:
        raise CliError(str(exc)) from exc

    print("Prediction results for", result["mission_id"])
    print("Preselected:", result["preselected_count"])
    print("Lookups:", result["lookups_done"])

    for rank, item in enumerate(result["results"], start=1):
        print(
            "{0:02d}. {1} (NORAD {2}) | probability={3} | sep={4:.3f} deg | slot={5} -> {6}".format(
                rank,
                item["object_name"],
                item["norad_cat_id"],
                item["probability"],
                item["min_sep_deg"],
                item["possible_rfi_start_utc"],
                item["possible_rfi_end_utc"],
            )
        )

    safe_name = build_safe_output_name(
        station_id=station_id,
        mission_id=result["mission_id"],
        start_utc=result["start_utc"],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "{0}.csv".format(safe_name)
    write_prediction_csv(csv_path, result["results"])
    print("CSV:", csv_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except CliError as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        sys.exit(2)
