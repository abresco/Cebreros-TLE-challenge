#!/usr/bin/env python3
"""
CLI entrypoint for Prediction v1.
"""

import argparse
import csv
from pathlib import Path

from cebreros_rfi.gui.prediction_service import (
    run_prediction_interval,
    run_prediction_schedule,
)


def write_prediction_csv(path: Path, rows):
    if not rows:
        return

    fieldnames = [
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
        "history_confirmed",
        "history_rejected",
        "history_uncertain",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = dict(row)
            out["all_satnogs_freqs_mhz"] = ",".join("{0:.3f}".format(x) for x in row.get("all_satnogs_freqs_mhz", []))
            writer.writerow({key: out.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Prediction v1 using Horizons + local ACTIVE catalog.")
    parser.add_argument("--station-id", default="CEB")
    parser.add_argument("--mission-id")
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--schedule-xml")
    parser.add_argument("--output-dir", default="results_horizons_prediction")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.schedule_xml:
        batch = run_prediction_schedule(
            station_id=args.station_id,
            xml_path=args.schedule_xml,
        )

        print("Prediction batch jobs:", batch["job_count"])
        for index, job in enumerate(batch["jobs"], start=1):
            print(
                "[{0}] {1} {2} -> {3} | candidates={4}".format(
                    index,
                    job["mission_id"],
                    job["start_utc"],
                    job["end_utc"],
                    len(job["results"]),
                )
            )

            safe_name = "{0}_{1}_{2}".format(
                args.station_id,
                job["mission_id"],
                job["start_utc"].replace(" ", "_").replace(":", ""),
            )
            csv_path = output_dir / "{0}.csv".format(safe_name)
            write_prediction_csv(csv_path, job["results"])
            print("  CSV:", csv_path.resolve())
        return

    if not args.mission_id or not args.start_utc or not args.end_utc:
        parser.error("Manual mode requires --mission-id, --start-utc and --end-utc.")

    result = run_prediction_interval(
        station_id=args.station_id,
        mission_id=args.mission_id,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
    )

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

    safe_name = "{0}_{1}_{2}".format(
        args.station_id,
        result["mission_id"],
        result["start_utc"].replace(" ", "_").replace(":", ""),
    )
    csv_path = output_dir / "{0}.csv".format(safe_name)
    write_prediction_csv(csv_path, result["results"])
    print("CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()