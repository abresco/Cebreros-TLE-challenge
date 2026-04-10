#!/usr/bin/env python3
"""
Benchmark Case 2 identification against the real station incident log.
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from skyfield.api import load

from case2_rfi_celestrak import (
    IdentificationRequest,
    ScriptError,
    load_satellites_from_celestrak,
    run_identification,
)
from mission_names import normalize_mission_name, resolve_mission_dir


# UTC timezone used across the benchmark.
UTC = timezone.utc

# Pass directory format used in the station folders.
PASS_DIR_RE = re.compile(
    r"^(?P<start_date>\d{8})_(?P<start_time>\d{6})_(?P<end_date>\d{8})_(?P<end_time>\d{6})$"
)


def main():
    # Read CLI arguments and prepare the output directory.
    args = build_argument_parser().parse_args()

    rfi_log_path = Path(args.rfi_log).expanduser().resolve()
    station_root = Path(args.station_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    incidents_df = pd.read_excel(rfi_log_path, sheet_name="RFI")
    groups = [item.strip().upper() for item in args.groups.split(",") if item.strip()]

    # Load the satellite catalogue once and reuse it for all incidents.
    print("Loading CelesTrak groups once for the whole benchmark...")
    ts = load.timescale()
    preloaded_satellites = load_satellites_from_celestrak(groups, ts)
    print("Satellite catalogue ready: {0} unique objects".format(len(preloaded_satellites)))

    normalized_rows = []
    incident_rows = []
    candidate_rows = []

    # Process every incident row from the station log.
    for idx, row in incidents_df.iterrows():
        incident_id = "RFI_{0:03d}".format(idx + 1)
        outcome = process_incident_row(
            row=row,
            incident_id=incident_id,
            station_root=station_root,
            args=args,
            groups=groups,
            preloaded_satellites=preloaded_satellites,
        )

        normalized_rows.append(outcome["normalized_row"])
        incident_rows.append(outcome["incident_row"])
        candidate_rows.extend(outcome["candidate_rows"])

    # Save the three benchmark outputs used by the rest of the workflow.
    normalized_output = output_dir / "benchmark_input_normalized.csv"
    incidents_output = output_dir / "benchmark_incidents.csv"
    candidates_output = output_dir / "benchmark_candidates.csv"

    pd.DataFrame(normalized_rows).to_csv(normalized_output, index=False)
    pd.DataFrame(incident_rows).to_csv(incidents_output, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_output, index=False)

    print("")
    print("Saved normalized benchmark input: {0}".format(normalized_output))
    print("Saved incidents benchmark: {0}".format(incidents_output))
    print("Saved candidates benchmark: {0}".format(candidates_output))


def build_argument_parser():
    # Define the command line interface for the benchmark script.
    parser = argparse.ArgumentParser(
        description="Benchmark Case 2 identification against the RFI station log."
    )
    parser.add_argument("--rfi-log", required=True, help="Path to the RFI.xlsx file")
    parser.add_argument("--station-root", required=True, help="Path to station data root")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument(
        "--groups",
        default="WEATHER,RESOURCE,PLANET,SPIRE",
        help="Comma-separated CelesTrak groups",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=120,
        help="Half-window around the incident time in seconds",
    )
    parser.add_argument(
        "--max-separation-deg",
        type=float,
        default=5.0,
        help="Maximum angular separation in degrees",
    )
    parser.add_argument(
        "--high-risk-sep-deg",
        type=float,
        default=1.0,
        help="High-risk threshold in degrees",
    )
    parser.add_argument(
        "--medium-risk-sep-deg",
        type=float,
        default=3.0,
        help="Medium-risk threshold in degrees",
    )
    parser.add_argument(
        "--merge-tolerance-seconds",
        type=float,
        default=2.0,
        help="Merge tolerance for AZ/EL station samples",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N candidates to store per incident",
    )
    return parser


def process_incident_row(
    row,
    incident_id,
    station_root,
    args,
    groups,
    preloaded_satellites,
):
    # Normalize the mission and incident time fields from the Excel row.
    mission_raw = row.get("Mission", "")
    reported_incident = str(row.get("Reported Incident", "") or "")
    investigation = str(row.get("Investigation", "") or "")

    try:
        mission = normalize_mission_name(mission_raw)
        _, mission_dir = resolve_mission_dir(station_root, mission_raw)
        mission_folder = mission_dir.name
        incident_time_utc = parse_incident_time(row.get("Date"), row.get("Time (UTC)"))
        label = classify_label(reported_incident, investigation)
    except Exception as exc:
        return {
            "normalized_row": build_normalization_error_row(
                incident_id=incident_id,
                row=row,
                mission_raw=mission_raw,
                reported_incident=reported_incident,
                investigation=investigation,
                error_message=str(exc),
            ),
            "incident_row": build_incident_error_row(
                incident_id=incident_id,
                reported_incident=reported_incident,
                investigation=investigation,
                error_message=str(exc),
            ),
            "candidate_rows": [],
        }

    normalized_row = {
        "incident_id": incident_id,
        "date_raw": row.get("Date"),
        "time_raw": row.get("Time (UTC)"),
        "mission_raw": mission_raw,
        "mission": mission,
        "mission_folder": mission_folder,
        "incident_time_utc": incident_time_utc.isoformat(),
        "reported_incident": reported_incident,
        "investigation": investigation,
        "label": label,
        "normalization_status": "ok",
        "normalization_error": None,
    }

    print(
        "[{0}] Processing {1} (folder={2}) at {3}".format(
            incident_id,
            mission,
            mission_folder,
            incident_time_utc.isoformat(),
        )
    )

    # Find the station pass that contains the incident timestamp.
    pass_info = find_matching_pass(mission_dir, incident_time_utc)
    incident_row = build_pending_incident_row(
        incident_id=incident_id,
        mission=mission,
        mission_folder=mission_folder,
        incident_time_utc=incident_time_utc,
        label=label,
        reported_incident=reported_incident,
        investigation=investigation,
    )

    if pass_info is None:
        incident_row["status"] = "no_pass_found"
        print("  -> no pass found in folder {0}".format(mission_dir))
        return {
            "normalized_row": normalized_row,
            "incident_row": incident_row,
            "candidate_rows": [],
        }

    incident_row["pass_name"] = pass_info["pass_name"]
    incident_row["pass_start_utc"] = pass_info["pass_start_utc"].isoformat()
    incident_row["pass_end_utc"] = pass_info["pass_end_utc"].isoformat()

    # Run the case 2 identification on the matching pass.
    request = IdentificationRequest(
        az_file=pass_info["az_file"],
        el_file=pass_info["el_file"],
        groups=groups,
        max_separation_deg=args.max_separation_deg,
        high_risk_sep_deg=args.high_risk_sep_deg,
        medium_risk_sep_deg=args.medium_risk_sep_deg,
        merge_tolerance_seconds=args.merge_tolerance_seconds,
        incident_time_utc=incident_time_utc,
        window_seconds=args.window_seconds,
    )

    try:
        result = run_identification(
            request,
            preloaded_satellites=preloaded_satellites,
        )
    except ScriptError as exc:
        incident_row["status"] = "identification_error"
        incident_row["error_message"] = str(exc)
        print("  -> identification error: {0}".format(exc))
        return {
            "normalized_row": normalized_row,
            "incident_row": incident_row,
            "candidate_rows": [],
        }
    except Exception as exc:
        incident_row["status"] = "unexpected_error"
        incident_row["error_message"] = str(exc)
        print("  -> unexpected error: {0}".format(exc))
        return {
            "normalized_row": normalized_row,
            "incident_row": incident_row,
            "candidate_rows": [],
        }

    # Update incident metrics and save the top candidate rows.
    metrics = compute_incident_metrics(result, incident_time_utc)
    incident_row.update(metrics)
    incident_row["status"] = "processed"

    print(
        "  -> processed | candidates={0} | top1_sep={1}".format(
            incident_row["num_candidates"],
            incident_row["top1_min_sep_deg"],
        )
    )

    candidate_rows = []
    for rank, candidate in enumerate(result.summaries[: args.top_n], start=1):
        candidate_rows.append(
            {
                "incident_id": incident_id,
                "mission": mission,
                "mission_folder": mission_folder,
                "incident_time_utc": incident_time_utc.isoformat(),
                "label": label,
                "pass_name": pass_info["pass_name"],
                "rank": rank,
                "object_name": candidate.object_name,
                "norad_cat_id": candidate.norad_cat_id,
                "groups": candidate.groups,
                "visible_samples": candidate.visible_samples,
                "close_samples": candidate.close_samples,
                "min_sep_deg": candidate.min_sep_deg,
                "closest_time_utc": candidate.closest_time_utc.isoformat(),
                "time_offset_sec": abs(
                    (candidate.closest_time_utc - incident_time_utc).total_seconds()
                ),
                "sat_az_deg": candidate.sat_az_deg,
                "sat_el_deg": candidate.sat_el_deg,
                "sat_range_km": candidate.sat_range_km,
                "antenna_az_deg": candidate.antenna_az_deg,
                "antenna_el_deg": candidate.antenna_el_deg,
                "risk_level": candidate.risk_level,
                "frequency_band": candidate.frequency_band,
            }
        )

    return {
        "normalized_row": normalized_row,
        "incident_row": incident_row,
        "candidate_rows": candidate_rows,
    }


def build_normalization_error_row(
    incident_id,
    row,
    mission_raw,
    reported_incident,
    investigation,
    error_message,
):
    # Build the normalization output row for incidents that fail early.
    return {
        "incident_id": incident_id,
        "date_raw": row.get("Date"),
        "time_raw": row.get("Time (UTC)"),
        "mission_raw": mission_raw,
        "mission": None,
        "mission_folder": None,
        "incident_time_utc": None,
        "reported_incident": reported_incident,
        "investigation": investigation,
        "label": None,
        "normalization_status": "error",
        "normalization_error": error_message,
    }


def build_incident_error_row(
    incident_id,
    reported_incident,
    investigation,
    error_message,
):
    # Build the incident output row for normalization failures.
    row = build_pending_incident_row(
        incident_id=incident_id,
        mission=None,
        mission_folder=None,
        incident_time_utc=None,
        label=None,
        reported_incident=reported_incident,
        investigation=investigation,
    )
    row["status"] = "normalization_error"
    row["error_message"] = error_message
    return row


def build_pending_incident_row(
    incident_id,
    mission,
    mission_folder,
    incident_time_utc,
    label,
    reported_incident,
    investigation,
):
    # Create the base incident record before processing the pass.
    return {
        "incident_id": incident_id,
        "mission": mission,
        "mission_folder": mission_folder,
        "incident_time_utc": None if incident_time_utc is None else incident_time_utc.isoformat(),
        "label": label,
        "reported_incident": reported_incident,
        "investigation": investigation,
        "status": "pending",
        "pass_name": None,
        "pass_start_utc": None,
        "pass_end_utc": None,
        "num_candidates": None,
        "top1_object_name": None,
        "top1_norad_cat_id": None,
        "top1_min_sep_deg": None,
        "top1_risk_level": None,
        "top1_groups": None,
        "top1_closest_time_utc": None,
        "top1_time_offset_sec": None,
        "has_candidate_within_1deg": None,
        "has_candidate_within_3deg": None,
        "has_candidate_within_5deg": None,
        "error_message": None,
    }


def compute_incident_metrics(result, incident_time_utc):
    # Extract the benchmark metrics from the identification result.
    if not result.summaries:
        return {
            "num_candidates": 0,
            "top1_object_name": None,
            "top1_norad_cat_id": None,
            "top1_min_sep_deg": None,
            "top1_risk_level": None,
            "top1_groups": None,
            "top1_closest_time_utc": None,
            "top1_time_offset_sec": None,
            "has_candidate_within_1deg": False,
            "has_candidate_within_3deg": False,
            "has_candidate_within_5deg": False,
        }

    top1 = result.summaries[0]
    time_offset_sec = abs((top1.closest_time_utc - incident_time_utc).total_seconds())

    return {
        "num_candidates": len(result.summaries),
        "top1_object_name": top1.object_name,
        "top1_norad_cat_id": top1.norad_cat_id,
        "top1_min_sep_deg": top1.min_sep_deg,
        "top1_risk_level": top1.risk_level,
        "top1_groups": top1.groups,
        "top1_closest_time_utc": top1.closest_time_utc.isoformat(),
        "top1_time_offset_sec": time_offset_sec,
        "has_candidate_within_1deg": any(item.min_sep_deg <= 1.0 for item in result.summaries),
        "has_candidate_within_3deg": any(item.min_sep_deg <= 3.0 for item in result.summaries),
        "has_candidate_within_5deg": any(item.min_sep_deg <= 5.0 for item in result.summaries),
    }


def classify_label(reported_incident, investigation):
    # Map the station log text to a coarse benchmark label.
    text = "{0}\n{1}".format(reported_incident, investigation).lower()

    if "not related to station issues" in text:
        return "discarded"
    if "caused by antenna swap onboard" in text:
        return "discarded"
    if "could be due to" in text:
        return "possible_rfi"
    if "might be related to rfi" in text:
        return "possible_rfi"
    if "high possibility of an rfi" in text:
        return "possible_rfi"
    return "uncertain"


def parse_incident_time(date_value, time_value):
    # Build one UTC datetime from the separate date and time columns.
    date_ts = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date_ts):
        raise ValueError("Invalid incident date: {!r}".format(date_value))

    time_str = str(time_value).strip()
    time_ts = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")

    if pd.isna(time_ts):
        time_ts = pd.to_datetime(time_str, format="%H:%M", errors="coerce")
    if pd.isna(time_ts):
        raise ValueError("Invalid incident time: {!r}".format(time_value))

    return datetime(
        year=date_ts.year,
        month=date_ts.month,
        day=date_ts.day,
        hour=time_ts.hour,
        minute=time_ts.minute,
        second=time_ts.second,
        tzinfo=UTC,
    )


def find_matching_pass(mission_dir, incident_time_utc):
    # Find the pass folder whose time range contains the incident.
    if not mission_dir.exists():
        return None

    pass_dirs = sorted([path for path in mission_dir.iterdir() if path.is_dir()])

    for pass_dir in pass_dirs:
        parsed = parse_pass_dir_name(pass_dir.name)
        if parsed is None:
            continue

        pass_start_utc, pass_end_utc = parsed
        if pass_start_utc <= incident_time_utc <= pass_end_utc:
            az_path = pass_dir / "AZ.csv"
            el_path = pass_dir / "EL.csv"

            if az_path.exists() and el_path.exists():
                return {
                    "pass_dir": pass_dir,
                    "pass_name": pass_dir.name,
                    "pass_start_utc": pass_start_utc,
                    "pass_end_utc": pass_end_utc,
                    "az_file": az_path,
                    "el_file": el_path,
                }

    return None


def parse_pass_dir_name(pass_dir_name):
    # Parse the station pass folder name into start and end UTC datetimes.
    match = PASS_DIR_RE.match(pass_dir_name)
    if not match:
        return None

    data = match.groupdict()
    start_dt = datetime.strptime(
        "{0}_{1}".format(data["start_date"], data["start_time"]),
        "%Y%m%d_%H%M%S",
    ).replace(tzinfo=UTC)

    end_dt = datetime.strptime(
        "{0}_{1}".format(data["end_date"], data["end_time"]),
        "%Y%m%d_%H%M%S",
    ).replace(tzinfo=UTC)

    return start_dt, end_dt


if __name__ == "__main__":
    main()
