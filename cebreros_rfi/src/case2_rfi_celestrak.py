#!/usr/bin/env python3
"""
Case 2 RFI candidate screening using real antenna pointing and CelesTrak GP data.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

# CelesTrak endpoint used to download GP data in JSON format.
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# UTC timezone used across the whole script.
UTC = timezone.utc

# Station filename format used in the flat-file layout.
FILENAME_RE = re.compile(
    r"^(?P<start_date>\d{8})-(?P<start_time>\d{6})-(?P<end_date>\d{8})-(?P<end_time>\d{6})-(?P<mission>.+)-(?P<axis>AZ|EL)\.csv$",
    re.IGNORECASE,
)

# Pass directory format used in the mission folder layout.
PASS_DIR_RE = re.compile(
    r"^(?P<start_date>\d{8})_(?P<start_time>\d{6})_(?P<end_date>\d{8})_(?P<end_time>\d{6})$"
)

# Supported time formats found in the station CSV files.
TIME_FORMATS = (
    "%H:%M:%S.%f",
    "%H:%M:%S",
    "%H%M%S.%f",
    "%H%M%S",
)


@dataclass(frozen=True)
class StationSample:
    timestamp: datetime
    az_deg: float
    el_deg: float


@dataclass
class CandidateSummary:
    object_name: str
    norad_cat_id: str
    groups: str
    visible_samples: int
    close_samples: int
    min_sep_deg: float
    closest_time_utc: datetime
    sat_az_deg: float
    sat_el_deg: float
    sat_range_km: float
    antenna_az_deg: float
    antenna_el_deg: float
    risk_level: str
    frequency_band: str = "UNKNOWN"


@dataclass(frozen=True)
class IdentificationRequest:
    az_file: Path
    el_file: Path
    groups: List[str]
    max_separation_deg: float = 5.0
    high_risk_sep_deg: float = 1.0
    medium_risk_sep_deg: float = 3.0
    merge_tolerance_seconds: float = 2.0
    station_lat: float = 40.4526889
    station_lon: float = -4.36755
    station_elevation_m: float = 794.0
    incident_time_utc: Optional[datetime] = None
    window_seconds: Optional[int] = None


@dataclass
class IdentificationResult:
    mission_name: str
    pass_start_utc: datetime
    pass_end_utc: datetime
    station_samples: List[StationSample]
    summaries: List[CandidateSummary]
    detailed_rows: List[dict]


class ScriptError(RuntimeError):
    pass


def filter_station_samples_around_incident(
    station_samples,
    incident_time_utc=None,
    window_seconds=None,
):
    # Keep only the samples around the incident when a time window is requested.
    if incident_time_utc is None or window_seconds is None:
        return station_samples

    half_window = timedelta(seconds=window_seconds)
    start_utc = incident_time_utc - half_window
    end_utc = incident_time_utc + half_window

    filtered = [sample for sample in station_samples if start_utc <= sample.timestamp <= end_utc]

    if not filtered:
        raise ScriptError("No station samples found inside the requested incident window.")

    return filtered


def run_identification(request, preloaded_satellites=None):
    # Load the station samples, run the satellite comparison and return the result.
    az_path = Path(request.az_file)
    el_path = Path(request.el_file)

    if not az_path.exists():
        raise ScriptError("AZ file not found: {0}".format(az_path))
    if not el_path.exists():
        raise ScriptError("EL file not found: {0}".format(el_path))

    az_meta = parse_pass_metadata(az_path)
    el_meta = parse_pass_metadata(el_path)

    if az_meta["mission"] != el_meta["mission"]:
        raise ScriptError("AZ and EL files do not belong to the same mission/pass.")

    pass_start_date = datetime.strptime(az_meta["start_date"], "%Y%m%d").date()
    az_samples = read_axis_csv(az_path, pass_start_date)
    el_samples = read_axis_csv(el_path, pass_start_date)

    station_samples = merge_az_el_samples(
        az_samples,
        el_samples,
        tolerance_seconds=request.merge_tolerance_seconds,
    )

    station_samples = filter_station_samples_around_incident(
        station_samples=station_samples,
        incident_time_utc=request.incident_time_utc,
        window_seconds=request.window_seconds,
    )

    mission_name = az_meta["mission"]
    ts = load.timescale()
    station = build_station(
        request.station_lat,
        request.station_lon,
        request.station_elevation_m,
    )

    if preloaded_satellites is None:
        satellites = load_satellites_from_celestrak(request.groups, ts)
    else:
        satellites = preloaded_satellites

    summaries, detailed_rows = analyze_candidates(
        station_samples=station_samples,
        satellites=satellites,
        station=station,
        ts=ts,
        max_separation_deg=request.max_separation_deg,
        high_risk_sep_deg=request.high_risk_sep_deg,
        medium_risk_sep_deg=request.medium_risk_sep_deg,
    )

    return IdentificationResult(
        mission_name=mission_name,
        pass_start_utc=station_samples[0].timestamp,
        pass_end_utc=station_samples[-1].timestamp,
        station_samples=station_samples,
        summaries=summaries,
        detailed_rows=detailed_rows,
    )


def main():
    # Build the identification request from the CLI arguments.
    args = parse_args()

    request = IdentificationRequest(
        az_file=Path(args.az_file),
        el_file=Path(args.el_file),
        groups=[item.strip().upper() for item in args.groups.split(",") if item.strip()],
        max_separation_deg=args.max_separation_deg,
        high_risk_sep_deg=args.high_risk_sep_deg,
        medium_risk_sep_deg=args.medium_risk_sep_deg,
        merge_tolerance_seconds=args.merge_tolerance_seconds,
        station_lat=args.station_lat,
        station_lon=args.station_lon,
        station_elevation_m=args.station_elevation_m,
    )

    result = run_identification(request)

    # Print a short execution summary and save the output files.
    print("\n----------------------------------")
    print("Case 2 candidate screening")
    print("----------------------------------")
    print("Mission/pass     : {0}".format(result.mission_name))
    print(
        "Time window UTC  : {0} -> {1}".format(
            result.pass_start_utc.isoformat(),
            result.pass_end_utc.isoformat(),
        )
    )
    print("Merged samples   : {0}".format(len(result.station_samples)))
    print("CelesTrak groups : {0}".format(",".join(request.groups)))
    print("----------------------------------\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_mission = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.mission_name)
    summary_path = output_dir / "{0}_candidate_summary.csv".format(safe_mission)
    detailed_path = output_dir / "{0}_candidate_samples.csv".format(safe_mission)

    top_summaries = result.summaries[: args.top_n]
    write_summary_csv(summary_path, top_summaries)

    if args.write_detailed_samples:
        write_detailed_csv(detailed_path, result.detailed_rows)

    print("\n----------------------------------")
    print("Analysis summary")
    print("----------------------------------")
    print("Candidate satellites found   : {0}".format(len(result.summaries)))
    print("Summary CSV                  : {0}".format(summary_path))

    if args.write_detailed_samples:
        print("Detailed samples CSV         : {0}".format(detailed_path))

    if top_summaries:
        print("\nTop candidates:")
        for rank, item in enumerate(top_summaries[:10], start=1):
            print(
                "{0:02d}. {1} (NORAD {2}) | min_sep={3:.3f} deg | risk={4} | groups={5} | closest={6}".format(
                    rank,
                    item.object_name,
                    item.norad_cat_id,
                    item.min_sep_deg,
                    item.risk_level,
                    item.groups,
                    item.closest_time_utc.isoformat(),
                )
            )
    else:
        print("No candidates found within the configured separation threshold.")

    print("----------------------------------")
    return 0


def parse_args():
    # Define the command line interface for the case 2 workflow.
    parser = argparse.ArgumentParser(
        description="Rank Case 2 RFI candidates using real antenna AZ/EL and CelesTrak GP data."
    )
    parser.add_argument("--az-file", required=True, help="Path to the station AZ CSV file.")
    parser.add_argument("--el-file", required=True, help="Path to the station EL CSV file.")

    parser.add_argument(
        "--groups",
        default="WEATHER,RESOURCE,PLANET,SPIRE",
        help="Comma-separated CelesTrak groups to query. Default: WEATHER,RESOURCE,PLANET,SPIRE",
    )
    parser.add_argument(
        "--max-separation-deg",
        type=float,
        default=5.0,
        help="Maximum angular separation in degrees. Default: 5.0",
    )
    parser.add_argument(
        "--high-risk-sep-deg",
        type=float,
        default=1.0,
        help="High-risk threshold in degrees. Default: 1.0",
    )
    parser.add_argument(
        "--medium-risk-sep-deg",
        type=float,
        default=3.0,
        help="Medium-risk threshold in degrees. Default: 3.0",
    )
    parser.add_argument(
        "--top-n",
        "--top-N",
        dest="top_n",
        type=int,
        default=50,
        help="Maximum number of summary rows to write. Default: 50",
    )
    parser.add_argument(
        "--merge-tolerance-seconds",
        type=float,
        default=2.0,
        help="Maximum time mismatch when merging AZ and EL. Default: 2.0",
    )
    parser.add_argument(
        "--station-lat",
        type=float,
        default=40.4526889,
        help="Ground-station latitude in degrees. Default: Cebreros latitude.",
    )
    parser.add_argument(
        "--station-lon",
        type=float,
        default=-4.36755,
        help="Ground-station longitude in degrees. Default: Cebreros longitude.",
    )
    parser.add_argument(
        "--station-elevation-m",
        type=float,
        default=794.0,
        help="Ground-station elevation in meters. Default: 794.0",
    )
    parser.add_argument(
        "--output-dir",
        default="results_case2",
        help="Directory where the output CSV files will be written.",
    )
    parser.add_argument(
        "--write-detailed-samples",
        action="store_true",
        help="Write a detailed CSV with all close-approach samples.",
    )
    return parser.parse_args()


def parse_pass_metadata(path):
    # Support both the flat filename layout and the mission/pass folder layout.
    match = FILENAME_RE.match(path.name)
    if match:
        return match.groupdict()

    axis = path.stem.upper()
    parent_match = PASS_DIR_RE.match(path.parent.name)
    mission = path.parent.parent.name.strip()

    if parent_match and axis in {"AZ", "EL"} and mission:
        metadata = parent_match.groupdict()
        metadata["mission"] = mission
        metadata["axis"] = axis
        return metadata

    raise ScriptError(
        "Could not parse pass metadata from filename: {0}. Supported layouts are: "
        "YYYYMMDD-HHMMSS-YYYYMMDD-HHMMSS-NAME-AZ.csv or "
        ".../MISSION/YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS/AZ.csv".format(path.name)
    )


def read_axis_csv(path, start_date):
    # Read one AZ or EL CSV and rebuild absolute UTC timestamps.
    samples = []
    current_date = start_date
    previous_clock = None

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 2:
                continue

            try:
                clock_time = parse_time_string(row[0])
                angle_deg = float(row[1])
            except (ScriptError, ValueError):
                continue

            if previous_clock is not None and clock_time < previous_clock:
                current_date += timedelta(days=1)
            previous_clock = clock_time

            timestamp = datetime.combine(current_date, clock_time, tzinfo=UTC)
            samples.append((timestamp, angle_deg))

    if not samples:
        raise ScriptError("No valid samples found in {0}".format(path))

    return samples


def parse_time_string(value):
    # Accept the time formats seen in the station CSV files.
    cleaned = value.strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue

    raise ScriptError("Unsupported time format in station file: {0!r}".format(value))


def merge_az_el_samples(
    az_samples,
    el_samples,
    tolerance_seconds,
):
    # Merge AZ and EL streams using the nearest timestamps.
    merged = []
    i = 0
    j = 0
    tolerance = timedelta(seconds=tolerance_seconds)

    while i < len(az_samples) and j < len(el_samples):
        az_time, az_deg = az_samples[i]
        el_time, el_deg = el_samples[j]
        delta = az_time - el_time

        if abs(delta) <= tolerance:
            chosen_time = az_time if az_time >= el_time else el_time
            merged.append(StationSample(chosen_time, az_deg, el_deg))
            i += 1
            j += 1
        elif delta < timedelta(0):
            i += 1
        else:
            j += 1

    if not merged:
        raise ScriptError("AZ and EL files could not be merged within the requested tolerance.")

    return merged


def build_station(lat_deg, lon_deg, elevation_m):
    # Build the observer position used by Skyfield.
    return wgs84.latlon(
        latitude_degrees=lat_deg,
        longitude_degrees=lon_deg,
        elevation_m=elevation_m,
    )


def load_satellites_from_celestrak(groups, ts):
    # Download OMM data group by group and deduplicate satellites by NORAD id.
    if not hasattr(EarthSatellite, "from_omm"):
        raise ScriptError(
            "Your Skyfield version does not support EarthSatellite.from_omm(). "
            "Please upgrade Skyfield to >= 1.49."
        )

    satellites_by_catnr = {}

    for group in groups:
        rows = fetch_celestrak_group(group)
        print("Loaded {0} objects from CelesTrak group {1}".format(len(rows), group.upper()))

        for row in rows:
            catnr = str(row.get("NORAD_CAT_ID", "")).strip()
            if not catnr:
                continue

            if catnr not in satellites_by_catnr:
                satellite = EarthSatellite.from_omm(ts, row)
                satellites_by_catnr[catnr] = (satellite, set([group.upper()]), row)
            else:
                satellites_by_catnr[catnr][1].add(group.upper())

    result = []
    for satellite, group_set, row in satellites_by_catnr.values():
        result.append((satellite, sorted(group_set), row))

    print("Total unique satellites loaded: {0}".format(len(result)))
    return result


def fetch_celestrak_group(group):
    # Request one CelesTrak GP group in JSON format.
    params = {"GROUP": group.upper(), "FORMAT": "JSON"}
    url = "{0}?{1}".format(CELESTRAK_GP_URL, urlencode(params))

    with urlopen(url, timeout=60) as response:
        payload = response.read().decode("utf-8")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ScriptError("Invalid JSON received from CelesTrak for group {0}: {1}".format(group, exc))

    if not isinstance(data, list):
        raise ScriptError("Unexpected response format from CelesTrak for group {0}".format(group))

    return data


def analyze_candidates(
    station_samples,
    satellites,
    station,
    ts,
    max_separation_deg,
    high_risk_sep_deg,
    medium_risk_sep_deg,
):
    # Convert the real antenna samples into arrays for vectorized comparisons.
    timestamps = [sample.timestamp for sample in station_samples]
    times = ts.from_datetimes(timestamps)

    antenna_az = np.array([sample.az_deg for sample in station_samples], dtype=float)
    antenna_el = np.array([sample.el_deg for sample in station_samples], dtype=float)

    summaries = []
    detailed_rows = []

    # Propagate every satellite and keep only visible close approaches.
    for satellite, groups, raw_row in satellites:
        difference = satellite - station
        topocentric = difference.at(times)
        alt, az, distance = topocentric.altaz()

        sat_el = np.asarray(alt.degrees, dtype=float)
        sat_az = np.asarray(az.degrees, dtype=float)
        sat_range_km = np.asarray(distance.km, dtype=float)

        visible_mask = sat_el > 0.0
        if not np.any(visible_mask):
            continue

        separation = angular_separation_deg(antenna_az, antenna_el, sat_az, sat_el)
        valid_mask = visible_mask & np.isfinite(separation)
        if not np.any(valid_mask):
            continue

        min_idx = int(np.argmin(np.where(valid_mask, separation, np.inf)))
        min_sep = float(separation[min_idx])
        close_mask = valid_mask & (separation <= max_separation_deg)

        if not np.any(close_mask):
            continue

        summary = CandidateSummary(
            object_name=str(raw_row.get("OBJECT_NAME", satellite.name)).strip() or satellite.name,
            norad_cat_id=str(raw_row.get("NORAD_CAT_ID", "")).strip(),
            groups=",".join(groups),
            visible_samples=int(np.count_nonzero(visible_mask)),
            close_samples=int(np.count_nonzero(close_mask)),
            min_sep_deg=min_sep,
            closest_time_utc=station_samples[min_idx].timestamp,
            sat_az_deg=float(sat_az[min_idx]),
            sat_el_deg=float(sat_el[min_idx]),
            sat_range_km=float(sat_range_km[min_idx]),
            antenna_az_deg=float(antenna_az[min_idx]),
            antenna_el_deg=float(antenna_el[min_idx]),
            risk_level=classify_risk(min_sep, high_risk_sep_deg, medium_risk_sep_deg),
        )
        summaries.append(summary)

        for idx in np.where(close_mask)[0]:
            detailed_rows.append(
                {
                    "object_name": summary.object_name,
                    "norad_cat_id": summary.norad_cat_id,
                    "groups": summary.groups,
                    "time_utc": station_samples[idx].timestamp.isoformat(),
                    "separation_deg": "{0:.6f}".format(separation[idx]),
                    "sat_az_deg": "{0:.6f}".format(sat_az[idx]),
                    "sat_el_deg": "{0:.6f}".format(sat_el[idx]),
                    "sat_range_km": "{0:.3f}".format(sat_range_km[idx]),
                    "antenna_az_deg": "{0:.6f}".format(antenna_az[idx]),
                    "antenna_el_deg": "{0:.6f}".format(antenna_el[idx]),
                    "risk_level": summary.risk_level,
                }
            )

    summaries.sort(key=lambda item: (item.min_sep_deg, -item.close_samples, item.object_name))
    detailed_rows.sort(key=lambda row: (float(row["separation_deg"]), row["time_utc"], row["object_name"]))
    return summaries, detailed_rows


def angular_separation_deg(
    az1_deg,
    el1_deg,
    az2_deg,
    el2_deg,
):
    # Compute angular separation on the sky between two AZ/EL pointings.
    az1 = np.radians(az1_deg)
    el1 = np.radians(el1_deg)
    az2 = np.radians(az2_deg)
    el2 = np.radians(el2_deg)

    cos_sep = (
        np.sin(el1) * np.sin(el2)
        + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    )
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))


def classify_risk(min_sep_deg, high_sep_deg, medium_sep_deg):
    # Convert the minimum separation into a simple risk label.
    if min_sep_deg <= high_sep_deg:
        return "HIGH"
    if min_sep_deg <= medium_sep_deg:
        return "MEDIUM"
    return "LOW"


def write_summary_csv(path, summaries):
    # Write the ranked candidate summary table.
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "object_name",
                "norad_cat_id",
                "groups",
                "visible_samples",
                "close_samples",
                "min_separation_deg",
                "closest_time_utc",
                "sat_az_deg",
                "sat_el_deg",
                "sat_range_km",
                "antenna_az_deg",
                "antenna_el_deg",
                "risk_level",
                "frequency_band",
            ]
        )

        for rank, item in enumerate(summaries, start=1):
            writer.writerow(
                [
                    rank,
                    item.object_name,
                    item.norad_cat_id,
                    item.groups,
                    item.visible_samples,
                    item.close_samples,
                    "{0:.6f}".format(item.min_sep_deg),
                    item.closest_time_utc.isoformat(),
                    "{0:.6f}".format(item.sat_az_deg),
                    "{0:.6f}".format(item.sat_el_deg),
                    "{0:.3f}".format(item.sat_range_km),
                    "{0:.6f}".format(item.antenna_az_deg),
                    "{0:.6f}".format(item.antenna_el_deg),
                    item.risk_level,
                    item.frequency_band,
                ]
            )


def write_detailed_csv(path, rows):
    # Write all close-approach samples when requested.
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        raise SystemExit(1)
