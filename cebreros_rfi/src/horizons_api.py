import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# Horizons API endpoint.
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Input datetime format used in this script.
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Mission names mapped to Horizons command codes.
MISSION_COMMANDS = {
    "HERA": "-91",
    "JUICE": "-28",
    "SOLO": "-144",
    "BEPI": "-121",
    "MEX1": "-41",
    "EUCL": "-680",
}

# Folder names normalized to the expected mission names.
MISSION_ALIASES = {
    "JUIC": "JUICE",
}


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    height_km: float


# Default ground station used for Horizons queries.
CEBREROS = GroundStation(
    name="Cebreros",
    latitude_deg=40.4526907,
    longitude_deg=-4.3675477,
    height_km=0.794132,
)


def main():
    # Read CLI arguments and prepare output folders.
    parser = build_argument_parser()
    args = parser.parse_args()

    station_root = Path(args.station_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = []

    # Process each requested mission independently.
    for mission in args.missions:
        mission_dir = station_root / mission
        if not mission_dir.exists():
            print(f"Skipping {mission}: directory not found -> {mission_dir}")
            continue

        summary_df = process_mission(
            mission_dir=mission_dir,
            output_root=output_root,
            step_size=args.step,
            tolerance=args.tolerance,
        )

        if not summary_df.empty:
            all_summary.append(summary_df)

    # Save a global summary when at least one mission produced results.
    if all_summary:
        combined = pd.concat(all_summary, ignore_index=True)
        combined_path = output_root / "all_missions_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined summary: {combined_path}")


def build_argument_parser():
    # Define the command line interface for the comparison workflow.
    parser = argparse.ArgumentParser(
        description="Fetch Horizons ephemerides and compare them against real station AZ/EL CSVs."
    )

    parser.add_argument(
        "--station-root",
        required=True,
        help="Root station directory, e.g. /home/agustibs/Projects/Cebreros-TLE-challenge/cebreros_rfi/data/station",
    )

    parser.add_argument(
        "--missions",
        nargs="+",
        default=["HERA", "JUICE", "SOLO"],
        help="Missions to process. Default: HERA JUICE SOLO",
    )

    parser.add_argument(
        "--step",
        default="2s",
        help="Horizons step size requested by user. Example: 2s. Default: 2s",
    )

    parser.add_argument(
        "--tolerance",
        default="2s",
        help="Time alignment tolerance for merge_asof. Default: 2s",
    )

    parser.add_argument(
        "--output-dir",
        default="~/Projects/Cebreros-TLE-challenge/cebreros_rfi/data/outputs/comparisons",
        help="Output directory for comparisons and summaries",
    )

    return parser


def process_mission(
    mission_dir,
    output_root,
    step_size,
    tolerance,
):
    # Normalize the mission name and prepare the output directory.
    mission_raw = mission_dir.name.upper()
    mission = MISSION_ALIASES.get(mission_raw, mission_raw)
    if mission not in MISSION_COMMANDS:
        raise KeyError(f"No Horizons command configured for mission '{mission}'")

    command = MISSION_COMMANDS[mission]
    mission_output_dir = output_root / mission
    mission_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pass_dirs = sorted([path for path in mission_dir.iterdir() if path.is_dir()])

    if not pass_dirs:
        raise RuntimeError(f"No pass directories found in {mission_dir}")

    # Compare every pass folder against the Horizons prediction.
    for pass_dir in pass_dirs:
        pass_name = pass_dir.name
        print(f"Processing {mission} / {pass_name}")

        start_utc, stop_utc = parse_pass_folder_name(pass_name)
        station_df = load_station_csvs(pass_dir, start_utc=start_utc)
        horizons_df = fetch_mission_observer_ephemeris(
            start_time_utc=start_utc,
            stop_time_utc=stop_utc,
            step_size=step_size,
            command=command,
        )

        comparison_df = compare_pass(
            station_df=station_df,
            horizons_df=horizons_df,
            tolerance=tolerance,
        )

        if comparison_df.empty:
            print(f"  WARNING: no aligned samples for {mission} / {pass_name}")
            continue

        summary_rows.append(compute_summary(comparison_df, mission, pass_name))

        output_csv = mission_output_dir / f"{pass_name}_comparison.csv"
        comparison_df.to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv}")

    # Save one summary file per mission.
    summary_df = pd.DataFrame(summary_rows)
    summary_path = mission_output_dir / f"{mission}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    return summary_df


def compute_summary(metrics_df, mission, pass_name):
    # Aggregate error metrics for one pass.
    return {
        "mission": mission,
        "pass_name": pass_name,
        "n_samples": len(metrics_df),
        "az_mae_deg": metrics_df["AZ_abs_err_deg"].mean(),
        "el_mae_deg": metrics_df["EL_abs_err_deg"].mean(),
        "az_rmse_deg": (metrics_df["AZ_err_deg"] ** 2).mean() ** 0.5,
        "el_rmse_deg": (metrics_df["EL_err_deg"] ** 2).mean() ** 0.5,
        "az_max_abs_err_deg": metrics_df["AZ_abs_err_deg"].max(),
        "el_max_abs_err_deg": metrics_df["EL_abs_err_deg"].max(),
    }


def compare_pass(
    station_df,
    horizons_df,
    tolerance="2s",
):
    # Align both datasets by UTC using the nearest timestamp.
    merged = pd.merge_asof(
        station_df.sort_values("UTC"),
        horizons_df.sort_values("UTC"),
        on="UTC",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    # Keep aligned rows and compute signed and absolute errors.
    merged = merged.dropna(subset=["AZ_horizons_deg", "EL_horizons_deg"]).copy()
    merged["AZ_err_deg"] = merged["AZ_real_deg"] - merged["AZ_horizons_deg"]
    merged["EL_err_deg"] = merged["EL_real_deg"] - merged["EL_horizons_deg"]
    merged["AZ_abs_err_deg"] = merged["AZ_err_deg"].abs()
    merged["EL_abs_err_deg"] = merged["EL_err_deg"].abs()

    return merged


def load_station_csvs(pass_dir, start_utc):
    # Load the real AZ and EL files for one pass.
    az_path = pass_dir / "AZ.csv"
    el_path = pass_dir / "EL.csv"

    if not az_path.exists():
        raise FileNotFoundError(f"AZ.csv not found in {pass_dir}")
    if not el_path.exists():
        raise FileNotFoundError(f"EL.csv not found in {pass_dir}")

    az_df = pd.read_csv(
        az_path,
        header=None,
        names=["time_str", "AZ_real_deg"],
        skipinitialspace=True,
    )
    el_df = pd.read_csv(
        el_path,
        header=None,
        names=["time_str", "EL_real_deg"],
        skipinitialspace=True,
    )

    # Clean time strings and rebuild the full UTC timestamps.
    az_df["time_str"] = az_df["time_str"].astype(str).str.strip()
    el_df["time_str"] = el_df["time_str"].astype(str).str.strip()

    az_df["UTC"] = build_full_utc_from_time_only(
        df=az_df,
        time_col="time_str",
        start_utc=start_utc,
    )
    el_df["UTC"] = build_full_utc_from_time_only(
        df=el_df,
        time_col="time_str",
        start_utc=start_utc,
    )

    # Keep only the useful columns and align AZ with EL by time.
    az_df = az_df[["UTC", "AZ_real_deg"]].dropna().sort_values("UTC").reset_index(drop=True)
    el_df = el_df[["UTC", "EL_real_deg"]].dropna().sort_values("UTC").reset_index(drop=True)

    merged = pd.merge_asof(
        az_df,
        el_df,
        on="UTC",
        direction="nearest",
        tolerance=pd.Timedelta("1s"),
    )

    merged = merged.dropna(subset=["AZ_real_deg", "EL_real_deg"])
    return merged.reset_index(drop=True)


def build_full_utc_from_time_only(
    df,
    time_col,
    start_utc,
):
    # Rebuild full UTC timestamps from HH:MM:SS values only.
    start_dt = parse_utc_datetime(start_utc)
    current_date = start_dt.date()
    previous_time = None
    full_datetimes = []

    # Advance the date when the time wraps after midnight.
    for time_str in df[time_col].astype(str):
        time_str = time_str.strip()
        current_time = datetime.strptime(time_str, "%H:%M:%S").time()

        if previous_time is not None and current_time < previous_time:
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()

        dt = datetime.combine(current_date, current_time).replace(tzinfo=timezone.utc)
        full_datetimes.append(dt)
        previous_time = current_time

    return pd.to_datetime(full_datetimes, utc=True)


def fetch_mission_observer_ephemeris(
    start_time_utc,
    stop_time_utc,
    step_size,
    command,
    station=CEBREROS,
):
    # Build the Horizons request for the selected mission and station.
    params = build_horizons_params(
        command=command,
        station=station,
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
        step_size=step_size,
    )

    # Fetch the raw ephemeris block and convert it to a DataFrame.
    raw_result = query_horizons(params)
    csv_block = extract_soe_block(raw_result)

    try:
        df = parse_horizons_csv_block(csv_block)
    except Exception:
        print("DEBUG: first 20 lines of Horizons SOE block")
        for line in csv_block.splitlines()[:20]:
            print(line)
        raise

    return df


def build_horizons_params(
    command,
    station,
    start_time_utc,
    stop_time_utc,
    step_size,
):
    # Convert the user step size into the Horizons format first.
    horizons_step_size = normalize_step_size_for_horizons(
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
        step_size=step_size,
    )

    # Prepare the complete observer ephemeris query.
    return {
        "format": "json",
        "COMMAND": f"'{command}'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "OBJ_DATA": "'YES'",
        "CENTER": "'coord'",
        "COORD_TYPE": "'GEODETIC'",
        "SITE_COORD": f"'{station.longitude_deg},{station.latitude_deg},{station.height_km}'",
        "START_TIME": f"'{start_time_utc}'",
        "STOP_TIME": f"'{stop_time_utc}'",
        "STEP_SIZE": f"'{horizons_step_size}'",
        "TIME_DIGITS": "'SECONDS'",
        "TIME_TYPE": "'UT'",
        "CAL_FORMAT": "'CAL'",
        "CSV_FORMAT": "'YES'",
        "ANG_FORMAT": "'DEG'",
        "APPARENT": "'AIRLESS'",
        "RANGE_UNITS": "'AU'",
        "SUPPRESS_RANGE_RATE": "'NO'",
        "EXTRA_PREC": "'YES'",
        "QUANTITIES": "'4'",
    }


def query_horizons(params):
    # Call the Horizons API and fail fast on HTTP errors.
    response = requests.get(HORIZONS_API_URL, params=params, timeout=180)
    response.raise_for_status()

    # Validate the JSON payload before using the result block.
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"Horizons API error: {data['error']}")

    result = data.get("result")
    if not result:
        raise RuntimeError("Horizons API returned no 'result' field.")

    return result


def extract_soe_block(raw_result):
    # Extract the data section between the Horizons SOE and EOE markers.
    match = re.search(r"\$\$SOE(.*?)\$\$EOE", raw_result, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Could not find $$SOE/$$EOE block in Horizons response.")

    block = match.group(1).strip()
    if not block:
        raise RuntimeError("Horizons ephemeris block is empty.")

    return block


def parse_horizons_csv_block(csv_block):
    # Parse the Horizons SOE lines into UTC, AZ and EL columns.
    rows = []

    for line in csv_block.splitlines():
        line = line.strip()
        if not line:
            continue

        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue

        utc_str = fields[0]

        try:
            az = float(fields[3])
            el = float(fields[4])
        except ValueError:
            continue

        rows.append(
            {
                "UTC": pd.to_datetime(utc_str, utc=True, errors="coerce"),
                "AZ_horizons_deg": az,
                "EL_horizons_deg": el,
            }
        )

    # Reject empty results and drop rows with invalid timestamps.
    if not rows:
        raise RuntimeError("Failed to parse Horizons ephemeris rows.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["UTC"]).reset_index(drop=True)
    return df


def parse_pass_folder_name(folder_name):
    # Decode the pass folder name into start and stop UTC strings.
    match = re.match(
        r"^(\d{8})_(\d{6})_(\d{8})_(\d{6})$",
        folder_name,
    )
    if not match:
        raise ValueError(f"Invalid pass folder name: {folder_name}")

    start_date, start_time, stop_date, stop_time = match.groups()

    start_utc = datetime.strptime(
        f"{start_date}_{start_time}",
        "%Y%m%d_%H%M%S",
    ).strftime(UTC_DATETIME_FORMAT)

    stop_utc = datetime.strptime(
        f"{stop_date}_{stop_time}",
        "%Y%m%d_%H%M%S",
    ).strftime(UTC_DATETIME_FORMAT)

    return start_utc, stop_utc


def normalize_step_size_for_horizons(
    start_time_utc,
    stop_time_utc,
    step_size,
):
    # Normalize the input before checking the supported formats.
    step_size = step_size.strip().lower()

    if step_size.isdigit():
        return step_size

    if step_size.endswith(("d", "h", "m")):
        return step_size

    # Convert second-based steps into Horizons interval counts.
    if step_size.endswith("s"):
        seconds = int(step_size[:-1])
        start_dt = parse_utc_datetime(start_time_utc)
        stop_dt = parse_utc_datetime(stop_time_utc)
        total_seconds = int((stop_dt - start_dt).total_seconds())

        if total_seconds <= 0:
            raise ValueError("stop_time_utc must be later than start_time_utc.")

        if seconds <= 0:
            raise ValueError("Step size in seconds must be positive.")

        if total_seconds % seconds != 0:
            raise ValueError(
                f"The selected step '{step_size}' does not divide the interval exactly. "
                f"Interval duration is {total_seconds} seconds."
            )

        intervals = total_seconds // seconds
        return str(intervals)

    raise ValueError(
        f"Unsupported step size '{step_size}'. Use values like '2s', '10s', '1m', '10m', or a plain integer."
    )


def parse_utc_datetime(dt_str):
    # Convert a UTC string into a timezone-aware datetime.
    return datetime.strptime(dt_str, UTC_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    main()
