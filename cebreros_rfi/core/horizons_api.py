import argparse
from datetime import datetime, timezone
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_COLUMN_CANDIDATES = [
    "Calendar Date (UTC)",
    "Date__(UT)__HR:MN:SC.fff",
    "Date__(UT)__HR:MN:SC",
]
COLUMN_RENAMES = {
    "AZ": "AZ_deg",
    "EL": "EL_deg",
    "delta": "RANGE_au",
    "deldot": "RANGE_rate_km_s",
}
PREFERRED_COLUMNS = [
    "UTC",
    "AZ_deg",
    "EL_deg",
    "RANGE_au",
    "RANGE_rate_km_s",
]


# Diccionario con los nombres de las misiones y sus respectivos comandos.
MISSION_NAMES = {
    "-91": "HERA",
    "-144": "SOLO",
    "-28": "JUICE",
    # Añadir más misiones si es necesario
}


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    height_km: float


CEBREROS = GroundStation(
    name="Cebreros",
    latitude_deg=40.4526889,
    longitude_deg=-4.36755,
    height_km=0.794,
)


def parse_utc_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, UTC_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def normalize_step_size_for_horizons(
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str,
) -> str:
    """
    Convert a user-friendly step size into a Horizons-compatible STEP_SIZE.
    Handles both unit-based step sizes (e.g., "1m", "10s") and unitless interval counts.
    
    Examples:
    '1m'  -> '1m'
    '10s' -> '<computed integer intervals>'
    '1800' -> '<calculated interval count>'
    """
    normalized_step = step_size.strip().lower()

    if normalized_step.isdigit():
        return normalized_step

    if normalized_step.endswith(("d", "h", "m")):
        return normalized_step

    if not normalized_step.endswith("s"):
        raise ValueError(
            f"Unsupported step size '{step_size}'. Use values like '1m', '10m', '1h', or '10s'."
        )

    seconds_text = normalized_step[:-1]
    if not seconds_text.isdigit():
        raise ValueError(
            f"Unsupported step size '{step_size}'. Use values like '1m', '10m', '1h', or '10s'."
        )

    seconds_per_step = int(seconds_text)
    start_dt = parse_utc_datetime(start_time_utc)
    stop_dt = parse_utc_datetime(stop_time_utc)
    total_seconds = int((stop_dt - start_dt).total_seconds())

    if total_seconds <= 0:
        raise ValueError("stop_time_utc must be later than start_time_utc.")

    if seconds_per_step <= 0:
        raise ValueError("Step size in seconds must be positive.")

    if total_seconds % seconds_per_step != 0:
        raise ValueError(
            f"The selected step '{step_size}' does not divide the interval exactly. "
            f"Interval duration is {total_seconds} seconds."
        )

    intervals = total_seconds // seconds_per_step
    return str(intervals)


def build_horizons_params(
    command: str,
    station: GroundStation,
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str,
):
    horizons_step_size = normalize_step_size_for_horizons(
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
        step_size=step_size,
    )

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


def query_horizons(params, timeout_s: int = 60) -> str:
    response = requests.get(HORIZONS_API_URL, params=params, timeout=timeout_s)
    response.raise_for_status()

    payload = response.json()

    error_message = payload.get("error")
    if error_message:
        raise RuntimeError(f"Horizons API error: {error_message}")

    result = payload.get("result")
    if not result:
        raise RuntimeError("Horizons API returned no 'result' field.")

    return result


def extract_soe_block(raw_result: str) -> str:
    start_marker = "$$SOE"
    end_marker = "$$EOE"
    start_index = raw_result.find(start_marker)
    end_index = raw_result.find(end_marker)

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise RuntimeError(
            "Could not find $$SOE/$$EOE block in Horizons response.\n"
            "Check the saved raw response file to see the real Horizons message."
        )

    block_start = start_index + len(start_marker)
    block = raw_result[block_start:end_index].strip()
    if not block:
        raise RuntimeError("Horizons ephemeris block is empty.")

    return block


def parse_horizons_csv_block(csv_block: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_block), skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]

    for column_name in TIME_COLUMN_CANDIDATES:
        if column_name in df.columns:
            df = df.rename(columns={column_name: "UTC"})
            break

    for old_name, new_name in COLUMN_RENAMES.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    if "UTC" in df.columns:
        df["UTC"] = pd.to_datetime(df["UTC"], utc=True, errors="coerce")

    ordered_columns = [column for column in PREFERRED_COLUMNS if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]

    return df[ordered_columns + remaining_columns]


def save_text_file(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_outputs(
    raw_result: str,
    df: pd.DataFrame,
    output_csv: Path,
    save_raw: bool = True,
) -> None:
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if save_raw:
        raw_path = output_csv.with_suffix(".raw.txt")
        raw_path.write_text(raw_result, encoding="utf-8")


def fetch_hera_observer_ephemeris(
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str,
    command: str,
    station: GroundStation = CEBREROS,
):
    params = build_horizons_params(
        command=command,
        station=station,
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
        step_size=step_size,
    )

    raw_result = query_horizons(params)
    csv_block = extract_soe_block(raw_result)
    df = parse_horizons_csv_block(csv_block)

    return df, raw_result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch observer ephemerides from NASA JPL Horizons for individual missions."
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start time in UTC, e.g. '2025-03-19 00:55:00'",
    )

    parser.add_argument(
        "--stop",
        required=True,
        help="Stop time in UTC, e.g. '2025-03-19 01:35:00'",
    )

    parser.add_argument(
        "--step",
        default="10s",
        help="Step size, e.g. '1s', '10s', '1m'. Default: 10s",
    )

    parser.add_argument(
        "--output-dir",
        default="~/Projects/Cebreros-TLE-challenge/cebreros_rfi/data/outputs/horizons/",
        help="Directory to save CSVs",
    )

    parser.add_argument(
        "--command",
        required=True,
        help="Command for mission (e.g., '-91' for HERA)",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    # Map the command number to the mission name using the dictionary
    mission_name = MISSION_NAMES.get(args.command, "Unknown Mission")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Fetching ephemerides for mission: {mission_name} (Command: {args.command})")

    df, raw_result = fetch_hera_observer_ephemeris(
        start_time_utc=args.start,
        stop_time_utc=args.stop,
        step_size=args.step,
        command=args.command,
    )

    output_csv = output_dir / f"{mission_name}_ephemerides.csv"
    save_outputs(
        raw_result=raw_result,
        df=df,
        output_csv=output_csv,
        save_raw=True,
    )

    print(f"CSV saved for mission {mission_name} ({args.command}): {output_csv}")


if __name__ == "__main__":
    main()
