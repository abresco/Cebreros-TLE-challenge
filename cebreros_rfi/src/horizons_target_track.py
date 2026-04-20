#!/usr/bin/env python3
"""
Fetch target mission AZ/EL track from JPL Horizons for a given station and time slot.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import requests

from cebreros_rfi.src.mission_names import normalize_mission_name

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

MISSION_COMMANDS = {
    "HERA": "-91",
    "JUICE": "-28",
    "SOLO": "-144",
    "BEPI": "-121",
    "MEX1": "-41",
    "EUCL": "-680",
}


@dataclass(frozen=True)
class GroundStation:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    height_km: float


STATIONS: Dict[str, GroundStation] = {
    "CEB": GroundStation(
        station_id="CEB",
        latitude_deg=40.4526907,
        longitude_deg=-4.3675477,
        height_km=0.794132,
    )
}


class HorizonsError(RuntimeError):
    pass


def get_station(station_id: str) -> GroundStation:
    """
    Resolve a supported station id into the coordinates expected by Horizons.
    """
    station_key = str(station_id or "").strip().upper()
    if station_key not in STATIONS:
        raise HorizonsError("Unsupported station ID: {0}".format(station_id))
    return STATIONS[station_key]


def get_mission_command(mission_id: str) -> str:
    """
    Convert the project mission id into the Horizons target command.
    """
    mission = normalize_mission_name(mission_id)
    if mission not in MISSION_COMMANDS:
        raise HorizonsError("Unsupported mission ID for Horizons: {0}".format(mission_id))
    return MISSION_COMMANDS[mission]


def parse_utc_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, UTC_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def normalize_step_size_for_horizons(
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str,
) -> str:
    """
    Accept project-friendly step sizes and map them to the Horizons API format.
    """
    step_size = step_size.strip().lower()

    if step_size.isdigit():
        return step_size

    if step_size.endswith(("d", "h", "m")):
        return step_size

    if step_size.endswith("s"):
        seconds = int(step_size[:-1])
        start_dt = parse_utc_datetime(start_time_utc)
        stop_dt = parse_utc_datetime(stop_time_utc)
        total_seconds = int((stop_dt - start_dt).total_seconds())

        if total_seconds <= 0:
            raise HorizonsError("stop_time_utc must be later than start_time_utc.")
        if seconds <= 0:
            raise HorizonsError("Step size in seconds must be positive.")
        if total_seconds % seconds != 0:
            raise HorizonsError(
                "Step '{0}' does not divide the interval exactly ({1} seconds).".format(
                    step_size,
                    total_seconds,
                )
            )

        intervals = total_seconds // seconds
        return str(intervals)

    raise HorizonsError(
        "Unsupported step size '{0}'. Use values like '30s', '1m', '10m', '1h'.".format(step_size)
    )


def build_horizons_params(
    mission_command: str,
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
        "COMMAND": "'{0}'".format(mission_command),
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "OBJ_DATA": "'YES'",
        "CENTER": "'coord'",
        "COORD_TYPE": "'GEODETIC'",
        "SITE_COORD": "'{0},{1},{2}'".format(
            station.longitude_deg,
            station.latitude_deg,
            station.height_km,
        ),
        "START_TIME": "'{0}'".format(start_time_utc),
        "STOP_TIME": "'{0}'".format(stop_time_utc),
        "STEP_SIZE": "'{0}'".format(horizons_step_size),
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
    """
    Execute the Horizons request and return the raw result block.
    """
    response = requests.get(HORIZONS_API_URL, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise HorizonsError("Horizons API error: {0}".format(data["error"]))

    result = data.get("result")
    if not result:
        raise HorizonsError("Horizons API returned no 'result' field.")

    return result


def extract_soe_block(raw_result: str) -> str:
    match = re.search(r"\$\$SOE(.*?)\$\$EOE", raw_result, flags=re.DOTALL)
    if not match:
        raise HorizonsError("Could not find $$SOE/$$EOE block in Horizons response.")

    block = match.group(1).strip()
    if not block:
        raise HorizonsError("Horizons ephemeris block is empty.")

    return block


def parse_horizons_csv_block(csv_block: str) -> pd.DataFrame:
    """
    Parse the $$SOE/$$EOE CSV payload into the AZ/EL dataframe used downstream.
    """
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
                "AZ_target_deg": az,
                "EL_target_deg": el,
            }
        )

    if not rows:
        raise HorizonsError("Failed to parse Horizons ephemeris rows.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["UTC"]).reset_index(drop=True)
    return df


def fetch_target_track(
    mission_id: str,
    station_id: str,
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str = "30s",
) -> pd.DataFrame:
    """
    End-to-end Horizons fetch for one mission pass interval.
    """
    station = get_station(station_id)
    mission_command = get_mission_command(mission_id)

    params = build_horizons_params(
        mission_command=mission_command,
        station=station,
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
        step_size=step_size,
    )

    raw_result = query_horizons(params)
    csv_block = extract_soe_block(raw_result)
    return parse_horizons_csv_block(csv_block)
