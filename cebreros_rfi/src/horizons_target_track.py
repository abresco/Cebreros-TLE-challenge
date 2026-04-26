#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Fetch target mission AZ/EL track from JPL Horizons for a given station and time slot.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import requests

from cebreros_rfi.src.config_loader import (
    ConfigError,
    get_horizons_mission_command,
    get_station_horizons_config,
    normalize_station_id as normalize_station_id_from_config,
)
from cebreros_rfi.src.mission_names import normalize_mission_name

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class GroundStation:
    station_id: str
    mode: str
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    height_km: Optional[float] = None
    center_name: Optional[str] = None


class HorizonsError(RuntimeError):
    pass


def normalize_station_id(station_id: str) -> str:
    return normalize_station_id_from_config(station_id)


def get_station(station_id: str) -> GroundStation:
    station_key = normalize_station_id(station_id)
    try:
        cfg = get_station_horizons_config(station_key)
    except ConfigError as exc:
        raise HorizonsError(str(exc)) from exc

    mode = str(cfg.get("mode", "")).strip().lower()
    if mode == "geodetic":
        return GroundStation(
            station_id=station_key,
            mode=mode,
            latitude_deg=float(cfg["latitude_deg"]),
            longitude_deg=float(cfg["longitude_deg"]),
            height_km=float(cfg["height_km"]),
        )

    if mode == "center_name":
        center_name = str(cfg.get("center_name", "")).strip()
        if not center_name:
            raise HorizonsError(
                "Missing center_name for station {0} in Horizons config.".format(station_key)
            )
        return GroundStation(
            station_id=station_key,
            mode=mode,
            center_name=center_name,
        )

    raise HorizonsError("Unsupported Horizons mode for station {0}: {1}".format(station_key, mode))


def get_mission_command(mission_id: str) -> str:
    mission = normalize_mission_name(mission_id)
    try:
        return get_horizons_mission_command(mission)
    except ConfigError as exc:
        raise HorizonsError(str(exc)) from exc


def parse_utc_datetime(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, UTC_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def normalize_step_size_for_horizons(
    start_time_utc: str,
    stop_time_utc: str,
    step_size: str,
) -> str:
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

        intervals = max(1, int(round(float(total_seconds) / float(seconds))))
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

    params = {
        "format": "json",
        "COMMAND": "'{0}'".format(mission_command),
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "OBJ_DATA": "'YES'",
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

    if station.mode == "geodetic":
        params["CENTER"] = "'coord'"
        params["COORD_TYPE"] = "'GEODETIC'"
        params["SITE_COORD"] = "'{0},{1},{2}'".format(
            station.longitude_deg,
            station.latitude_deg,
            station.height_km,
        )
    elif station.mode == "center_name":
        params["CENTER"] = "'{0}'".format(station.center_name)
    else:
        raise HorizonsError("Unsupported station mode: {0}".format(station.mode))

    return params


def query_horizons(params):
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
