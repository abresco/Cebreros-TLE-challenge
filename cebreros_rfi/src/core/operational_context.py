#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Shared operational configuration and runtime loading helpers.

This keeps station setup and catalog/track loading consistent across the GUI
services and the CLI entrypoints.
"""

from dataclasses import dataclass
from typing import Any, Dict, Set

from skyfield.api import load

from cebreros_rfi.src.config_loader import (
    ConfigError,
    get_station_allowed_bands,
    get_station_ids,
    get_station_skyfield_config,
)
from cebreros_rfi.src.core.geometry_utils import build_station
from cebreros_rfi.src.horizons_target_track import (
    fetch_target_track,
    normalize_station_id,
)
from cebreros_rfi.src.local_candidate_catalog import (
    get_local_catalog_metadata,
    load_local_candidate_catalog,
)
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.update_candidate_catalog import ensure_local_catalog_is_fresh


class OperationalContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class StationConfig:
    lat_deg: float
    lon_deg: float
    elevation_m: float
    allowed_bands: Set[str]


@dataclass
class OperationalContext:
    station_id: str
    mission_id: str
    station_config: StationConfig
    allowed_bands: Set[str]
    catalog_meta: Dict[str, Any]
    target_track_df: Any
    satellites: Any
    station: Any


def get_station_config(station_id: str) -> StationConfig:
    station_key = normalize_station_id(station_id)
    try:
        skyfield_cfg = get_station_skyfield_config(station_key)
        allowed_bands = {
            str(item).upper() for item in get_station_allowed_bands(station_key)
        }
    except ConfigError as exc:
        raise OperationalContextError(str(exc)) from exc

    return StationConfig(
        lat_deg=float(skyfield_cfg["lat_deg"]),
        lon_deg=float(skyfield_cfg["lon_deg"]),
        elevation_m=float(skyfield_cfg["elevation_m"]),
        allowed_bands=allowed_bands,
    )


def get_supported_station_ids():
    try:
        return tuple(get_station_ids())
    except ConfigError as exc:
        raise OperationalContextError(str(exc)) from exc


def build_operational_context(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
    step_size: str,
) -> OperationalContext:
    """
    Load the shared runtime context needed by identification/prediction flows.
    """
    normalized_station_id = normalize_station_id(station_id)
    normalized_mission_id = normalize_mission_name(mission_id)

    try:
        ensure_local_catalog_is_fresh()
        catalog_meta = get_local_catalog_metadata()
        if not catalog_meta["exists"]:
            raise OperationalContextError(
                "Local ACTIVE catalog not found. Run update_candidate_catalog.py first."
            )

        station_config = get_station_config(normalized_station_id)
        target_track_df = fetch_target_track(
            mission_id=normalized_mission_id,
            station_id=normalized_station_id,
            start_time_utc=start_utc,
            stop_time_utc=end_utc,
            step_size=step_size,
        )

        ts = load.timescale()
        satellites = load_local_candidate_catalog(ts)
        station = build_station(
            station_config.lat_deg,
            station_config.lon_deg,
            station_config.elevation_m,
        )

        return OperationalContext(
            station_id=normalized_station_id,
            mission_id=normalized_mission_id,
            station_config=station_config,
            allowed_bands=set(station_config.allowed_bands),
            catalog_meta=catalog_meta,
            target_track_df=target_track_df,
            satellites=satellites,
            station=station,
        )
    except OperationalContextError:
        raise
    except Exception as exc:
        raise OperationalContextError(str(exc))
