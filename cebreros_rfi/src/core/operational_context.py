#!/usr/bin/env python3
"""
Shared operational configuration and runtime loading helpers.

This keeps station setup and catalog/track loading consistent across the GUI
services and the CLI entrypoints.
"""

from dataclasses import dataclass
from typing import Any, Dict, Set

from skyfield.api import load

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


DEFAULT_STEP_SIZE = "60s"
PRESELECTION_SEP_DEG = 10.0
FINAL_SEP_DEG = 5.0
MAX_REASONABLE_RANGE_KM = 100000.0


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


STATION_CONFIGS = {
    "CEB": StationConfig(
        lat_deg=40.4526889,
        lon_deg=-4.36755,
        elevation_m=794.0,
        allowed_bands={"X", "KA"},
    ),
    "MLG": StationConfig(
        lat_deg=-35.7760083,
        lon_deg=-69.3981972,
        elevation_m=1550.0,
        allowed_bands={"X", "KA"},
    ),
    "NNO": StationConfig(
        lat_deg=-31.03,
        lon_deg=116.11,
        elevation_m=252.0,
        allowed_bands={"X", "KA"},
    ),
}

SUPPORTED_STATION_IDS = tuple(sorted(STATION_CONFIGS.keys()))


def get_station_config(station_id: str) -> StationConfig:
    station_key = normalize_station_id(station_id)
    if station_key not in STATION_CONFIGS:
        raise OperationalContextError("Unsupported station ID: {0}".format(station_id))
    return STATION_CONFIGS[station_key]


def get_supported_station_ids():
    return SUPPORTED_STATION_IDS


def build_operational_context(
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
    step_size: str = DEFAULT_STEP_SIZE,
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
