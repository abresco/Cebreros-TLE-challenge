#!/usr/bin/env python3
"""
Central configuration loader.

Users should edit config/app_config.yaml instead of touching Python code.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from cebreros_rfi.src.mission_names import normalize_mission_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.yaml"

_CONFIG_CACHE = None


class ConfigError(RuntimeError):
    pass


def _require_mapping(value: Any, key_path: str) -> Dict:
    if not isinstance(value, dict):
        raise ConfigError("Configuration key '{0}' must be a mapping.".format(key_path))
    return value


def _require_list(value: Any, key_path: str) -> List:
    if not isinstance(value, list):
        raise ConfigError("Configuration key '{0}' must be a list.".format(key_path))
    return value


def _require_keys(mapping: Dict, key_path: str, keys: List[str]) -> Dict:
    for key in keys:
        if key not in mapping:
            raise ConfigError("Configuration key '{0}.{1}' is required.".format(key_path, key))
    return mapping


def load_app_config(force_reload: bool = False) -> Dict:
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    if not DEFAULT_CONFIG_PATH.exists():
        raise ConfigError(
            "Configuration file not found: {0}".format(DEFAULT_CONFIG_PATH)
        )

    try:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError("Configuration file is malformed YAML: {0}".format(exc)) from exc
    except OSError as exc:
        raise ConfigError("Could not read configuration file: {0}".format(exc)) from exc

    if not isinstance(config, dict):
        raise ConfigError("Configuration file must contain a YAML dictionary at root.")

    _CONFIG_CACHE = config
    return _CONFIG_CACHE


def get_station_aliases() -> Dict[str, str]:
    config = load_app_config()
    stations_cfg = _require_mapping(config.get("stations", {}), "stations")
    aliases = stations_cfg.get("aliases", {})
    return _require_mapping(aliases, "stations.aliases")


def normalize_station_id(station_id: str) -> str:
    raw = str(station_id or "").strip().upper()
    aliases = get_station_aliases()
    return str(aliases.get(raw, raw)).strip().upper()


def get_station_ids() -> List[str]:
    config = load_app_config()
    stations_cfg = _require_mapping(config.get("stations", {}), "stations")
    definitions = _require_mapping(stations_cfg.get("definitions", {}), "stations.definitions")
    return sorted(list(definitions.keys()))


def get_station_config(station_id: str) -> Dict:
    config = load_app_config()
    station_key = normalize_station_id(station_id)
    stations_cfg = _require_mapping(config.get("stations", {}), "stations")
    definitions = _require_mapping(stations_cfg.get("definitions", {}), "stations.definitions")

    if station_key not in definitions:
        raise ConfigError("Unsupported station ID: {0}".format(station_id))

    return _require_mapping(definitions[station_key], "stations.definitions.{0}".format(station_key))


def get_station_allowed_bands(station_id: str) -> List[str]:
    station_cfg = get_station_config(station_id)
    station_key = normalize_station_id(station_id)
    if "allowed_bands" not in station_cfg:
        raise ConfigError(
            "Configuration key 'stations.definitions.{0}.allowed_bands' is required.".format(station_key)
        )
    return list(
        _require_list(
            station_cfg["allowed_bands"],
            "stations.definitions.{0}.allowed_bands".format(station_key),
        )
    )


def get_station_skyfield_config(station_id: str) -> Dict:
    station_key = normalize_station_id(station_id)
    station_cfg = get_station_config(station_id)
    skyfield_cfg = station_cfg.get("skyfield", {})
    if not skyfield_cfg:
        raise ConfigError("Missing skyfield configuration for station {0}".format(station_id))
    return _require_keys(
        _require_mapping(
            skyfield_cfg,
            "stations.definitions.{0}.skyfield".format(station_key),
        ),
        "stations.definitions.{0}.skyfield".format(station_key),
        ["lat_deg", "lon_deg", "elevation_m"],
    )


def get_station_horizons_config(station_id: str) -> Dict:
    station_key = normalize_station_id(station_id)
    station_cfg = get_station_config(station_id)
    horizons_cfg = station_cfg.get("horizons", {})
    if not horizons_cfg:
        raise ConfigError("Missing Horizons configuration for station {0}".format(station_id))
    horizons_cfg = _require_keys(
        _require_mapping(
            horizons_cfg,
            "stations.definitions.{0}.horizons".format(station_key),
        ),
        "stations.definitions.{0}.horizons".format(station_key),
        ["mode"],
    )
    mode = str(horizons_cfg["mode"]).strip().lower()
    if mode == "geodetic":
        return _require_keys(
            horizons_cfg,
            "stations.definitions.{0}.horizons".format(station_key),
            ["mode", "latitude_deg", "longitude_deg", "height_km"],
        )
    if mode == "center_name":
        return _require_keys(
            horizons_cfg,
            "stations.definitions.{0}.horizons".format(station_key),
            ["mode", "center_name"],
        )
    raise ConfigError(
        "Configuration key 'stations.definitions.{0}.horizons.mode' must be 'geodetic' or 'center_name'.".format(
            station_key
        )
    )


def get_supported_bands() -> List[str]:
    bands = set()
    for station_id in get_station_ids():
        bands.update(str(item).upper() for item in get_station_allowed_bands(station_id))
    return sorted(bands)


def get_mission_ids() -> List[str]:
    config = load_app_config()
    missions_cfg = _require_mapping(config.get("missions", {}), "missions")
    return list(_require_list(missions_cfg.get("supported_ids", []), "missions.supported_ids"))


def get_horizons_mission_command(mission_id: str) -> str:
    config = load_app_config()
    mission_key = normalize_mission_name(mission_id)
    missions_cfg = _require_mapping(config.get("missions", {}), "missions")
    commands = _require_mapping(missions_cfg.get("horizons_commands", {}), "missions.horizons_commands")

    if mission_key not in commands:
        raise ConfigError("Unsupported mission ID for Horizons: {0}".format(mission_id))

    return str(commands[mission_key])


def get_identification_config() -> Dict:
    config = load_app_config()
    return dict(
        _require_keys(
            _require_mapping(config.get("identification", {}), "identification"),
            "identification",
            [
                "step_size",
                "preselection_sep_deg",
                "final_sep_deg",
                "max_reasonable_range_km",
                "max_rf_lookups",
                "top_n_known_output",
                "top_n_unknown_output",
            ],
        )
    )


def get_prediction_config() -> Dict:
    config = load_app_config()
    return dict(
        _require_keys(
            _require_mapping(config.get("prediction", {}), "prediction"),
            "prediction",
            [
                "step_size",
                "preselection_sep_deg",
                "final_sep_deg",
                "max_reasonable_range_km",
                "max_rf_lookups",
                "top_n_output",
                "scoring",
            ],
        )
    )


def get_prediction_scoring_config() -> Dict:
    return dict(
        _require_keys(
            _require_mapping(get_prediction_config().get("scoring", {}), "prediction.scoring"),
            "prediction.scoring",
            [
                "geometry_base",
                "geometry_penalty_per_deg",
                "duration_gain_per_sample",
                "duration_cap",
                "history_confirmed_gain",
                "history_confirmed_cap",
                "history_rejected_penalty",
                "history_rejected_cap",
                "history_uncertain_gain",
                "history_uncertain_cap",
                "global_history_weight",
            ],
        )
    )


def get_catalog_config() -> Dict:
    config = load_app_config()
    return dict(_require_mapping(config.get("catalog", {}), "catalog"))


def get_catalog_refresh_max_age_seconds() -> int:
    cfg = get_catalog_config()
    hours = float(cfg.get("refresh_max_age_hours", 2))
    return int(hours * 3600)


def get_schedule_config() -> Dict:
    config = load_app_config()
    return dict(_require_mapping(config.get("schedule", {}), "schedule"))


def get_schedule_effective_interval() -> str:
    schedule_cfg = get_schedule_config()
    if "effective_interval" not in schedule_cfg:
        raise ConfigError("Configuration key 'schedule.effective_interval' is required.")
    return str(schedule_cfg["effective_interval"]).strip().upper()


def get_gui_config() -> Dict:
    config = load_app_config()
    return dict(
        _require_keys(
            _require_mapping(config.get("gui", {}), "gui"),
            "gui",
            [
                "default_station",
                "default_identification_mission",
                "default_prediction_mission",
                "default_identification_start_utc",
                "default_identification_end_utc",
                "default_prediction_start_utc",
                "default_prediction_end_utc",
            ],
        )
    )
