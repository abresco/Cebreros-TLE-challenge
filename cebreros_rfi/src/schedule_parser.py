#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Schedule parsers for Prediction inputs.

Current supported format:
- CSV schedules with columns like:
  S/C, GS, DOY, BOA, BOT, EOT, EOA, Duration, Reference, Comment

Rules:
- BOT/EOT is used as the effective prediction interval
- station filter currently supports CEB, MLG, NNO
- NNO3 is treated as NNO
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from cebreros_rfi.src.config_loader import get_mission_ids, get_schedule_effective_interval
from cebreros_rfi.src.core.operational_context import get_supported_station_ids
from cebreros_rfi.src.horizons_target_track import normalize_station_id
from cebreros_rfi.src.mission_names import normalize_mission_name


SUPPORTED_PREDICTION_STATIONS = set(get_supported_station_ids())
SUPPORTED_MISSION_IDS = set(get_mission_ids())
SUPPORTED_EFFECTIVE_INTERVAL = "BOT_EOT"


class ScheduleParserError(RuntimeError):
    pass


def normalize_schedule_station(value: str) -> str:
    return normalize_station_id(value)


def _parse_schedule_datetime(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty datetime value.")
    dt = pd.to_datetime(text, format="%Y/%m/%d %H:%M:%S", utc=False)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_schedule_csv(
    csv_path: str,
    station_filter: str,
) -> Dict:
    effective_interval = get_schedule_effective_interval()
    if effective_interval != SUPPORTED_EFFECTIVE_INTERVAL:
        raise ScheduleParserError(
            "Unsupported configured schedule.effective_interval: {0}".format(effective_interval)
        )

    path = Path(csv_path)
    if not path.exists():
        raise ScheduleParserError("Schedule CSV not found: {0}".format(path))

    df = pd.read_csv(str(path))
    required_columns = ["S/C", "GS", "BOT", "EOT"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ScheduleParserError(
            "Schedule CSV is missing required columns: {0}".format(", ".join(missing))
        )

    station_filter = normalize_schedule_station(station_filter)
    if station_filter not in SUPPORTED_PREDICTION_STATIONS:
        raise ScheduleParserError(
            "Unsupported station filter for Prediction: {0}".format(station_filter)
        )

    jobs = []
    skipped_wrong_station = 0
    skipped_unsupported_mission = 0
    skipped_invalid_rows = 0

    for row_index, row in df.iterrows():
        gs_raw = _safe_text(row.get("GS"))
        gs_norm = normalize_schedule_station(gs_raw)

        if gs_norm != station_filter:
            skipped_wrong_station += 1
            continue

        sc_raw = _safe_text(row.get("S/C"))
        mission_id = normalize_mission_name(sc_raw)

        if mission_id not in SUPPORTED_MISSION_IDS:
            skipped_unsupported_mission += 1
            continue

        bot_raw = _safe_text(row.get("BOT"))
        eot_raw = _safe_text(row.get("EOT"))

        try:
            start_utc = _parse_schedule_datetime(bot_raw)
            end_utc = _parse_schedule_datetime(eot_raw)
        except Exception:
            skipped_invalid_rows += 1
            continue

        jobs.append(
            {
                "job_id": "row_{0}".format(int(row_index)),
                "mission_id": mission_id,
                "sc_raw": sc_raw,
                "station_id": gs_norm,
                "gs_raw": gs_raw,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "boa_utc": _safe_text(row.get("BOA")),
                "bot_utc": bot_raw,
                "eot_utc": eot_raw,
                "eoa_utc": _safe_text(row.get("EOA")),
                "duration": _safe_text(row.get("Duration")),
                "reference": _safe_text(row.get("Reference")),
                "comment": _safe_text(row.get("Comment")),
            }
        )

    return {
        "source_path": str(path),
        "station_filter": station_filter,
        "job_count": len(jobs),
        "jobs": jobs,
        "summary": {
            "total_rows": int(len(df)),
            "usable_jobs": int(len(jobs)),
            "skipped_wrong_station": int(skipped_wrong_station),
            "skipped_unsupported_mission": int(skipped_unsupported_mission),
            "skipped_invalid_rows": int(skipped_invalid_rows),
        },
    }
