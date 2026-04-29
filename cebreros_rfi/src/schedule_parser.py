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
- station support comes from config/app_config.yaml
- station aliases such as NNO3 -> NNO are normalized through config
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from cebreros_rfi.src.config_loader import get_schedule_effective_interval
from cebreros_rfi.src.input_validation import (
    InputValidationError,
    parse_schedule_datetime,
    validate_mission_id,
    validate_prediction_interval,
    validate_station_id,
)


SUPPORTED_EFFECTIVE_INTERVAL = "BOT_EOT"
MAX_ROW_ERRORS = 20


# Description:
#   Error raised when a schedule file cannot be parsed into usable jobs.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class ScheduleParserError(RuntimeError):
    pass


# Description:
#   Normalize a station label found in the schedule CSV.
# input:-
#   value: raw station text from the schedule row.
# output:-
#   None.
# return:-
#   Canonical station ID after applying configured aliases.
def normalize_schedule_station(value: str) -> str:
    return validate_station_id(value)


# Description:
#   Convert a CSV cell value into clean text.
# input:-
#   value: raw pandas cell value.
# output:-
#   None.
# return:-
#   Stripped string, or an empty string for null/nan values.
def _safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


# Description:
#   Parse a schedule CSV into usable Prediction jobs.
# input:-
#   csv_path: path to the schedule CSV file.
#   station_filter: station selected by the user.
# output:-
#   Reads the CSV file and skips malformed/unusable rows.
# return:-
#   Dict with source path, station filter, parsed jobs, and summary counters.
def parse_schedule_csv(
    csv_path: str,
    station_filter: str,
) -> Dict:
    # Only BOT/EOT is currently implemented as the effective interval. Keeping
    # this config check explicit avoids silently changing operational meaning.
    effective_interval = get_schedule_effective_interval()
    if effective_interval != SUPPORTED_EFFECTIVE_INTERVAL:
        raise ScheduleParserError(
            "Unsupported configured schedule.effective_interval: {0}".format(effective_interval)
        )

    path = Path(csv_path)
    if not path.exists():
        raise ScheduleParserError("Schedule CSV not found: {0}".format(path))

    try:
        df = pd.read_csv(str(path))
    except Exception as exc:
        raise ScheduleParserError("Could not read schedule CSV: {0}".format(exc)) from exc

    required_columns = ["S/C", "GS", "BOT", "EOT"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ScheduleParserError(
            "Schedule CSV is missing required columns: {0}".format(", ".join(missing))
        )

    try:
        # The station filter comes from the user; accept aliases before matching
        # schedule row station values.
        station_filter = normalize_schedule_station(station_filter)
    except InputValidationError as exc:
        raise ScheduleParserError(str(exc)) from exc

    jobs = []
    skipped_wrong_station = 0
    skipped_unsupported_mission = 0
    skipped_invalid_rows = 0
    row_errors = []

    for row_index, row in df.iterrows():
        # Normalize the row station first so aliases like NNO3 are compared
        # against the canonical station filter.
        gs_raw = _safe_text(row.get("GS"))
        try:
            gs_norm = normalize_schedule_station(gs_raw)
        except InputValidationError as exc:
            skipped_invalid_rows += 1
            if len(row_errors) < MAX_ROW_ERRORS:
                row_errors.append("row {0}: {1}".format(int(row_index), exc))
            continue

        if gs_norm != station_filter:
            skipped_wrong_station += 1
            continue

        # Mission aliases are accepted, but unsupported missions are skipped
        # because schedules may include spacecraft outside the configured demo.
        sc_raw = _safe_text(row.get("S/C"))
        try:
            mission_id = validate_mission_id(sc_raw)
        except InputValidationError:
            skipped_unsupported_mission += 1
            continue

        bot_raw = _safe_text(row.get("BOT"))
        eot_raw = _safe_text(row.get("EOT"))

        try:
            # The configured schedule behavior is explicitly BOT-EOT.
            start_utc = parse_schedule_datetime(bot_raw, "BOT")
            end_utc = parse_schedule_datetime(eot_raw, "EOT")
            # Prediction schedules should describe upcoming intervals; past or
            # reversed rows are skipped and summarized for the operator.
            validate_prediction_interval(start_utc, end_utc, start_label="BOT", end_label="EOT")
        except InputValidationError as exc:
            skipped_invalid_rows += 1
            if len(row_errors) < MAX_ROW_ERRORS:
                row_errors.append("row {0}: {1}".format(int(row_index), exc))
            continue

        # Preserve raw schedule fields that are useful for GUI preview/audit,
        # while also storing canonical start/end timestamps for Prediction.
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
            "row_errors": row_errors,
        },
    }
