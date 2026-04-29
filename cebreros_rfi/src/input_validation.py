#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Reusable validation for user-facing GUI, CLI, and schedule inputs.
"""

from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

from cebreros_rfi.src.config_loader import (
    ConfigError,
    get_mission_ids,
    get_station_ids,
    normalize_station_id,
)
from cebreros_rfi.src.mission_names import normalize_mission_name


USER_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SCHEDULE_DATETIME_FORMATS = (
    "%Y/%m/%d %H:%M:%S",
    USER_DATETIME_FORMAT,
)


# Description:
#   Error raised when user-facing input validation fails.
# input:-
#   Same constructor input as ValueError.
# output:-
#   None.
# return:-
#   Exception instance.
class InputValidationError(ValueError):
    pass


# Description:
#   Format supported values for display in validation errors.
# input:-
#   values: iterable of supported station or mission IDs.
# output:-
#   None.
# return:-
#   Comma-separated string sorted alphabetically.
def _supported_values_text(values: Iterable[str]) -> str:
    return ", ".join(sorted(str(value) for value in values))


# Description:
#   Parse one timestamp with an exact text format.
# input:-
#   value: raw user text.
#   fmt: datetime format string expected by this parser.
#   label: field name used in error messages.
# output:-
#   None.
# return:-
#   Timezone-aware UTC datetime.
def _parse_datetime_exact(value: str, fmt: str, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise InputValidationError("{0} is required.".format(label))

    try:
        parsed = datetime.strptime(text, fmt)
    except ValueError as exc:
        raise InputValidationError(
            "{0} must use format YYYY-MM-DD HH:MM:SS.".format(label)
        ) from exc

    if text != parsed.strftime(fmt):
        raise InputValidationError(
            "{0} must use zero-padded format YYYY-MM-DD HH:MM:SS.".format(label)
        )

    return parsed.replace(tzinfo=timezone.utc)


# Description:
#   Parse a manual GUI/CLI UTC timestamp.
# input:-
#   value: raw datetime text.
#   label: field name used in error messages.
# output:-
#   None.
# return:-
#   Timezone-aware UTC datetime.
def parse_user_datetime(value: str, label: str = "Datetime") -> datetime:
    return _parse_datetime_exact(value, USER_DATETIME_FORMAT, label)


# Description:
#   Convert a datetime into the app's canonical timestamp text.
# input:-
#   value: datetime object, naive values are interpreted as UTC.
# output:-
#   None.
# return:-
#   String formatted as YYYY-MM-DD HH:MM:SS.
def format_user_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime(USER_DATETIME_FORMAT)


# Description:
#   Validate timestamp format and interval ordering.
# input:-
#   start_utc: interval start timestamp text.
#   end_utc: interval end timestamp text.
#   start_label: label used for start errors.
#   end_label: label used for end errors.
# output:-
#   None.
# return:-
#   Tuple with parsed UTC start and end datetimes.
def validate_time_interval(
    start_utc: str,
    end_utc: str,
    start_label: str = "Start UTC",
    end_label: str = "End UTC",
) -> Tuple[datetime, datetime]:
    start_dt = parse_user_datetime(start_utc, start_label)
    end_dt = parse_user_datetime(end_utc, end_label)

    if start_dt >= end_dt:
        raise InputValidationError("{0} must be earlier than {1}.".format(start_label, end_label))

    return start_dt, end_dt


# Description:
#   Validate an Identification interval.
# input:-
#   start_utc: interval start timestamp text.
#   end_utc: interval end timestamp text.
#   now_utc: optional injected current time for tests.
# output:-
#   None.
# return:-
#   Tuple with parsed UTC start and end datetimes.
def validate_identification_interval(
    start_utc: str,
    end_utc: str,
    now_utc: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    start_dt, end_dt = validate_time_interval(start_utc, end_utc)
    now_utc = now_utc or datetime.now(timezone.utc)

    if end_dt > now_utc:
        raise InputValidationError("Identification expects a past interval ending before current UTC time.")

    return start_dt, end_dt


# Description:
#   Validate a Prediction interval.
# input:-
#   start_utc: interval start timestamp text.
#   end_utc: interval end timestamp text.
#   now_utc: optional injected current time for tests.
#   start_label: label used for start errors.
#   end_label: label used for end errors.
# output:-
#   None.
# return:-
#   Tuple with parsed UTC start and end datetimes.
def validate_prediction_interval(
    start_utc: str,
    end_utc: str,
    now_utc: Optional[datetime] = None,
    start_label: str = "Start UTC",
    end_label: str = "End UTC",
) -> Tuple[datetime, datetime]:
    start_dt, end_dt = validate_time_interval(start_utc, end_utc, start_label, end_label)
    now_utc = now_utc or datetime.now(timezone.utc)

    if start_dt < now_utc:
        raise InputValidationError("Prediction expects a future interval starting after current UTC time.")

    return start_dt, end_dt


# Description:
#   Validate and normalize a station ID.
# input:-
#   station_id: raw station ID from GUI, CLI, or schedule.
# output:-
#   None.
# return:-
#   Canonical station ID after applying aliases.
def validate_station_id(station_id: str) -> str:
    station_key = normalize_station_id(station_id)

    try:
        supported = set(get_station_ids())
    except ConfigError as exc:
        raise InputValidationError(str(exc)) from exc

    if station_key not in supported:
        raise InputValidationError(
            "Unsupported station ID '{0}'. Supported stations: {1}.".format(
                str(station_id or "").strip(),
                _supported_values_text(supported),
            )
        )

    return station_key


# Description:
#   Validate and normalize a mission ID.
# input:-
#   mission_id: raw mission ID or mission name variant.
# output:-
#   None.
# return:-
#   Canonical mission ID.
def validate_mission_id(mission_id: str) -> str:
    mission_key = normalize_mission_name(mission_id)

    try:
        supported = set(get_mission_ids())
    except ConfigError as exc:
        raise InputValidationError(str(exc)) from exc

    if mission_key not in supported:
        raise InputValidationError(
            "Unsupported mission ID '{0}'. Supported missions: {1}.".format(
                str(mission_id or "").strip(),
                _supported_values_text(supported),
            )
        )

    return mission_key


# Description:
#   Parse a schedule BOT/EOT timestamp.
# input:-
#   value: raw schedule datetime text.
#   label: field name used in error messages.
# output:-
#   None.
# return:-
#   Canonical timestamp string formatted as YYYY-MM-DD HH:MM:SS.
def parse_schedule_datetime(value: str, label: str = "Schedule datetime") -> str:
    text = str(value or "").strip()
    if not text:
        raise InputValidationError("{0} is required.".format(label))

    for fmt in SCHEDULE_DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if text == parsed.strftime(fmt):
            return format_user_datetime(parsed)

    raise InputValidationError(
        "{0} must use YYYY/MM/DD HH:MM:SS or YYYY-MM-DD HH:MM:SS.".format(label)
    )
