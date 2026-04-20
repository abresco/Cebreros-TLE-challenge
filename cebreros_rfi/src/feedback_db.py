#!/usr/bin/env python3
"""
SQLite database for Identification feedback and future Prediction scoring.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


DB_PATH = Path("cebreros_rfi/data/db/rfi_feedback.sqlite3")
DbPath = Optional[Union[str, Path]]
CandidatePayload = Mapping[str, Any]
HistoryStats = Dict[str, int]


def resolve_db_path(db_path: DbPath = None) -> Path:
    return Path(db_path or DB_PATH)


def ensure_db_dir(db_path: DbPath = None):
    """
    Create the SQLite parent directory on demand.
    """
    resolve_db_path(db_path).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: DbPath = None):
    """
    Open a connection to the feedback database, creating directories if needed.
    """
    ensure_db_dir(db_path)
    path = str(resolve_db_path(db_path))
    return sqlite3.connect(path)


def init_db(db_path: DbPath = None):
    """
    Ensure the feedback table and lookup indexes exist.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identification_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                station_id TEXT NOT NULL,
                mission_id TEXT NOT NULL,
                pass_start_utc TEXT,
                pass_end_utc TEXT,
                rfi_start_utc TEXT,
                rfi_end_utc TEXT,
                norad_cat_id TEXT,
                object_name TEXT,
                lookup_status TEXT,
                all_freqs_mhz_json TEXT,
                min_sep_deg REAL,
                closest_time_utc TEXT,
                probability REAL,
                user_label TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_norad
            ON identification_feedback (norad_cat_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_mission
            ON identification_feedback (mission_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_mission_norad
            ON identification_feedback (mission_id, norad_cat_id)
            """
        )


def record_identification_feedback(
    station_id: str,
    mission_id: str,
    pass_start_utc: str,
    pass_end_utc: str,
    rfi_start_utc: str,
    rfi_end_utc: str,
    candidate: CandidatePayload,
    user_label: str,
    probability: float = None,
    notes: str = "",
    db_path: DbPath = None,
):
    init_db(db_path)

    created_at_utc = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO identification_feedback (
                created_at_utc,
                station_id,
                mission_id,
                pass_start_utc,
                pass_end_utc,
                rfi_start_utc,
                rfi_end_utc,
                norad_cat_id,
                object_name,
                lookup_status,
                all_freqs_mhz_json,
                min_sep_deg,
                closest_time_utc,
                probability,
                user_label,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_utc,
                station_id,
                mission_id,
                pass_start_utc,
                pass_end_utc,
                rfi_start_utc,
                rfi_end_utc,
                str(candidate.get("norad_cat_id", "")).strip(),
                str(candidate.get("object_name", "")).strip(),
                str(candidate.get("lookup_status", "")).strip(),
                json.dumps(candidate.get("all_satnogs_freqs_mhz", [])),
                float(candidate.get("min_sep_deg", 0.0)) if candidate.get("min_sep_deg") is not None else None,
                candidate.get("closest_time_utc"),
                probability,
                user_label,
                notes,
            ),
        )


def _empty_stats() -> HistoryStats:
    return {
        "confirmed": 0,
        "rejected": 0,
        "uncertain": 0,
        "total": 0,
    }


def _rows_to_counts(rows: Iterable) -> HistoryStats:
    """
    Convert grouped SQL rows into the normalized stats structure used elsewhere.
    """
    counts = _empty_stats()

    for label, count in rows:
        key = str(label or "").strip().lower()
        if key in counts:
            counts[key] = int(count)

    counts["total"] = counts["confirmed"] + counts["rejected"] + counts["uncertain"]
    return counts


def get_candidate_history_stats(norad_cat_id: str, db_path: DbPath = None) -> HistoryStats:
    """
    Global candidate history, regardless of victim mission.
    """
    init_db(db_path)

    norad = str(norad_cat_id or "").strip()
    if not norad:
        return _empty_stats()

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT user_label, COUNT(*)
            FROM identification_feedback
            WHERE norad_cat_id = ?
            GROUP BY user_label
            """,
            (norad,),
        ).fetchall()

    return _rows_to_counts(rows)


def get_candidate_history_stats_for_mission(
    mission_id: str,
    norad_cat_id: str,
    db_path: DbPath = None,
) -> HistoryStats:
    """
    Mission-specific candidate history:
    how often this NORAD interfered with this ESA victim mission.
    """
    init_db(db_path)

    mission = str(mission_id or "").strip().upper()
    norad = str(norad_cat_id or "").strip()

    if not mission or not norad:
        return _empty_stats()

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT user_label, COUNT(*)
            FROM identification_feedback
            WHERE mission_id = ?
              AND norad_cat_id = ?
            GROUP BY user_label
            """,
            (mission, norad),
        ).fetchall()

    return _rows_to_counts(rows)
