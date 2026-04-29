#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Load a locally cached ACTIVE candidate catalog and convert it to Skyfield satellites.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import EarthSatellite


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CATALOG_PATH = PROJECT_ROOT / "cebreros_rfi" / "data" / "cache" / "full_active_catalog.json"


# Description:
#   Error raised when the cached CelesTrak catalog cannot be used.
# input:-
#   Same constructor input as RuntimeError.
# output:-
#   None.
# return:-
#   Exception instance.
class LocalCatalogError(RuntimeError):
    pass


# Description:
#   Build metadata for a missing local catalog.
# input:-
#   catalog_path: expected catalog file path.
# output:-
#   None.
# return:-
#   Metadata dictionary with exists=False.
def _build_missing_catalog_metadata(catalog_path: Path) -> dict:
    return {
        "exists": False,
        "path": catalog_path,
        "count": 0,
        "fetched_at_utc": None,
        "age_hours": None,
    }


# Description:
#   Compute the age of a cached catalog payload.
# input:-
#   fetched_at_raw: ISO timestamp stored in the catalog payload.
# output:-
#   None.
# return:-
#   Age in hours, or None when unavailable.
def _compute_age_hours(fetched_at_raw):
    if not fetched_at_raw:
        return None

    try:
        fetched_at_dt = datetime.fromisoformat(fetched_at_raw.replace("Z", "+00:00"))
    except Exception:
        return None

    now_utc = datetime.now(timezone.utc)
    return (now_utc - fetched_at_dt).total_seconds() / 3600.0


# Description:
#   Read lightweight metadata for the local candidate catalog.
# input:-
#   catalog_path: optional catalog path override.
# output:-
#   Reads catalog JSON if present.
# return:-
#   Metadata dictionary with existence, count, fetch time, and age.
def get_local_catalog_metadata(catalog_path=None):
    catalog_path = Path(catalog_path or LOCAL_CATALOG_PATH)

    if not catalog_path.exists():
        return _build_missing_catalog_metadata(catalog_path)

    with catalog_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    fetched_at_raw = payload.get("fetched_at_utc")

    return {
        "exists": True,
        "path": catalog_path,
        "count": payload.get("count", 0),
        "fetched_at_utc": fetched_at_raw,
        "age_hours": _compute_age_hours(fetched_at_raw),
    }


# Description:
#   Load the cached ACTIVE catalog as Skyfield satellites.
# input:-
#   ts: Skyfield timescale object.
#   catalog_path: optional catalog path override.
# output:-
#   Reads local catalog JSON.
# return:-
#   List of tuples containing EarthSatellite, group labels, and raw row data.
def load_local_candidate_catalog(ts, catalog_path=None):
    catalog_path = Path(catalog_path or LOCAL_CATALOG_PATH)

    if not catalog_path.exists():
        raise LocalCatalogError(
            "Local candidate catalog not found: {0}. "
            "Run update_candidate_catalog.py first.".format(catalog_path)
        )

    with catalog_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise LocalCatalogError("Local candidate catalog is empty or invalid.")

    if not hasattr(EarthSatellite, "from_omm"):
        raise LocalCatalogError(
            "Your Skyfield version does not support EarthSatellite.from_omm()."
        )

    satellites = []
    seen = set()

    for row in rows:
        norad = str(row.get("NORAD_CAT_ID", "")).strip()
        if not norad:
            continue
        if norad in seen:
            continue

        satellite = EarthSatellite.from_omm(ts, row)
        satellites.append((satellite, ["ACTIVE"], row))
        seen.add(norad)

    if not satellites:
        raise LocalCatalogError("No valid satellites could be built from local catalog.")

    return satellites
