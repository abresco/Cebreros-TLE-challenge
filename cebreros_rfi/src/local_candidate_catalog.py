#!/usr/bin/env python3
"""
Load a locally cached ACTIVE candidate catalog and convert it to Skyfield satellites.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import EarthSatellite


LOCAL_CATALOG_PATH = Path("cebreros_rfi/data/cache/full_active_catalog.json")


class LocalCatalogError(RuntimeError):
    pass


def _build_missing_catalog_metadata(catalog_path: Path) -> dict:
    return {
        "exists": False,
        "path": catalog_path,
        "count": 0,
        "fetched_at_utc": None,
        "age_hours": None,
    }


def _compute_age_hours(fetched_at_raw):
    if not fetched_at_raw:
        return None

    try:
        fetched_at_dt = datetime.fromisoformat(fetched_at_raw.replace("Z", "+00:00"))
    except Exception:
        return None

    now_utc = datetime.now(timezone.utc)
    return (now_utc - fetched_at_dt).total_seconds() / 3600.0


def get_local_catalog_metadata(catalog_path=None):
    """
    Return lightweight metadata without constructing Skyfield objects.
    """
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


def load_local_candidate_catalog(ts, catalog_path=None):
    """
    Load the cached catalog and convert unique NORAD rows into Skyfield objects.
    """
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
