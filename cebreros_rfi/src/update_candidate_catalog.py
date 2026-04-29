#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Refresh the local ACTIVE candidate catalog from CelesTrak when possible.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from cebreros_rfi.src.config_loader import get_catalog_refresh_max_age_seconds


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_OUTPUT_PATH = PROJECT_ROOT / "cebreros_rfi" / "data" / "cache" / "full_active_catalog.json"
USER_AGENT = "CEB-RFI-Identifier/1.0"

DEFAULT_MAX_AGE_SECONDS = get_catalog_refresh_max_age_seconds()


# Description:
#   Download one CelesTrak GP group in JSON format.
# input:-
#   group_name: CelesTrak group name, normally ACTIVE.
# output:-
#   Performs an HTTP GET request.
# return:-
#   List of catalog row dictionaries.
def fetch_celestrak_group(group_name):
    url = "{0}?GROUP={1}&FORMAT=JSON".format(
        CELESTRAK_GP_URL,
        str(group_name).strip().upper(),
    )

    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected CelesTrak response format.")

    return data


# Description:
#   Load an existing local catalog payload.
# input:-
#   catalog_path: optional catalog path override.
# output:-
#   Reads local JSON when present.
# return:-
#   Catalog payload dictionary, or None when file does not exist.
def load_existing_payload(catalog_path=None):
    catalog_path = Path(catalog_path or CATALOG_OUTPUT_PATH)

    if not catalog_path.exists():
        return None

    with catalog_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Description:
#   Save fetched catalog rows to the local cache.
# input:-
#   rows: list of CelesTrak catalog rows.
#   catalog_path: optional catalog path override.
# output:-
#   Writes local JSON payload.
# return:-
#   Payload dictionary that was written.
def save_payload(rows, catalog_path=None):
    catalog_path = Path(catalog_path or CATALOG_OUTPUT_PATH)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "CelesTrak",
        "group": "ACTIVE",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "rows": rows,
    }

    with catalog_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return payload


# Description:
#   Check whether a local file is fresh enough.
# input:-
#   path: file path to inspect.
#   max_age_seconds: maximum accepted age.
# output:-
#   None.
# return:-
#   True when file exists and is fresh, otherwise False.
def is_fresh(path, max_age_seconds):
    path = Path(path)
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= max_age_seconds


# Description:
#   Ensure the local CelesTrak ACTIVE catalog can be used.
# input:-
#   catalog_path: optional catalog path override.
#   max_age_seconds: freshness threshold.
#   group_name: CelesTrak group name.
# output:-
#   May read/write local cache and call CelesTrak.
# return:-
#   Status dictionary describing fresh, refreshed, or fallback catalog usage.
def ensure_local_catalog_is_fresh(
    catalog_path=None,
    max_age_seconds=DEFAULT_MAX_AGE_SECONDS,
    group_name="ACTIVE",
):
    catalog_path = Path(catalog_path or CATALOG_OUTPUT_PATH)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    if is_fresh(catalog_path, max_age_seconds=max_age_seconds):
        payload = load_existing_payload(catalog_path)
        return {
            "status": "fresh_local",
            "path": catalog_path,
            "count": payload.get("count", 0) if payload else 0,
            "fetched_at_utc": payload.get("fetched_at_utc") if payload else None,
            "refreshed": False,
        }

    existing_payload = load_existing_payload(catalog_path)

    try:
        rows = fetch_celestrak_group(group_name)
        payload = save_payload(rows, catalog_path=catalog_path)
        return {
            "status": "refreshed",
            "path": catalog_path,
            "count": payload["count"],
            "fetched_at_utc": payload["fetched_at_utc"],
            "refreshed": True,
        }
    except Exception as exc:
        if existing_payload is not None:
            return {
                "status": "stale_local_fallback",
                "path": catalog_path,
                "count": existing_payload.get("count", 0),
                "fetched_at_utc": existing_payload.get("fetched_at_utc"),
                "refreshed": False,
                "warning": str(exc),
            }

        raise RuntimeError(
            "Could not refresh ACTIVE from CelesTrak and no local catalog exists yet: {0}".format(exc)
        )


# Description:
#   Run the catalog refresh helper from CLI.
# input:-
#   None.
# output:-
#   Prints catalog status to stdout.
# return:-
#   None.
def main():
    result = ensure_local_catalog_is_fresh()

    if result["status"] == "fresh_local":
        print("Using fresh local catalog: {0}".format(result["path"].resolve()))
        print("Objects stored: {0}".format(result["count"]))
        return

    if result["status"] == "refreshed":
        print("Refreshed local ACTIVE catalog from CelesTrak.")
        print("Saved local catalog: {0}".format(result["path"].resolve()))
        print("Objects stored: {0}".format(result["count"]))
        return

    if result["status"] == "stale_local_fallback":
        print("WARNING: refresh failed: {0}".format(result.get("warning", "unknown error")))
        print("Using existing local catalog instead: {0}".format(result["path"].resolve()))
        print("Objects stored: {0}".format(result["count"]))
        return


if __name__ == "__main__":
    main()
