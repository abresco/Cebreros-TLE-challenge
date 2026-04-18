#!/usr/bin/env python3
"""
Refresh the local ACTIVE candidate catalog from CelesTrak when possible.

Behavior:
- If a fresh local catalog already exists, use it and exit.
- If the local catalog is old or missing, try to refresh it from CelesTrak.
- If refresh fails but a local catalog exists, keep using the local catalog.
- If refresh fails and no local catalog exists, raise an error.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CATALOG_OUTPUT_PATH = Path("cebreros_rfi/data/cache/full_active_catalog.json")
USER_AGENT = "CEB-RFI-Identifier/1.0"

# Adjust this as you prefer. 12h is a good default.
MAX_AGE_SECONDS = 12 * 3600


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


def load_existing_payload():
    if not CATALOG_OUTPUT_PATH.exists():
        return None

    with CATALOG_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_payload(rows):
    CATALOG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "CelesTrak",
        "group": "ACTIVE",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "rows": rows,
    }

    with CATALOG_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return payload


def is_fresh(path):
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= MAX_AGE_SECONDS


def main():
    CATALOG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if is_fresh(CATALOG_OUTPUT_PATH):
        payload = load_existing_payload()
        print("Using fresh local catalog: {0}".format(CATALOG_OUTPUT_PATH.resolve()))
        print("Objects stored: {0}".format(payload.get("count", "unknown")))
        return

    existing_payload = load_existing_payload()

    print("Refreshing local ACTIVE catalog from CelesTrak...")
    try:
        rows = fetch_celestrak_group("ACTIVE")
        payload = save_payload(rows)
        print("Saved local catalog: {0}".format(CATALOG_OUTPUT_PATH.resolve()))
        print("Objects stored: {0}".format(payload["count"]))
        return
    except Exception as exc:
        if existing_payload is not None:
            print("WARNING: refresh failed: {0}".format(exc))
            print("Using existing local catalog instead: {0}".format(CATALOG_OUTPUT_PATH.resolve()))
            print("Objects stored: {0}".format(existing_payload.get("count", "unknown")))
            return

        raise RuntimeError(
            "Could not refresh ACTIVE from CelesTrak and no local catalog exists yet: {0}".format(exc)
        )


if __name__ == "__main__":
    main()