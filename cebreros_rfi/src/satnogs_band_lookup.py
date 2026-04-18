#!/usr/bin/env python3
"""
SatNOGS DB lookup by NORAD ID using the REST API.

Workflow:
1. Query satellites endpoint by NORAD ID
2. Get the SatNOGS internal satellite id
3. Query transmitters endpoint
4. Keep only transmitters linked to that satellite id
5. Infer bands from downlink frequencies
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


SATNOGS_API_SATELLITES_URL = "https://db.satnogs.org/api/satellites/"
SATNOGS_API_TRANSMITTERS_URL = "https://db.satnogs.org/api/transmitters/"
CACHE_PATH = Path("cebreros_rfi/data/cache/satnogs_band_cache.json")
USER_AGENT = "CEB-RFI-Identifier/1.0"


class SatnogsLookupError(RuntimeError):
    pass


def ensure_cache_dir():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_cache() -> Dict[str, dict]:
    ensure_cache_dir()
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(cache: Dict[str, dict]):
    ensure_cache_dir()
    with CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def clear_cache_for_norad(norad_cat_id: str):
    norad = str(norad_cat_id or "").strip()
    if not norad:
        return
    cache = load_cache()
    if norad in cache:
        del cache[norad]
        save_cache(cache)


def api_get_json(url: str, params: Optional[dict] = None):
    response = requests.get(
        url,
        params=params,
        timeout=10,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def infer_band_from_frequency_mhz(freq_mhz: float) -> str:
    if 30 <= freq_mhz < 300:
        return "VHF"
    if 300 <= freq_mhz < 1000:
        return "UHF"
    if 1000 <= freq_mhz < 2000:
        return "L"
    if 2000 <= freq_mhz < 4000:
        return "S"
    if 4000 <= freq_mhz < 8000:
        return "C"
    if 8000 <= freq_mhz < 12000:
        return "X"
    if 12000 <= freq_mhz < 18000:
        return "KU"
    if 18000 <= freq_mhz < 27000:
        return "K"
    if 27000 <= freq_mhz < 40000:
        return "KA"
    return "UNKNOWN"


def infer_bands_from_frequencies(freqs_mhz: List[float]) -> List[str]:
    bands = []
    for freq in freqs_mhz:
        band = infer_band_from_frequency_mhz(freq)
        if band not in bands:
            bands.append(band)
    return bands if bands else ["UNKNOWN"]


def normalize_freq_to_mhz(raw_value) -> Optional[float]:
    if raw_value in (None, "", 0):
        return None

    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None

    # SatNOGS API values are usually in Hz
    if numeric > 1_000_000:
        return numeric / 1_000_000.0
    return numeric


def find_satellite_record_by_norad(norad_cat_id: str) -> Optional[dict]:
    norad = str(norad_cat_id).strip()

    data = api_get_json(
        SATNOGS_API_SATELLITES_URL,
        params={"norad_cat_id": norad},
    )

    if not isinstance(data, list):
        raise SatnogsLookupError("Unexpected satellites API response format.")

    if not data:
        return None

    # Keep exact NORAD match only
    for item in data:
        item_norad = str(item.get("norad_cat_id", "")).strip()
        if item_norad == norad:
            return item

    return None


def fetch_transmitters_for_satellite(satellite_id) -> List[dict]:
    """
    Fetch all transmitters and keep only those belonging to the given satellite id.
    We paginate until there is no next page.
    """
    results = []
    url = SATNOGS_API_TRANSMITTERS_URL
    params = {}

    while url:
        payload = api_get_json(url, params=params)

        if isinstance(payload, dict) and "results" in payload:
            rows = payload.get("results", [])
            next_url = payload.get("next")
        elif isinstance(payload, list):
            rows = payload
            next_url = None
        else:
            raise SatnogsLookupError("Unexpected transmitters API response format.")

        for row in rows:
            if row.get("satellite") == satellite_id:
                results.append(row)

        url = next_url
        params = None

    return results


def extract_frequency_values_mhz(transmitters: List[dict]) -> List[float]:
    freqs_mhz = []

    for item in transmitters:
        for key in ("downlink_low", "downlink_high"):
            value_mhz = normalize_freq_to_mhz(item.get(key))
            if value_mhz is not None:
                freqs_mhz.append(value_mhz)

    return sorted(set(freqs_mhz))


def lookup_bands_by_norad(norad_cat_id: str, force_refresh: bool = False) -> dict:
    norad = str(norad_cat_id or "").strip()
    if not norad:
        return {
            "norad_cat_id": None,
            "satnogs_url": None,
            "satnogs_satellite_id": None,
            "frequencies_mhz": [],
            "bands": ["UNKNOWN"],
            "lookup_status": "missing_norad",
        }

    cache = load_cache()
    if norad in cache and not force_refresh:
        return cache[norad]

    result = {
        "norad_cat_id": norad,
        "satnogs_url": SATNOGS_API_SATELLITES_URL + "?norad_cat_id=" + norad,
        "satnogs_satellite_id": None,
        "frequencies_mhz": [],
        "bands": ["UNKNOWN"],
        "lookup_status": "not_found",
    }

    try:
        satellite = find_satellite_record_by_norad(norad)
        if satellite is None:
            cache[norad] = result
            save_cache(cache)
            return result

        satellite_id = satellite.get("id")
        transmitters = fetch_transmitters_for_satellite(satellite_id)
        freqs_mhz = extract_frequency_values_mhz(transmitters)
        bands = infer_bands_from_frequencies(freqs_mhz)

        result = {
            "norad_cat_id": norad,
            "satnogs_url": SATNOGS_API_SATELLITES_URL + "?norad_cat_id=" + norad,
            "satnogs_satellite_id": satellite_id,
            "frequencies_mhz": freqs_mhz,
            "bands": bands,
            "lookup_status": "ok" if freqs_mhz else "no_frequency_found",
        }

    except Exception:
        result["lookup_status"] = "lookup_error"

    cache[norad] = result
    save_cache(cache)
    return result