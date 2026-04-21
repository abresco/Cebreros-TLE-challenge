#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
SatNOGS DB lookup by NORAD ID using the transmitters endpoint directly.

Behavior:
- query transmitters by norad_cat_id
- extract all downlink frequencies
- keep all found frequencies visible
- separately identify whether any frequency matches Cebreros receive bands
  (X: 8400-8500 MHz, Ka: 31800-32300 MHz)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


SATNOGS_API_TRANSMITTERS_URL = "https://db.satnogs.org/api/transmitters/"
CACHE_PATH = Path("cebreros_rfi/data/cache/satnogs_band_cache.json")
USER_AGENT = "CEB-RFI-Identifier/1.0"

# Cebreros receive windows
CEB_X_RANGE_MHZ = (8400.0, 8500.0)
CEB_KA_RANGE_MHZ = (31800.0, 32300.0)


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


def normalize_freq_to_mhz(raw_value) -> Optional[float]:
    """
    SatNOGS usually returns Hz, but keep already-MHz values untouched.
    """
    if raw_value in (None, "", 0):
        return None

    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None

    # SatNOGS values are typically in Hz.
    if numeric > 1_000_000:
        return numeric / 1_000_000.0
    return numeric


def fetch_transmitters_by_norad(norad_cat_id: str) -> List[dict]:
    norad = str(norad_cat_id).strip()

    payload = api_get_json(
        SATNOGS_API_TRANSMITTERS_URL,
        params={
            "norad_cat_id": norad,
            "format": "json",
        },
    )

    if isinstance(payload, dict) and "results" in payload:
        rows = payload.get("results", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise SatnogsLookupError("Unexpected transmitters API response format.")

    filtered = []
    for row in rows:
        row_norad = str(row.get("norad_cat_id", "")).strip()
        if row_norad == norad:
            filtered.append(row)

    return filtered


def extract_frequency_values_mhz(transmitters: List[dict]) -> List[float]:
    freqs_mhz = []

    for item in transmitters:
        for key in ("downlink_low", "downlink_high"):
            value_mhz = normalize_freq_to_mhz(item.get(key))
            if value_mhz is not None:
                freqs_mhz.append(value_mhz)

    return sorted(set(freqs_mhz))


def infer_band_from_frequency_mhz(freq_mhz: float) -> str:
    if CEB_X_RANGE_MHZ[0] <= freq_mhz <= CEB_X_RANGE_MHZ[1]:
        return "X"
    if CEB_KA_RANGE_MHZ[0] <= freq_mhz <= CEB_KA_RANGE_MHZ[1]:
        return "KA"
    return "OUT_OF_CEB_RANGE"


def infer_bands_from_frequencies(freqs_mhz: List[float]) -> List[str]:
    bands = []
    for freq in freqs_mhz:
        band = infer_band_from_frequency_mhz(freq)
        if band not in bands:
            bands.append(band)
    return bands if bands else ["UNKNOWN"]


def filter_ceb_relevant_frequencies(freqs_mhz: List[float]) -> List[float]:
    relevant = []
    for freq in freqs_mhz:
        if CEB_X_RANGE_MHZ[0] <= freq <= CEB_X_RANGE_MHZ[1]:
            relevant.append(freq)
        elif CEB_KA_RANGE_MHZ[0] <= freq <= CEB_KA_RANGE_MHZ[1]:
            relevant.append(freq)
    return sorted(set(relevant))


def lookup_bands_by_norad(norad_cat_id: str, force_refresh: bool = False) -> dict:
    """
    Return cached SatNOGS RF metadata for one NORAD catalog id.
    """
    norad = str(norad_cat_id or "").strip()
    if not norad:
        return {
            "norad_cat_id": None,
            "satnogs_url": None,
            "frequencies_mhz": [],
            "all_frequencies_mhz": [],
            "bands": ["UNKNOWN"],
            "all_bands": ["UNKNOWN"],
            "lookup_status": "missing_norad",
        }

    cache = load_cache()
    if norad in cache and not force_refresh:
        return cache[norad]

    result = {
        "norad_cat_id": norad,
        "satnogs_url": SATNOGS_API_TRANSMITTERS_URL + "?norad_cat_id=" + norad,
        "frequencies_mhz": [],
        "all_frequencies_mhz": [],
        "bands": ["UNKNOWN"],
        "all_bands": ["UNKNOWN"],
        "lookup_status": "not_found",
    }

    try:
        transmitters = fetch_transmitters_by_norad(norad)
        if not transmitters:
            cache[norad] = result
            save_cache(cache)
            return result

        all_freqs_mhz = extract_frequency_values_mhz(transmitters)
        ceb_freqs_mhz = filter_ceb_relevant_frequencies(all_freqs_mhz)

        all_bands = infer_bands_from_frequencies(all_freqs_mhz)
        ceb_bands = infer_bands_from_frequencies(ceb_freqs_mhz)

        if ceb_freqs_mhz:
            lookup_status = "ok"
        elif all_freqs_mhz:
            lookup_status = "no_ceb_band_match"
        else:
            lookup_status = "no_frequency_found"

        result = {
            "norad_cat_id": norad,
            "satnogs_url": SATNOGS_API_TRANSMITTERS_URL + "?norad_cat_id=" + norad,
            "frequencies_mhz": ceb_freqs_mhz,
            "all_frequencies_mhz": all_freqs_mhz,
            "bands": ceb_bands if ceb_freqs_mhz else ["UNKNOWN"],
            "all_bands": all_bands,
            "lookup_status": lookup_status,
        }

    except Exception:
        result["lookup_status"] = "lookup_error"

    cache[norad] = result
    save_cache(cache)
    return result
