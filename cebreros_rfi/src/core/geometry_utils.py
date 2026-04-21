#!/usr/bin/env python3
# Author: Agustí Brescó
# Entity: UPC - EETAC
# Copyright (c) 2026

"""
Core geometry utilities used by the current operational engine.

This module is intentionally separate from validation / legacy workflows.
"""

import numpy as np
from skyfield.api import wgs84


def build_station(lat_deg, lon_deg, elevation_m):
    """
    Build the observer position used by Skyfield.
    """
    return wgs84.latlon(
        latitude_degrees=lat_deg,
        longitude_degrees=lon_deg,
        elevation_m=elevation_m,
    )


def angular_separation_deg(
    az1_deg,
    el1_deg,
    az2_deg,
    el2_deg,
):
    """
    Compute angular separation on the sky between two AZ/EL pointings.
    Supports scalars or numpy arrays.
    """
    az1 = np.radians(az1_deg)
    el1 = np.radians(el1_deg)
    az2 = np.radians(az2_deg)
    el2 = np.radians(el2_deg)

    cos_sep = (
        np.sin(el1) * np.sin(el2)
        + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    )
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))
