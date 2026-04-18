#!/usr/bin/env python3
"""
Interactive Identification tool:
- target track from Horizons
- external candidates from local full ACTIVE catalog
- candidate bands from SatNOGS
- outputs to console and HTML
"""

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from skyfield.api import load

from case2_rfi_celestrak import (
    ScriptError,
    angular_separation_deg,
    build_station,
)
from horizons_target_track import fetch_target_track
from local_candidate_catalog import (
    get_local_catalog_metadata,
    load_local_candidate_catalog,
)
from mission_names import normalize_mission_name
from satnogs_band_lookup import lookup_bands_by_norad


UTC = timezone.utc

STATION_CONFIG = {
    "CEB": {
        "lat_deg": 40.4526889,
        "lon_deg": -4.36755,
        "elevation_m": 794.0,
        "allowed_bands": {"X", "KA"},
    }
}

DEFAULT_STEP_SIZE = "60s"
PRESELECTION_SEP_DEG = 10.0
FINAL_SEP_DEG = 5.0
TOP_N_BAND_LOOKUP = 10
TOP_N_OUTPUT = 20
MAX_REASONABLE_RANGE_KM = 100000.0


@dataclass
class CandidatePreselection:
    object_name: str
    norad_cat_id: str
    groups: str
    min_sep_deg: float
    closest_time_utc: datetime
    visible_samples: int
    close_samples: int
    sat_az_deg: float
    sat_el_deg: float
    sat_range_km: float
    target_az_deg: float
    target_el_deg: float


@dataclass
class CandidateResult:
    object_name: str
    norad_cat_id: str
    groups: str
    min_sep_deg: float
    closest_time_utc: datetime
    visible_samples: int
    close_samples: int
    sat_az_deg: float
    sat_el_deg: float
    sat_range_km: float
    target_az_deg: float
    target_el_deg: float
    satnogs_bands: List[str]
    satnogs_freqs_mhz: List[float]
    band_match: str
    score: float
    satnogs_url: str
    lookup_status: str
    satnogs_satellite_id: object


def parse_console_datetime(prompt_text: str) -> str:
    raw_value = input(prompt_text).strip()
    dt = datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def prompt_inputs():
    station_id = input("Station ID: ").strip().upper()
    mission_id = normalize_mission_name(input("Mission ID: ").strip())
    start_utc = parse_console_datetime("Start UTC (YYYY-MM-DD HH:MM:SS): ")
    end_utc = parse_console_datetime("End UTC (YYYY-MM-DD HH:MM:SS): ")
    return station_id, mission_id, start_utc, end_utc


def get_station_config(station_id: str) -> dict:
    station_key = str(station_id or "").strip().upper()
    if station_key not in STATION_CONFIG:
        raise ScriptError("Unsupported station ID: {0}".format(station_id))
    return STATION_CONFIG[station_key]


def classify_band_match(candidate_bands: List[str], allowed_bands: set) -> str:
    normalized = set([str(item).upper() for item in candidate_bands if item])
    if not normalized or normalized == {"UNKNOWN"}:
        return "UNKNOWN"
    if normalized.intersection(allowed_bands):
        return "MATCH"
    return "MISMATCH"


def compute_band_penalty(band_match: str) -> float:
    if band_match == "MATCH":
        return 0.0
    if band_match == "UNKNOWN":
        return 1.0
    return 5.0


def preselect_candidates(
    target_track_df: pd.DataFrame,
    satellites,
    station,
) -> List[CandidatePreselection]:
    ts = load.timescale()

    timestamps = list(target_track_df["UTC"])
    times = ts.from_datetimes([value.to_pydatetime() for value in timestamps])

    target_az = np.array(target_track_df["AZ_target_deg"].values, dtype=float)
    target_el = np.array(target_track_df["EL_target_deg"].values, dtype=float)

    results = []

    for satellite, groups, raw_row in satellites:
        difference = satellite - station
        topocentric = difference.at(times)
        alt, az, distance = topocentric.altaz()

        sat_el = np.asarray(alt.degrees, dtype=float)
        sat_az = np.asarray(az.degrees, dtype=float)
        sat_range_km = np.asarray(distance.km, dtype=float)

        visible_mask = sat_el > 0.0
        if not np.any(visible_mask):
            continue

        separation = angular_separation_deg(target_az, target_el, sat_az, sat_el)
        valid_mask = visible_mask & np.isfinite(separation)
        if not np.any(valid_mask):
            continue

        min_idx = int(np.argmin(np.where(valid_mask, separation, np.inf)))
        min_sep = float(separation[min_idx])
        min_range_km = float(sat_range_km[min_idx])

        if min_sep > PRESELECTION_SEP_DEG:
            continue

        # Filter out obviously broken propagation results
        if not np.isfinite(min_range_km) or min_range_km > MAX_REASONABLE_RANGE_KM:
            continue

        close_mask = valid_mask & (separation <= FINAL_SEP_DEG)
        close_samples = int(np.count_nonzero(close_mask))
        visible_samples = int(np.count_nonzero(visible_mask))
        closest_time_utc = timestamps[min_idx].to_pydatetime()

        results.append(
            CandidatePreselection(
                object_name=str(raw_row.get("OBJECT_NAME", "")).strip() or "UNKNOWN",
                norad_cat_id=str(raw_row.get("NORAD_CAT_ID", "")).strip(),
                groups=",".join(groups),
                min_sep_deg=min_sep,
                closest_time_utc=closest_time_utc,
                visible_samples=visible_samples,
                close_samples=close_samples,
                sat_az_deg=float(sat_az[min_idx]),
                sat_el_deg=float(sat_el[min_idx]),
                sat_range_km=min_range_km,
                target_az_deg=float(target_az[min_idx]),
                target_el_deg=float(target_el[min_idx]),
            )
        )

    results.sort(key=lambda item: (item.min_sep_deg, -item.close_samples, item.closest_time_utc))
    return results


def enrich_candidates_with_bands(
    preselected_results: List[CandidatePreselection],
    allowed_bands: set,
) -> List[CandidateResult]:
    enriched = []

    for item in preselected_results[:TOP_N_BAND_LOOKUP]:
        satnogs_info = lookup_bands_by_norad(item.norad_cat_id)
        satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
        satnogs_freqs_mhz = satnogs_info.get("frequencies_mhz", [])
        band_match = classify_band_match(satnogs_bands, allowed_bands)

        score = (
            compute_band_penalty(band_match)
            + item.min_sep_deg
            - 0.05 * item.close_samples
        )

        enriched.append(
            CandidateResult(
                object_name=item.object_name,
                norad_cat_id=item.norad_cat_id,
                groups=item.groups,
                min_sep_deg=item.min_sep_deg,
                closest_time_utc=item.closest_time_utc,
                visible_samples=item.visible_samples,
                close_samples=item.close_samples,
                sat_az_deg=item.sat_az_deg,
                sat_el_deg=item.sat_el_deg,
                sat_range_km=item.sat_range_km,
                target_az_deg=item.target_az_deg,
                target_el_deg=item.target_el_deg,
                satnogs_bands=satnogs_bands,
                satnogs_freqs_mhz=satnogs_freqs_mhz,
                band_match=band_match,
                score=score,
                satnogs_url=satnogs_info.get("satnogs_url"),
                lookup_status=satnogs_info.get("lookup_status", "unknown"),
                satnogs_satellite_id=satnogs_info.get("satnogs_satellite_id"),
            )
        )

    enriched.sort(key=lambda item: (item.score, item.min_sep_deg, -item.close_samples, item.closest_time_utc))
    return enriched


def print_console_results(results: List[CandidateResult]):
    print("\n----------------------------------")
    print("Top candidates")
    print("----------------------------------")

    if not results:
        print("No candidates found after geometric preselection.")
        return

    for rank, item in enumerate(results[:TOP_N_OUTPUT], start=1):
        freqs_text = ",".join("{0:.3f}".format(x) for x in item.satnogs_freqs_mhz[:6])
        if not freqs_text:
            freqs_text = "none"

        print(
            "{0:02d}. {1} (NORAD {2}) | bands={3} | freqs_mhz={4} | match={5} | "
            "min_sep={6:.3f} deg | closest={7} | lookup={8} | satnogs_id={9}".format(
                rank,
                item.object_name,
                item.norad_cat_id,
                ",".join(item.satnogs_bands),
                freqs_text,
                item.band_match,
                item.min_sep_deg,
                item.closest_time_utc.isoformat(),
                item.lookup_status,
                item.satnogs_satellite_id,
            )
        )


def write_html_report(
    output_path: Path,
    station_id: str,
    mission_id: str,
    start_utc: str,
    end_utc: str,
    results: List[CandidateResult],
):
    rows = []
    for rank, item in enumerate(results[:TOP_N_OUTPUT], start=1):
        rows.append(
            """
            <tr>
                <td>{rank}</td>
                <td>{object_name}</td>
                <td>{norad}</td>
                <td>{bands}</td>
                <td>{freqs}</td>
                <td>{band_match}</td>
                <td>{lookup_status}</td>
                <td>{min_sep:.3f}</td>
                <td>{closest_time}</td>
                <td>{sat_az:.3f}</td>
                <td>{sat_el:.3f}</td>
                <td>{target_az:.3f}</td>
                <td>{target_el:.3f}</td>
                <td>{range_km:.1f}</td>
                <td>{groups}</td>
            </tr>
            """.format(
                rank=rank,
                object_name=html.escape(item.object_name),
                norad=html.escape(item.norad_cat_id),
                bands=html.escape(", ".join(item.satnogs_bands)),
                freqs=html.escape(", ".join("{0:.3f}".format(x) for x in item.satnogs_freqs_mhz[:6])),
                band_match=html.escape(item.band_match),
                lookup_status=html.escape(item.lookup_status),
                min_sep=item.min_sep_deg,
                closest_time=html.escape(item.closest_time_utc.isoformat()),
                sat_az=item.sat_az_deg,
                sat_el=item.sat_el_deg,
                target_az=item.target_az_deg,
                target_el=item.target_el_deg,
                range_km=item.sat_range_km,
                groups=html.escape(item.groups),
            )
        )

    html_text = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Identification Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; text-align: left; }}
        th {{ background: #f2f2f2; }}
      </style>
    </head>
    <body>
      <h1>Candidate Identification Report</h1>
      <p><strong>Station ID:</strong> {station_id}</p>
      <p><strong>Mission ID:</strong> {mission_id}</p>
      <p><strong>Time slot:</strong> {start_utc} to {end_utc}</p>
      <p><strong>Candidates found:</strong> {candidate_count}</p>

      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Object</th>
            <th>NORAD</th>
            <th>Bands</th>
            <th>Freqs (MHz)</th>
            <th>Band Match</th>
            <th>Lookup Status</th>
            <th>Min Sep (deg)</th>
            <th>Closest Time UTC</th>
            <th>Sat AZ</th>
            <th>Sat EL</th>
            <th>Target AZ</th>
            <th>Target EL</th>
            <th>Range (km)</th>
            <th>Groups</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </body>
    </html>
    """.format(
        station_id=html.escape(station_id),
        mission_id=html.escape(mission_id),
        start_utc=html.escape(start_utc),
        end_utc=html.escape(end_utc),
        candidate_count=len(results),
        rows="\n".join(rows),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main():
    station_id, mission_id, start_utc, end_utc = prompt_inputs()

    station_cfg = get_station_config(station_id)
    allowed_bands = set(station_cfg["allowed_bands"])

    print("\nFetching target track from Horizons...")
    target_track_df = fetch_target_track(
        mission_id=mission_id,
        station_id=station_id,
        start_time_utc=start_utc,
        stop_time_utc=end_utc,
        step_size=DEFAULT_STEP_SIZE,
    )

    catalog_meta = get_local_catalog_metadata()
    if not catalog_meta["exists"]:
        raise ScriptError(
            "Local ACTIVE catalog not found. Run update_candidate_catalog.py first."
        )

    print("Loading local full ACTIVE candidate catalog...")
    print(
        "Catalog snapshot: count={0}, fetched_at_utc={1}, age_hours={2}".format(
            catalog_meta["count"],
            catalog_meta["fetched_at_utc"],
            None if catalog_meta["age_hours"] is None else round(catalog_meta["age_hours"], 2),
        )
    )

    ts = load.timescale()
    satellites = load_local_candidate_catalog(ts)

    station = build_station(
        station_cfg["lat_deg"],
        station_cfg["lon_deg"],
        station_cfg["elevation_m"],
    )

    print("Running geometric preselection...")
    preselected = preselect_candidates(
        target_track_df=target_track_df,
        satellites=satellites,
        station=station,
    )
    print("Geometrically suspicious candidates: {0}".format(len(preselected)))

    print("Running SatNOGS band enrichment on top {0} candidates...".format(TOP_N_BAND_LOOKUP))
    results = enrich_candidates_with_bands(
        preselected_results=preselected,
        allowed_bands=allowed_bands,
    )

    print_console_results(results)

    output_dir = Path("results_horizons_identification")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_mission = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in mission_id)
    html_path = output_dir / "{0}_{1}_{2}.html".format(
        station_id,
        safe_mission,
        start_utc.replace(" ", "_").replace(":", ""),
    )

    write_html_report(
        output_path=html_path,
        station_id=station_id,
        mission_id=mission_id,
        start_utc=start_utc,
        end_utc=end_utc,
        results=results,
    )

    print("\nHTML report written to: {0}".format(html_path.resolve()))


if __name__ == "__main__":
    main()