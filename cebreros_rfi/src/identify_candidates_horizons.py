#!/usr/bin/env python3
"""
Interactive Identification tool:
- target track from Horizons
- external candidates from local full ACTIVE catalog
- candidate bands from SatNOGS
- outputs to console and HTML

This is part of the current operational engine.
It does not import functionality from validation workflows.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from cebreros_rfi.src.core.candidate_helpers import (
    candidate_sort_key,
    classify_band_match,
    compute_band_penalty,
    has_known_frequency,
    preselect_geometric_candidates,
)
from cebreros_rfi.src.core.operational_context import (
    DEFAULT_STEP_SIZE,
    FINAL_SEP_DEG,
    MAX_REASONABLE_RANGE_KM,
    PRESELECTION_SEP_DEG,
    build_operational_context,
)
from cebreros_rfi.src.mission_names import normalize_mission_name
from cebreros_rfi.src.satnogs_band_lookup import lookup_bands_by_norad


UTC = timezone.utc
TOP_N_KNOWN_OUTPUT = 10
TOP_N_UNKNOWN_OUTPUT = 10
MAX_RF_LOOKUPS = 120


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
    all_satnogs_bands: List[str]
    all_satnogs_freqs_mhz: List[float]
    band_match: str
    score: float
    satnogs_url: str
    lookup_status: str
    has_known_frequency: bool
    manual_review_candidate: bool


class ScriptError(RuntimeError):
    pass


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


def preselect_candidates(target_track_df, satellites, station) -> List[CandidatePreselection]:
    raw_results = preselect_geometric_candidates(
        target_track_df=target_track_df,
        satellites=satellites,
        station=station,
        preselection_sep_deg=PRESELECTION_SEP_DEG,
        final_sep_deg=FINAL_SEP_DEG,
        max_reasonable_range_km=MAX_REASONABLE_RANGE_KM,
    )

    return [
        CandidatePreselection(
            object_name=item["object_name"],
            norad_cat_id=item["norad_cat_id"],
            groups=item["groups"],
            min_sep_deg=item["min_sep_deg"],
            closest_time_utc=datetime.fromisoformat(item["closest_time_utc"]),
            visible_samples=item["visible_samples"],
            close_samples=item["close_samples"],
            sat_az_deg=item["sat_az_deg"],
            sat_el_deg=item["sat_el_deg"],
            sat_range_km=item["sat_range_km"],
            target_az_deg=item["target_az_deg"],
            target_el_deg=item["target_el_deg"],
        )
        for item in raw_results
    ]


def enrich_single_candidate(
    item: CandidatePreselection,
    allowed_bands: set,
) -> CandidateResult:
    satnogs_info = lookup_bands_by_norad(item.norad_cat_id)

    satnogs_bands = satnogs_info.get("bands", ["UNKNOWN"])
    satnogs_freqs_mhz = satnogs_info.get("frequencies_mhz", [])
    all_satnogs_bands = satnogs_info.get("all_bands", ["UNKNOWN"])
    all_satnogs_freqs_mhz = satnogs_info.get("all_frequencies_mhz", [])
    lookup_status = satnogs_info.get("lookup_status", "unknown")

    band_match = classify_band_match(satnogs_bands, allowed_bands)
    known_frequency = has_known_frequency(lookup_status)

    score = (
        compute_band_penalty(band_match)
        + item.min_sep_deg
        - 0.05 * item.close_samples
    )

    return CandidateResult(
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
        all_satnogs_bands=all_satnogs_bands,
        all_satnogs_freqs_mhz=all_satnogs_freqs_mhz,
        band_match=band_match,
        score=score,
        satnogs_url=satnogs_info.get("satnogs_url"),
        lookup_status=lookup_status,
        has_known_frequency=known_frequency,
        manual_review_candidate=not known_frequency,
    )


def build_rankings(
    preselected_results: List[CandidatePreselection],
    allowed_bands: set,
) -> Tuple[List[CandidateResult], List[CandidateResult], List[CandidateResult], int]:
    all_results = []
    known_results = []
    unknown_results = []
    lookups_done = 0

    for item in preselected_results:
        if lookups_done >= MAX_RF_LOOKUPS:
            break

        candidate = enrich_single_candidate(item, allowed_bands)
        all_results.append(candidate)
        lookups_done += 1

        if candidate.has_known_frequency:
            known_results.append(candidate)
        else:
            unknown_results.append(candidate)

        if (
            len(known_results) >= TOP_N_KNOWN_OUTPUT
            and len(unknown_results) >= TOP_N_UNKNOWN_OUTPUT
        ):
            break

    ranking_key = lambda item: (item.score,) + candidate_sort_key(
        {
            "min_sep_deg": item.min_sep_deg,
            "close_samples": item.close_samples,
            "closest_time_utc": item.closest_time_utc.isoformat(),
        }
    )
    all_results.sort(key=ranking_key)
    known_results.sort(key=ranking_key)
    unknown_results.sort(key=ranking_key)

    return all_results, known_results, unknown_results, lookups_done


def print_result_block(title: str, results: List[CandidateResult], limit: int):
    print("\n----------------------------------")
    print(title)
    print("----------------------------------")

    if not results:
        print("None")
        return

    for rank, item in enumerate(results[:limit], start=1):
        all_freqs_text = ",".join("{0:.3f}".format(x) for x in item.all_satnogs_freqs_mhz[:6]) or "none"

        print(
            "{0:02d}. {1} (NORAD {2}) | all_freqs_mhz={3} | lookup={4} | min_sep={5:.3f} deg | closest={6}".format(
                rank,
                item.object_name,
                item.norad_cat_id,
                all_freqs_text,
                item.lookup_status,
                item.min_sep_deg,
                item.closest_time_utc.isoformat(),
            )
        )


def main():
    station_id, mission_id, start_utc, end_utc = prompt_inputs()

    try:
        context = build_operational_context(
            station_id=station_id,
            mission_id=mission_id,
            start_utc=start_utc,
            end_utc=end_utc,
            step_size=DEFAULT_STEP_SIZE,
        )
    except Exception as exc:
        raise ScriptError(str(exc))

    allowed_bands = set(context.station_config.allowed_bands)

    print("\nUsing Horizons target track and local ACTIVE candidate catalog...")
    print(
        "Catalog snapshot: count={0}, fetched_at_utc={1}, age_hours={2}".format(
            context.catalog_meta["count"],
            context.catalog_meta["fetched_at_utc"],
            None if context.catalog_meta["age_hours"] is None else round(context.catalog_meta["age_hours"], 2),
        )
    )

    print("Running geometric preselection...")
    preselected = preselect_candidates(
        target_track_df=context.target_track_df,
        satellites=context.satellites,
        station=context.station,
    )
    print("Geometrically suspicious candidates: {0}".format(len(preselected)))

    print("Building RF-aware rankings...")
    all_results, known_results, unknown_results, lookups_done = build_rankings(
        preselected_results=preselected,
        allowed_bands=allowed_bands,
    )

    print_result_block("Known-frequency candidates", known_results, TOP_N_KNOWN_OUTPUT)
    print_result_block("Unknown-frequency candidates", unknown_results, TOP_N_UNKNOWN_OUTPUT)
    print_result_block("All checked candidates", all_results, max(TOP_N_KNOWN_OUTPUT, TOP_N_UNKNOWN_OUTPUT))

    output_dir = Path("results_horizons_identification")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_mission = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_"
        for ch in context.mission_id
    )
    html_path = output_dir / "{0}_{1}_{2}.html".format(
        context.station_id,
        safe_mission,
        start_utc.replace(" ", "_").replace(":", ""),
    )

    html_path.write_text(
        "<html><body><h1>Identification report generated.</h1></body></html>",
        encoding="utf-8",
    )

    print("\nHTML report written to: {0}".format(html_path.resolve()))


if __name__ == "__main__":
    main()
