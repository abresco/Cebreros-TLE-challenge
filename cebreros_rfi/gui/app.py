#!/usr/bin/env python3
"""
Streamlit entrypoint for identification and prediction workflows.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cebreros_rfi.gui.identification_service import (
    run_identification,
    IdentificationServiceError,
)
from cebreros_rfi.gui.prediction_service import (
    load_prediction_jobs_from_schedule_csv,
    run_prediction_interval,
    run_prediction_jobs,
    PredictionServiceError,
)
from cebreros_rfi.src.feedback_db import (
    init_db,
    record_identification_feedback,
)
from cebreros_rfi.src.core.operational_context import get_supported_station_ids
from cebreros_rfi.src.mission_names import MISSION_CANONICAL_NAMES

init_db()

st.set_page_config(
    page_title="Cebreros RFI Tool",
    page_icon="📡",
    layout="wide",
)

st.title("Cebreros RFI Identification & Prediction Tool")
st.caption("ESA demo mock-up — Identification active, Prediction v1 active")

with st.expander("Help / Method"):
    st.markdown(
        """
        ### Overview

        This application supports two workflows:

        **Identification**
        - Input: station, ESA mission, past UTC interval
        - Output: ranked candidate interferers crossing the victim mission track

        **Prediction**
        - Input: station, ESA mission, future UTC interval or schedule CSV
        - Output: ranked future candidates, probability, and possible RFI slots

        ### Data sources

        - **JPL Horizons**: target ESA mission track
        - **CelesTrak ACTIVE**: external satellite candidate catalog
        - **SatNOGS**: contextual RF metadata when available
        - **SQLite**: local feedback database

        ### Schedule-based Prediction

        For schedule-based Prediction, the current implementation:
        - parses a planning CSV
        - uses **BOT-EOT** as the effective tracking interval
        - supports **CEB**, **MLG**, and **NNO**
        - treats **NNO3** as **NNO**
        - allows filtering and selecting jobs before batch execution
        """
    )


STATION_OPTIONS = list(get_supported_station_ids())
MISSION_OPTIONS = list(MISSION_CANONICAL_NAMES)
FEEDBACK_LABEL_OPTIONS = ["confirmed", "rejected", "uncertain"]
LIST_LIKE_COLUMNS = [
    "satnogs_freqs_mhz",
    "all_satnogs_freqs_mhz",
    "all_satnogs_bands",
    "cebreros_freqs_mhz",
    "cebreros_bands",
]


def format_list_like_value(value):
    if not isinstance(value, list):
        return value
    if not value:
        return ""

    formatted = []
    for item in value:
        if isinstance(item, float):
            formatted.append("{0:.3f}".format(item))
        else:
            formatted.append(str(item))
    return ", ".join(formatted)


def dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list-heavy fields into compact strings for the Streamlit tables.
    """
    df = df.copy()

    for column in LIST_LIKE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(format_list_like_value)

    return df


tab_identification, tab_prediction = st.tabs(["Identification", "Prediction"])


with tab_identification:
    st.subheader("Run candidate identification")

    with st.form("identification_form"):
        col1, col2 = st.columns(2)

        with col1:
            station_id = st.selectbox("Station ID", STATION_OPTIONS, index=0, key="id_station")
            mission_id = st.selectbox(
                "Mission ID",
                MISSION_OPTIONS,
                index=1,
                key="id_mission",
            )

        with col2:
            start_utc = st.text_input("Start UTC", value="2025-07-25 04:56:00", key="id_start")
            end_utc = st.text_input("End UTC", value="2025-07-25 04:58:00", key="id_end")

        submitted = st.form_submit_button("Run Identification", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Running identification..."):
                result = run_identification(
                    station_id=station_id,
                    mission_id=mission_id,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )

            st.session_state["last_identification_result"] = result
            st.success("Identification completed")

        except IdentificationServiceError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("last_identification_result")
    if result:
        meta = result["catalog_meta"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Geometric preselection", result["preselected_count"])
        c2.metric("SatNOGS lookups", result["lookups_done"])
        c3.metric("Known-frequency candidates", len(result["known_results"]))
        c4.metric("Unknown-frequency candidates", len(result["unknown_results"]))

        st.caption(
            "Catalog snapshot — count={0}, fetched_at_utc={1}, age_hours={2}".format(
                meta["count"],
                meta["fetched_at_utc"],
                None if meta["age_hours"] is None else round(meta["age_hours"], 2),
            )
        )

        st.markdown("### Known-frequency candidates")
        if result["known_results"]:
            known_df = dataframe_for_display(pd.DataFrame(result["known_results"]))
            st.dataframe(
                known_df[
                    [
                        "object_name",
                        "norad_cat_id",
                        "lookup_status",
                        "all_satnogs_freqs_mhz",
                        "min_sep_deg",
                        "closest_time_utc",
                        "sat_az_deg",
                        "sat_el_deg",
                        "target_az_deg",
                        "target_el_deg",
                        "sat_range_km",
                    ]
                ].rename(
                    columns={
                        "object_name": "Object",
                        "norad_cat_id": "NORAD",
                        "lookup_status": "Lookup Status",
                        "all_satnogs_freqs_mhz": "All Freqs (MHz)",
                        "min_sep_deg": "Min Sep (deg)",
                        "closest_time_utc": "Closest Time UTC",
                        "sat_az_deg": "Sat AZ",
                        "sat_el_deg": "Sat EL",
                        "target_az_deg": "Target AZ",
                        "target_el_deg": "Target EL",
                        "sat_range_km": "Range (km)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No known-frequency candidates found.")

        st.markdown("### Unknown-frequency candidates")
        if result["unknown_results"]:
            unknown_df = dataframe_for_display(pd.DataFrame(result["unknown_results"]))
            st.dataframe(
                unknown_df[
                    [
                        "object_name",
                        "norad_cat_id",
                        "lookup_status",
                        "min_sep_deg",
                        "closest_time_utc",
                        "sat_az_deg",
                        "sat_el_deg",
                        "target_az_deg",
                        "target_el_deg",
                        "sat_range_km",
                    ]
                ].rename(
                    columns={
                        "object_name": "Object",
                        "norad_cat_id": "NORAD",
                        "lookup_status": "Lookup Status",
                        "min_sep_deg": "Min Sep (deg)",
                        "closest_time_utc": "Closest Time UTC",
                        "sat_az_deg": "Sat AZ",
                        "sat_el_deg": "Sat EL",
                        "target_az_deg": "Target AZ",
                        "target_el_deg": "Target EL",
                        "sat_range_km": "Range (km)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No unknown-frequency candidates found.")

        st.markdown("### Save user feedback to DB")

        feedback_candidates = result["known_results"] + result["unknown_results"]
        if feedback_candidates:
            candidate_options = {
                "{0} | NORAD {1} | sep={2:.3f}".format(
                    item["object_name"],
                    item["norad_cat_id"],
                    item["min_sep_deg"],
                ): item
                for item in feedback_candidates
            }

            with st.form("feedback_form"):
                selected_key = st.selectbox("Candidate", list(candidate_options.keys()))
                selected_label = st.selectbox("User label", FEEDBACK_LABEL_OPTIONS)
                notes = st.text_input("Notes", value="")
                save_feedback = st.form_submit_button("Save Feedback", use_container_width=True)

            if save_feedback:
                candidate = candidate_options[selected_key]
                record_identification_feedback(
                    station_id=result["station_id"],
                    mission_id=result["mission_id"],
                    pass_start_utc=result["start_utc"],
                    pass_end_utc=result["end_utc"],
                    rfi_start_utc=result["start_utc"],
                    rfi_end_utc=result["end_utc"],
                    candidate=candidate,
                    user_label=selected_label,
                    probability=None,
                    notes=notes,
                )
                st.success("Feedback saved to DB.")
        else:
            st.info("No candidates available for feedback.")


with tab_prediction:
    st.subheader("Prediction")

    mode = st.radio(
        "Prediction input mode",
        ["Manual future interval", "Schedule CSV"],
        horizontal=True,
    )

    if mode == "Manual future interval":
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                pred_station_id = st.selectbox("Station ID", STATION_OPTIONS, index=0, key="pred_station")
                pred_mission_id = st.selectbox(
                    "Mission ID",
                    MISSION_OPTIONS,
                    index=1,
                    key="pred_mission",
                )

            with col2:
                pred_start_utc = st.text_input("Future Start UTC", value="2026-01-05 10:00:00", key="pred_start")
                pred_end_utc = st.text_input("Future End UTC", value="2026-01-05 10:20:00", key="pred_end")

            run_prediction = st.form_submit_button("Run Prediction", use_container_width=True)

        if run_prediction:
            try:
                with st.spinner("Running prediction..."):
                    pred_result = run_prediction_interval(
                        station_id=pred_station_id,
                        mission_id=pred_mission_id,
                        start_utc=pred_start_utc,
                        end_utc=pred_end_utc,
                    )
                st.session_state["last_prediction_result"] = pred_result
                st.success("Prediction completed")
            except PredictionServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

        pred_result = st.session_state.get("last_prediction_result")
        if pred_result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Geometric preselection", pred_result["preselected_count"])
            c2.metric("SatNOGS lookups", pred_result["lookups_done"])
            c3.metric("Predicted candidates", len(pred_result["results"]))

            pred_df = dataframe_for_display(pd.DataFrame(pred_result["results"]))
            if not pred_df.empty:
                st.dataframe(
                    pred_df[
                        [
                            "object_name",
                            "norad_cat_id",
                            "probability",
                            "probability_label",
                            "lookup_status",
                            "all_satnogs_freqs_mhz",
                            "min_sep_deg",
                            "close_samples",
                            "closest_time_utc",
                            "possible_rfi_start_utc",
                            "possible_rfi_end_utc",
                            "sat_az_deg",
                            "sat_el_deg",
                            "target_az_deg",
                            "target_el_deg",
                            "sat_range_km",
                            "mission_history_confirmed",
                            "global_history_confirmed",
                        ]
                    ].rename(
                        columns={
                            "object_name": "Object",
                            "norad_cat_id": "NORAD",
                            "probability": "Probability",
                            "probability_label": "Probability Level",
                            "lookup_status": "Lookup Status",
                            "all_satnogs_freqs_mhz": "All Freqs (MHz)",
                            "min_sep_deg": "Min Sep (deg)",
                            "close_samples": "Close Samples",
                            "closest_time_utc": "Closest Time UTC",
                            "possible_rfi_start_utc": "Possible RFI Start UTC",
                            "possible_rfi_end_utc": "Possible RFI End UTC",
                            "sat_az_deg": "Sat AZ",
                            "sat_el_deg": "Sat EL",
                            "target_az_deg": "Target AZ",
                            "target_el_deg": "Target EL",
                            "sat_range_km": "Range (km)",
                            "mission_history_confirmed": "Mission Hist Confirmed",
                            "global_history_confirmed": "Global Hist Confirmed",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No predicted candidates found for this interval.")

    else:
        st.caption("Upload a schedule CSV, choose the station filter, preview usable jobs, and run batch Prediction on the selected passes.")

        schedule_station = st.selectbox(
            "Schedule station filter",
            STATION_OPTIONS,
            index=0,
            key="schedule_station_filter",
        )

        uploaded_csv = st.file_uploader("Upload schedule CSV", type=["csv"])

        if uploaded_csv is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_csv.read())
                    tmp_path = tmp.name

                schedule_info = load_prediction_jobs_from_schedule_csv(
                    csv_path=tmp_path,
                    station_filter=schedule_station,
                )

                st.success("Schedule parsed")
                st.write("Summary:", schedule_info["summary"])

                jobs_df = pd.DataFrame(schedule_info["jobs"])
                if jobs_df.empty:
                    st.warning("No usable jobs found for the selected station.")
                else:
                    mission_options = sorted(jobs_df["mission_id"].dropna().unique().tolist())

                    col1, col2 = st.columns(2)
                    with col1:
                        selected_missions = st.multiselect(
                            "Filter by mission",
                            mission_options,
                            default=mission_options,
                        )
                    with col2:
                        max_rows = st.number_input(
                            "Maximum rows to show",
                            min_value=1,
                            max_value=500,
                            value=min(50, len(jobs_df)),
                            step=1,
                        )

                    filtered_df = jobs_df[jobs_df["mission_id"].isin(selected_missions)].copy()
                    filtered_df = filtered_df.sort_values(by=["start_utc", "mission_id"]).reset_index(drop=True)

                    preview_df = filtered_df.head(int(max_rows)).copy()

                    st.markdown("### Schedule jobs preview")
                    st.dataframe(
                        preview_df[
                            [
                                "job_id",
                                "mission_id",
                                "station_id",
                                "start_utc",
                                "end_utc",
                                "reference",
                                "comment",
                            ]
                        ].rename(
                            columns={
                                "job_id": "Job ID",
                                "mission_id": "Mission",
                                "station_id": "Station",
                                "start_utc": "BOT UTC",
                                "end_utc": "EOT UTC",
                                "reference": "Reference",
                                "comment": "Comment",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    selectable_jobs = preview_df.to_dict(orient="records")
                    option_map = {
                        "{0} | {1} | {2} -> {3} | {4}".format(
                            item["job_id"],
                            item["mission_id"],
                            item["start_utc"],
                            item["end_utc"],
                            item["reference"] or "no_reference",
                        ): item
                        for item in selectable_jobs
                    }

                    selected_labels = st.multiselect(
                        "Select schedule jobs to run",
                        list(option_map.keys()),
                        default=list(option_map.keys())[: min(5, len(option_map))],
                    )

                    if st.button("Run Batch Prediction on Selected Jobs", use_container_width=True):
                        selected_jobs = [option_map[label] for label in selected_labels]

                        if not selected_jobs:
                            st.warning("Select at least one schedule job.")
                        else:
                            with st.spinner("Running batch prediction..."):
                                batch_result = run_prediction_jobs(
                                    station_id=schedule_station,
                                    jobs=selected_jobs,
                                )

                            st.success("Batch Prediction completed")
                            st.write("Jobs processed:", batch_result["job_count"])

                            for idx, job_bundle in enumerate(batch_result["jobs"], start=1):
                                job = job_bundle["job_metadata"]
                                pred = job_bundle["prediction"]

                                st.markdown(
                                    "#### Job {0}: {1} | {2} -> {3}".format(
                                        idx,
                                        job["mission_id"],
                                        job["start_utc"],
                                        job["end_utc"],
                                    )
                                )

                                job_df = dataframe_for_display(pd.DataFrame(pred["results"]))
                                if not job_df.empty:
                                    st.dataframe(
                                        job_df[
                                            [
                                                "object_name",
                                                "norad_cat_id",
                                                "probability",
                                                "probability_label",
                                                "lookup_status",
                                                "all_satnogs_freqs_mhz",
                                                "min_sep_deg",
                                                "close_samples",
                                                "possible_rfi_start_utc",
                                                "possible_rfi_end_utc",
                                            ]
                                        ].rename(
                                            columns={
                                                "object_name": "Object",
                                                "norad_cat_id": "NORAD",
                                                "probability": "Probability",
                                                "probability_label": "Probability Level",
                                                "lookup_status": "Lookup Status",
                                                "all_satnogs_freqs_mhz": "All Freqs (MHz)",
                                                "min_sep_deg": "Min Sep (deg)",
                                                "close_samples": "Close Samples",
                                                "possible_rfi_start_utc": "Possible RFI Start UTC",
                                                "possible_rfi_end_utc": "Possible RFI End UTC",
                                            }
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                else:
                                    st.info("No candidates found for this job.")

            except PredictionServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)
