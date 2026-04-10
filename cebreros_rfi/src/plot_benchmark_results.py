#!/usr/bin/env python3
"""
Visual report for Case 2 benchmark results.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Columns shown in the incident summary table.
INCIDENT_TABLE_COLUMNS = [
    "incident_id",
    "mission",
    "mission_folder",
    "label",
    "status",
    "quality_class",
    "pass_name",
    "num_candidates",
    "top1_object_name",
    "top1_min_sep_deg",
    "top1_time_offset_sec",
    "top1_risk_level",
    "top1_groups",
]

# Columns shown in the top-candidates HTML table.
TOP_CANDIDATE_COLUMNS = [
    "incident_id",
    "mission",
    "mission_folder",
    "label",
    "rank",
    "object_name",
    "norad_cat_id",
    "groups",
    "min_sep_deg",
    "time_offset_sec",
    "risk_level",
]


def main():
    # Read input paths and create the output directory.
    args = build_argument_parser().parse_args()

    incidents_csv = Path(args.incidents_csv).expanduser().resolve()
    candidates_csv = Path(args.candidates_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the benchmark CSV files and derive summary metrics.
    incidents_df, candidates_df = load_data(incidents_csv, candidates_csv)
    metrics = compute_summary_metrics(incidents_df)

    # Generate plots used by the HTML report.
    status_plot_path = make_status_counts_plot(incidents_df, output_dir)
    ranking_plot_path = make_top1_ranking_plot(incidents_df, output_dir)
    scatter_plot_path = make_top1_scatter_plot(incidents_df, output_dir)

    # Write the final HTML report referencing the generated plots.
    html_report_path = write_html_report(
        output_dir=output_dir,
        metrics=metrics,
        incidents_df=incidents_df,
        candidates_df=candidates_df,
        status_plot_path=status_plot_path,
        ranking_plot_path=ranking_plot_path,
        scatter_plot_path=scatter_plot_path,
    )

    print("Saved HTML report: {0}".format(html_report_path))
    print("Saved status plot: {0}".format(status_plot_path))

    if ranking_plot_path is not None:
        print("Saved ranking plot: {0}".format(ranking_plot_path))
    if scatter_plot_path is not None:
        print("Saved scatter plot: {0}".format(scatter_plot_path))


def build_argument_parser():
    # Define the command line interface for the benchmark report.
    parser = argparse.ArgumentParser(
        description="Generate a visual HTML report from benchmark CSV outputs."
    )
    parser.add_argument(
        "--incidents-csv",
        required=True,
        help="Path to benchmark_incidents.csv",
    )
    parser.add_argument(
        "--candidates-csv",
        required=True,
        help="Path to benchmark_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the report and plots will be written",
    )
    return parser


def load_data(incidents_csv, candidates_csv):
    # Load the benchmark CSV files and parse datetime columns.
    incidents_df = pd.read_csv(incidents_csv)
    candidates_df = pd.read_csv(candidates_csv)

    parse_datetime_column(incidents_df, "incident_time_utc")
    parse_datetime_column(incidents_df, "top1_closest_time_utc")
    parse_datetime_column(candidates_df, "incident_time_utc")
    parse_datetime_column(candidates_df, "closest_time_utc")

    incidents_df["quality_class"] = incidents_df.apply(classify_incident_quality, axis=1)
    return incidents_df, candidates_df


def parse_datetime_column(df, column_name):
    # Parse one datetime column when it exists in the DataFrame.
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce", utc=True)


def classify_incident_quality(row):
    # Assign a simple quality class to one incident row.
    status = str(row.get("status", "") or "")

    if status == "no_pass_found":
        return "no_pass_found"
    if status in ("normalization_error", "identification_error", "unexpected_error"):
        return "error"

    min_sep = row.get("top1_min_sep_deg")
    offset = row.get("top1_time_offset_sec")

    if pd.isna(min_sep) or pd.isna(offset):
        return "weak_candidate"
    if min_sep <= 1.0 and offset <= 30:
        return "strong_candidate"
    if min_sep <= 3.0 and offset <= 60:
        return "possible_candidate"
    return "weak_candidate"


def make_status_counts_plot(incidents_df, output_dir):
    # Plot how many incidents ended in each processing status.
    counts = incidents_df["status"].fillna("unknown").value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Incident Processing Status Counts")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "incident_status_counts.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def make_top1_ranking_plot(incidents_df, output_dir):
    # Plot the minimum separation of the top candidate per processed incident.
    df = incidents_df.copy()
    df = df[df["status"] == "processed"].copy()
    df = df[~df["top1_min_sep_deg"].isna()].copy()

    if df.empty:
        return None

    df = df.sort_values("top1_min_sep_deg", ascending=True)
    labels = ["{0} ({1})".format(row["incident_id"], row["mission"]) for _, row in df.iterrows()]
    values = df["top1_min_sep_deg"].tolist()

    plt.figure(figsize=(10, max(5, len(df) * 0.5)))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title("Top-1 Candidate Minimum Separation by Incident")
    plt.xlabel("Minimum Separation (deg)")
    plt.ylabel("Incident")
    plt.tight_layout()

    output_path = output_dir / "incident_top1_ranking.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def make_top1_scatter_plot(incidents_df, output_dir):
    # Plot time offset versus minimum separation for the top candidate.
    df = incidents_df.copy()
    df = df[df["status"] == "processed"].copy()
    df = df[~df["top1_min_sep_deg"].isna()].copy()
    df = df[~df["top1_time_offset_sec"].isna()].copy()

    if df.empty:
        return None

    plt.figure(figsize=(9, 6))

    for label in ["possible_rfi", "discarded", "uncertain"]:
        subset = df[df["label"] == label]
        if subset.empty:
            continue

        plt.scatter(
            subset["top1_time_offset_sec"],
            subset["top1_min_sep_deg"],
            label=label,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["incident_id"],
                (row["top1_time_offset_sec"], row["top1_min_sep_deg"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.axhline(1.0, linestyle="--")
    plt.axhline(3.0, linestyle="--")
    plt.axvline(30, linestyle="--")
    plt.axvline(60, linestyle="--")

    plt.title("Top-1 Candidate Quality: Time Offset vs Minimum Separation")
    plt.xlabel("Top-1 Time Offset (seconds)")
    plt.ylabel("Top-1 Minimum Separation (deg)")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "incident_top1_scatter.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def compute_summary_metrics(incidents_df):
    # Compute the headline metrics shown at the top of the report.
    total_incidents = len(incidents_df)
    processed = int((incidents_df["status"] == "processed").sum())
    no_pass_found = int((incidents_df["status"] == "no_pass_found").sum())

    strong = int((incidents_df["quality_class"] == "strong_candidate").sum())
    possible = int((incidents_df["quality_class"] == "possible_candidate").sum())
    weak = int((incidents_df["quality_class"] == "weak_candidate").sum())
    errors = int((incidents_df["quality_class"] == "error").sum())

    processed_with_candidate = incidents_df[
        (incidents_df["status"] == "processed")
        & (~incidents_df["top1_min_sep_deg"].isna())
    ]

    best_case = None
    if not processed_with_candidate.empty:
        best_row = processed_with_candidate.sort_values("top1_min_sep_deg").iloc[0]
        best_case = {
            "incident_id": best_row["incident_id"],
            "mission": best_row["mission"],
            "top1_object_name": best_row.get("top1_object_name"),
            "top1_min_sep_deg": best_row.get("top1_min_sep_deg"),
            "top1_time_offset_sec": best_row.get("top1_time_offset_sec"),
        }

    return {
        "total_incidents": total_incidents,
        "processed": processed,
        "no_pass_found": no_pass_found,
        "strong": strong,
        "possible": possible,
        "weak": weak,
        "errors": errors,
        "best_case": best_case,
    }


def build_incident_table_html(incidents_df):
    # Build the main incident summary table shown in the report.
    table_df = incidents_df.copy()
    for column in INCIDENT_TABLE_COLUMNS:
        if column not in table_df.columns:
            table_df[column] = None

    html_rows = []
    html_rows.append("<table>")
    html_rows.append("<thead><tr>{0}</tr></thead>".format(
        "".join("<th>{0}</th>".format(column) for column in INCIDENT_TABLE_COLUMNS)
    ))
    html_rows.append("<tbody>")

    for _, row in table_df.iterrows():
        html_rows.append('<tr style="{0}">'.format(get_incident_row_style(row)))
        for column in INCIDENT_TABLE_COLUMNS:
            html_rows.append("<td>{0}</td>".format(format_html_cell(row[column])))
        html_rows.append("</tr>")

    html_rows.append("</tbody>")
    html_rows.append("</table>")
    return "\n".join(html_rows)


def get_incident_row_style(row):
    # Color the incident table rows according to the quality class.
    quality = row["quality_class"]

    if quality == "strong_candidate":
        return "background-color: #d9f2d9;"
    if quality == "possible_candidate":
        return "background-color: #fff3cd;"
    if quality == "weak_candidate":
        return "background-color: #f8d7da;"
    if quality == "no_pass_found":
        return "background-color: #e2e3e5;"
    if quality == "error":
        return "background-color: #f5c6cb;"
    return ""


def build_top_candidates_html(candidates_df, max_rows=20):
    # Build the HTML table with the strongest candidate rows.
    if candidates_df.empty:
        return "<p>No candidate rows available.</p>"

    df = candidates_df.copy()
    df = df.sort_values(["min_sep_deg", "time_offset_sec"], ascending=[True, True]).head(max_rows)

    html_rows = []
    html_rows.append("<table>")
    html_rows.append("<thead><tr>{0}</tr></thead>".format(
        "".join("<th>{0}</th>".format(column) for column in TOP_CANDIDATE_COLUMNS)
    ))
    html_rows.append("<tbody>")

    for _, row in df.iterrows():
        html_rows.append("<tr>")
        for column in TOP_CANDIDATE_COLUMNS:
            html_rows.append("<td>{0}</td>".format(format_html_cell(row[column])))
        html_rows.append("</tr>")

    html_rows.append("</tbody>")
    html_rows.append("</table>")
    return "\n".join(html_rows)


def format_html_cell(value):
    # Format HTML cell values while keeping missing values readable.
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return "{0:.3f}".format(value)
    return value


def write_html_report(
    output_dir,
    metrics,
    incidents_df,
    candidates_df,
    status_plot_path,
    ranking_plot_path,
    scatter_plot_path,
):
    # Assemble the final HTML report from metrics, tables and plots.
    best_case_html = build_best_case_html(metrics["best_case"])
    incident_table_html = build_incident_table_html(incidents_df)
    top_candidates_html = build_top_candidates_html(candidates_df)

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Case 2 Benchmark Visual Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 24px;
    line-height: 1.4;
}}
h1, h2 {{
    margin-top: 28px;
}}
.card-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}}
.card {{
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 12px;
}}
.card-title {{
    font-size: 13px;
    color: #555;
}}
.card-value {{
    font-size: 26px;
    font-weight: bold;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 6px 8px;
    text-align: left;
    font-size: 13px;
}}
th {{
    background-color: #f2f2f2;
}}
img {{
    max-width: 100%;
    border: 1px solid #ccc;
    margin-top: 8px;
}}
.small {{
    color: #555;
    font-size: 13px;
}}
</style>
</head>
<body>
<h1>Case 2 Benchmark Visual Report</h1>

<div class="card-grid">
    <div class="card"><div class="card-title">Total incidents</div><div class="card-value">{total_incidents}</div></div>
    <div class="card"><div class="card-title">Processed</div><div class="card-value">{processed}</div></div>
    <div class="card"><div class="card-title">No pass found</div><div class="card-value">{no_pass_found}</div></div>
    <div class="card"><div class="card-title">Errors</div><div class="card-value">{errors}</div></div>
    <div class="card"><div class="card-title">Strong candidates</div><div class="card-value">{strong}</div></div>
    <div class="card"><div class="card-title">Possible candidates</div><div class="card-value">{possible}</div></div>
    <div class="card"><div class="card-title">Weak candidates</div><div class="card-value">{weak}</div></div>
</div>

<h2>Best case found</h2>
{best_case_html}

<h2>Legend</h2>
<ul>
<li><strong>strong_candidate</strong>: min_sep ≤ 1 deg and offset ≤ 30 s</li>
<li><strong>possible_candidate</strong>: min_sep ≤ 3 deg and offset ≤ 60 s</li>
<li><strong>weak_candidate</strong>: processed but not meeting the above thresholds</li>
<li><strong>no_pass_found</strong>: no station pass matched the incident timestamp</li>
<li><strong>error</strong>: normalization or execution problem</li>
</ul>

<h2>Processing status</h2>
{status_plot}

<h2>Top-1 ranking by minimum separation</h2>
{ranking_plot}

<h2>Top-1 quality scatter</h2>
<p class="small">X axis: time offset in seconds. Y axis: minimum separation in degrees.</p>
{scatter_plot}

<h2>Incident summary table</h2>
{incident_table_html}

<h2>Top candidate rows</h2>
{top_candidates_html}

</body>
</html>
""".format(
        total_incidents=metrics["total_incidents"],
        processed=metrics["processed"],
        no_pass_found=metrics["no_pass_found"],
        errors=metrics["errors"],
        strong=metrics["strong"],
        possible=metrics["possible"],
        weak=metrics["weak"],
        best_case_html=best_case_html,
        status_plot=build_plot_img_tag(status_plot_path, "status plot"),
        ranking_plot=build_plot_img_tag(ranking_plot_path, "ranking plot"),
        scatter_plot=build_plot_img_tag(scatter_plot_path, "scatter plot"),
        incident_table_html=incident_table_html,
        top_candidates_html=top_candidates_html,
    )

    output_path = output_dir / "benchmark_summary.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_best_case_html(best_case):
    # Build the HTML block for the best processed incident.
    if best_case is None:
        return "<p>No processed incidents with candidates.</p>"

    return """
    <ul>
        <li><strong>Incident:</strong> {incident_id}</li>
        <li><strong>Mission:</strong> {mission}</li>
        <li><strong>Top candidate:</strong> {top1_object_name}</li>
        <li><strong>Minimum separation:</strong> {top1_min_sep_deg:.3f} deg</li>
        <li><strong>Time offset:</strong> {top1_time_offset_sec:.1f} s</li>
    </ul>
    """.format(**best_case)


def build_plot_img_tag(path_obj, alt_text):
    # Build the HTML image tag only when the plot exists.
    if path_obj is None:
        return ""
    return '<img src="{0}" alt="{1}">'.format(path_obj.name, alt_text)


if __name__ == "__main__":
    main()
