# Cebreros-TLE-challenge

Python-based tool for **RFI Identification and Prediction** at the ESA Cebreros tracking station.

## Current Status

The project currently supports:

- **Identification** of possible interferers during past mission tracking intervals
- **Prediction v1** of possible future interferers for upcoming mission tracking intervals
- **Local feedback storage** in SQLite to support future probability refinement

## Main Data Sources

- **JPL Horizons** for the ESA victim mission track
- **CelesTrak ACTIVE** for external satellite candidates
- **SatNOGS** for contextual RF metadata
- **SQLite** for local feedback history

## Repository Structure

- `cebreros_rfi/gui/` — Streamlit GUI and service layers
- `cebreros_rfi/src/` — operational engine
- `cebreros_rfi/validation/` — validation, benchmarking, and legacy scripts

## Installation

Clone the repository:

```bash
git clone https://github.com/abresco/Cebreros-TLE-challenge.git
cd Cebreros-TLE-challenge
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the GUI

From the project root:

```bash
python -m streamlit run cebreros_rfi/gui/app.py
```

The GUI currently includes:

- **Identification**
- **Prediction**
- feedback saving to a local SQLite database

## Run Identification from CLI

```bash
python -m cebreros_rfi.src.identify_candidates_horizons
```

Inputs:

- Station ID
- Mission ID
- Start UTC
- End UTC

Outputs:

- ranked candidates in console
- HTML report

## Run Prediction from CLI

Example:

```bash
python -m cebreros_rfi.src.predict_candidates_horizons --station-id CEB --mission-id JUICE --start-utc "2026-01-05 10:00:00" --end-utc "2026-01-05 10:20:00"
```

Outputs:

- ranked candidates in console
- CSV report

## Local Catalog

The tool uses a locally cached **CelesTrak ACTIVE** catalog.

When the operational tool runs, the catalog is automatically checked and refreshed if it is older than the configured threshold.

## Feedback Database

User feedback is stored locally in:

`cebreros_rfi/data/db/rfi_feedback.sqlite3`

Stored feedback includes:

- victim ESA mission
- interferer NORAD ID
- user label (`confirmed`, `rejected`, `uncertain`)

Prediction reuses this history, prioritizing:

1. mission-specific victim/interferer history
2. global interferer history

## Probability Model

The current probability is a heuristic score from `0` to `100`.

It is based on:

- minimum angular separation
- persistence across close samples
- historical feedback recurrence

RF metadata is shown as contextual information, but it does not directly increase probability.

## Notes

- The current implementation is focused on **CEB**
- XML schedule support is prepared and may be adapted once the final ESA Scheduling XML structure is confirmed