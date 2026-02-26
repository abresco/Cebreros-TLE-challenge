# Cebreros-TLE-challenge

This repository aims to process satellite data for **RFI Identification and Prediction** at the Cebreros tracking station.

## Project Structure

- **data/**: Data files (CSV, TLEs).
- **core/**: Logic for orbital propagation, azimuth, and elevation calculations.
- **validation/**: Comparison with real data and metrics.
- **gui/**: Graphical User Interface (if implemented).

## Requirements

- Python 3.x
- Dependencies (see `requirements.txt`)

## How to Use

1. Clone the repository:
   ```bash
   git clone <https://github.com/abresco/Cebreros-TLE-challenge.git>
2. Navigate to the project folder:

cd Cebreros-TLE-challenge

3. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

4. Install dependencies:

pip install -r requirements.txt