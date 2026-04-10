import re

import requests


# Horizons API endpoint.
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Example target used in this prototype.
TARGET_ID = "61449"

# Example query parameters for a short observer ephemeris request.
HORIZONS_PARAMS = {
    "format": "text",
    "COMMAND": TARGET_ID,
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "OBSERVER",
    "CENTER": "c@399",
    "START_TIME": "2025-Mar-19",
    "STOP_TIME": "2025-Mar-19",
    "STEP_SIZE": "1m",
    "QUANTITIES": "23",
}


def main():
    # Request a small Horizons preview for the example target.
    response_text = fetch_horizons_text(HORIZONS_PARAMS)

    print("Response preview:\n")
    print(response_text[:2000])

    # Extract the azimuth and elevation points from the SOE block.
    az_el_data = extract_az_el_data(response_text)

    if not az_el_data:
        print("Some error occurred.")
        return

    # Print the last few valid points as a quick visual check.
    print("\nAzimuth and Elevation for every minute from Cebreros:")
    for item in az_el_data[-5:]:
        print("Azi: {0:.2f} deg, Elv: {1:.2f} deg".format(item["azi"], item["elv"]))

    last_item = az_el_data[-1]
    print("\nLast point: Azi={0:.2f} deg, Elv={1:.2f} deg".format(last_item["azi"], last_item["elv"]))


def fetch_horizons_text(params):
    # Send the Horizons request and return the raw text response.
    response = requests.get(HORIZONS_API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.text


def extract_az_el_data(response_text):
    # Read the ephemeris lines between the SOE and EOE markers.
    lines = response_text.splitlines()
    in_data = False
    az_el_data = []

    for line in lines:
        if "$$SOE" in line:
            in_data = True
            continue
        if "$$EOE" in line:
            break

        if not in_data or not line.strip() or line.startswith(" Date"):
            continue

        match = re.search(r"([-\d.]+)\s+([-\d.]+)", line, re.X)
        if not match:
            continue

        azi, elv = match.groups()
        az_el_data.append(
            {
                "azi": float(azi),
                "elv": float(elv),
                "line": line.strip(),
            }
        )

    return az_el_data


if __name__ == "__main__":
    main()
