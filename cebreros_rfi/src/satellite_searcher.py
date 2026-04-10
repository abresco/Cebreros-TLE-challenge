import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skyfield.api import Topos, load


# Ground station used in this visibility prototype.
STATION = Topos(
    latitude_degrees=40.4527,
    longitude_degrees=-4.3676,
    elevation_m=794,
)

# Example target time to inspect.
TARGET_TIME = datetime(2025, 11, 30, 11, 57, 0, tzinfo=timezone.utc)

# TLE files searched relative to this script.
TLE_FILES = ["active.tle", "geo.tle", "starlink.tle"]


def main():
    # Create the time grid around the target time.
    ts = load.timescale()
    times_list = build_time_window(TARGET_TIME, minutes=10, step_seconds=30)
    times = ts.from_datetimes(times_list)

    print("\nAnalyzing satellites visible from Cebreros at {0} +-10 minutes\n".format(TARGET_TIME))

    # Load all available TLE files from the local data folder.
    satellites = load_local_satellites()
    print("\nTotal satellites loaded: {0}\n".format(len(satellites)))

    # Compute the visible points and write them to CSV.
    output_path = Path("satellite_pass_results.csv")
    visible_rows, visible_count, rfi_candidates = analyze_visibility(satellites, times)
    write_results_csv(output_path, visible_rows)

    print("\n----------------------------------")
    print("Analysis Summary")
    print("----------------------------------")
    print("Total visible satellite points : {0}".format(visible_count))
    print("Potential RFI candidates       : {0}".format(rfi_candidates))
    print("Results saved to {0}".format(output_path))
    print("----------------------------------")


def build_time_window(target_time, minutes, step_seconds):
    # Build the list of timestamps around the target instant.
    start_time = target_time - timedelta(minutes=minutes)
    end_time = target_time + timedelta(minutes=minutes)

    times_list = []
    current = start_time

    while current <= end_time:
        times_list.append(current)
        current += timedelta(seconds=step_seconds)

    return times_list


def load_local_satellites():
    # Load all satellites found in the configured local TLE files.
    base_dir = Path(__file__).resolve().parent
    satellites = []

    for file_name in TLE_FILES:
        tle_path = base_dir.parent / "Data" / file_name
        if not tle_path.exists():
            continue

        loaded = load.tle_file(str(tle_path))
        satellites.extend(loaded)
        print("Loaded {0} satellites from {1}".format(len(loaded), file_name))

    return satellites


def analyze_visibility(satellites, times):
    # Propagate each satellite and keep only visible points.
    visible_rows = []
    visible_count = 0
    rfi_candidates = 0

    for satellite in satellites:
        difference = satellite - STATION
        topocentric = difference.at(times)
        alt, az, distance = topocentric.altaz()

        for index in range(len(times)):
            elevation = alt.degrees[index]
            if elevation <= 0:
                continue

            visible_count += 1
            rfi_flag = classify_rfi_risk(elevation)

            if rfi_flag in {"HIGH", "MEDIUM"}:
                rfi_candidates += 1

            row = {
                "Satellite": satellite.name,
                "Time (UTC)": times[index].utc_strftime("%H:%M:%S"),
                "Azimuth (deg)": "{0:.2f}".format(az.degrees[index]),
                "Elevation (deg)": "{0:.2f}".format(elevation),
                "Distance (km)": "{0:.2f}".format(distance.km[index]),
                "RFI Risk": rfi_flag,
            }
            visible_rows.append(row)

            print(
                "{0} | {1} UTC | Az={2:.2f} deg | El={3:.2f} deg | Dist={4:.2f} km | RFI Risk={5}".format(
                    satellite.name,
                    row["Time (UTC)"],
                    az.degrees[index],
                    elevation,
                    distance.km[index],
                    rfi_flag,
                )
            )

    return visible_rows, visible_count, rfi_candidates


def classify_rfi_risk(elevation):
    # Map elevation to a simple visibility-based risk label.
    if elevation > 30:
        return "HIGH"
    if elevation > 10:
        return "MEDIUM"
    return "LOW"


def write_results_csv(output_path, rows):
    # Save the visible points into a CSV file.
    fieldnames = [
        "Satellite",
        "Time (UTC)",
        "Azimuth (deg)",
        "Elevation (deg)",
        "Distance (km)",
        "RFI Risk",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
