"""
Mission name normalization helpers shared across CLI and GUI entrypoints.
"""

from pathlib import Path


# Mission names used as canonical values across the project.
MISSION_CANONICAL_NAMES = (
    "HERA",
    "JUICE",
    "SOLO",
    "BEPI",
    "MEX1",
    "EUCL",
)

# Alternative names accepted from folder names or user input.
MISSION_ALIASES = {
    "JUIC": "JUICE",
    "MEX": "MEX1",
}

# Folder names accepted for each canonical mission.
MISSION_DIRECTORY_CANDIDATES = {
    "HERA": ("HERA",),
    "JUICE": ("JUIC", "JUICE"),
    "SOLO": ("SOLO",),
    "BEPI": ("BEPI",),
    "MEX1": ("MEX1", "MEX"),
    "EUCL": ("EUCL",),
}


def normalize_mission_name(value):
    """
    Convert user input or folder names to the canonical mission name.
    """
    mission = str(value or "").strip().upper()
    return MISSION_ALIASES.get(mission, mission)


def accepted_mission_names_text():
    """
    Return the accepted mission names in a user-friendly format.
    """
    return "HERA, JUICE/JUIC, SOLO, BEPI, MEX1/MEX, EUCL"


def resolve_mission_dir(station_root, mission_value):
    """
    Find the first existing mission folder that matches the input name.
    """
    station_root = Path(station_root)
    mission_input = str(mission_value or "").strip().upper()
    canonical = normalize_mission_name(mission_input)

    candidate_names = []
    candidates = (mission_input,) + MISSION_DIRECTORY_CANDIDATES.get(canonical, (canonical,))

    for candidate in candidates:
        if candidate and candidate not in candidate_names:
            candidate_names.append(candidate)

    for candidate in candidate_names:
        mission_dir = station_root / candidate
        if mission_dir.exists():
            return canonical, mission_dir

    preferred_name = candidate_names[0] if candidate_names else canonical
    return canonical, station_root / preferred_name
