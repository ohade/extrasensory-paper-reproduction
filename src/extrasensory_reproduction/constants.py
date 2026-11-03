"""Constants transcribed from the paper and official dataset artifacts."""

from __future__ import annotations

from pathlib import Path

DATA_URL = (
    "http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip"
)
DATA_ARCHIVE_SHA256 = "412d307bdf54a60dff81a705c2f25833889a46dca64a184e3b588ad99c69cb56"
DATA_ARCHIVE_NAME = "ExtraSensory.per_uuid_features_labels.zip"
ORIGINAL_LABELS_URL = (
    "http://extrasensory.ucsd.edu/data/additional_data_files/"
    "ExtraSensory.per_uuid_original_labels.zip"
)
ORIGINAL_LABELS_ARCHIVE_SHA256 = "7c017f75cbf3ca6d157a6ecfd8aacb1173493d9864772ab648ba27ac81373e79"
ORIGINAL_LABELS_ARCHIVE_NAME = "ExtraSensory.per_uuid_original_labels.zip"

DEFAULT_RAW_DIR = Path("data/raw/features_labels")
DEFAULT_ARCHIVE_PATH = Path("data/raw") / DATA_ARCHIVE_NAME
DEFAULT_ORIGINAL_LABELS_DIR = Path("data/raw/original_labels")
DEFAULT_ORIGINAL_LABELS_ARCHIVE_PATH = Path("data/raw") / ORIGINAL_LABELS_ARCHIVE_NAME
DEFAULT_CACHE_PATH = Path("data/processed/core_dataset.npz")
DEFAULT_FOLDS_DIR = Path("data/reference/cv_5_folds")
DEFAULT_RESULTS_DIR = Path("results")

SENSOR_ORDER = ("Acc", "Gyro", "WAcc", "Loc", "Aud", "PS")

SENSOR_PREFIXES = {
    "Acc": ("raw_acc:",),
    "Gyro": ("proc_gyro:",),
    "WAcc": ("watch_acceleration:",),
    "Loc": ("location:", "location_quick_features:"),
    "Aud": ("audio_naive:",),
    "PS": ("discrete:",),
}

EXPECTED_SENSOR_FEATURE_COUNTS = {
    "Acc": 26,
    "Gyro": 26,
    "WAcc": 46,
    "Loc": 17,
    "Aud": 26,
    "PS": 34,
}

# Reconstruct the three columns whose values drifted when the public primary
# files gained missing-label semantics after the 2017 paper.
OR_INDOORS_ORIGINAL_COMPONENTS = (
    "INDOORS",
    "SLEEPING",
    "TOILET",
    "BATHING_-_BATH",
    "BATHING_-_SHOWER",
    "IN_CLASS",
    "AT_A_BAR",
    "AT_THE_GYM",
    "ELEVATOR",
)
OR_OUTSIDE_ORIGINAL_COMPONENTS = (
    "OUTSIDE",
    "SKATEBOARDING",
    "PLAYING_BASEBALL",
    "PLAYING_FRISBEE",
    "GARDENING",
    "RAKING_LEAVES",
    "STROLLING",
    "HIKING",
    "AT_THE_BEACH",
    "AT_SEA",
    "MOTORBIKE",
)
RECONSTRUCTED_PAPER_LABELS = (
    "OR_indoors",
    "OR_outside",
    "PHONE_IN_POCKET",
)

# These are the first 25 labels in the public files and the rows used for the
# headline average in Figure 3 / Supplementary Table 1.
HEADLINE_LABELS = (
    "LYING_DOWN",
    "SITTING",
    "FIX_walking",
    "FIX_running",
    "BICYCLING",
    "SLEEPING",
    "LAB_WORK",
    "IN_CLASS",
    "IN_A_MEETING",
    "LOC_main_workplace",
    "OR_indoors",
    "OR_outside",
    "IN_A_CAR",
    "ON_A_BUS",
    "DRIVE_-_I_M_THE_DRIVER",
    "DRIVE_-_I_M_A_PASSENGER",
    "LOC_home",
    "FIX_restaurant",
    "PHONE_IN_POCKET",
    "OR_exercise",
    "COOKING",
    "SHOPPING",
    "STROLLING",
    "DRINKING__ALCOHOL_",
    "BATHING_-_SHOWER",
)

PRETTY_LABELS = {
    "LYING_DOWN": "Lying down",
    "SITTING": "Sitting",
    "FIX_walking": "Walking",
    "FIX_running": "Running",
    "BICYCLING": "Bicycling",
    "SLEEPING": "Sleeping",
    "LAB_WORK": "Lab work",
    "IN_CLASS": "In class",
    "IN_A_MEETING": "In a meeting",
    "LOC_main_workplace": "At main workplace",
    "OR_indoors": "Indoors",
    "OR_outside": "Outside",
    "IN_A_CAR": "In a car",
    "ON_A_BUS": "On a bus",
    "DRIVE_-_I_M_THE_DRIVER": "Drive (I'm the driver)",
    "DRIVE_-_I_M_A_PASSENGER": "Drive (I'm a passenger)",
    "LOC_home": "At home",
    "FIX_restaurant": "At a restaurant",
    "PHONE_IN_POCKET": "Phone in pocket",
    "OR_exercise": "Exercise",
    "COOKING": "Cooking",
    "SHOPPING": "Shopping",
    "STROLLING": "Strolling",
    "DRINKING__ALCOHOL_": "Drinking (alcohol)",
    "BATHING_-_SHOWER": "Bathing - shower",
}

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_RANDOM_SEED = 2017
