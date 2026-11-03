"""Loading, caching, profiling, and validating ExtraSensory feature files."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DATA_ARCHIVE_SHA256,
    DATA_URL,
    DEFAULT_ARCHIVE_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_FOLDS_DIR,
    DEFAULT_ORIGINAL_LABELS_ARCHIVE_PATH,
    DEFAULT_ORIGINAL_LABELS_DIR,
    DEFAULT_RAW_DIR,
    EXPECTED_SENSOR_FEATURE_COUNTS,
    OR_INDOORS_ORIGINAL_COMPONENTS,
    OR_OUTSIDE_ORIGINAL_COMPONENTS,
    ORIGINAL_LABELS_ARCHIVE_SHA256,
    ORIGINAL_LABELS_URL,
    RECONSTRUCTED_PAPER_LABELS,
    SENSOR_ORDER,
    SENSOR_PREFIXES,
)


@dataclass(frozen=True)
class ExtraSensoryDataset:
    """In-memory representation limited to the six sensor groups in the paper."""

    features: np.ndarray
    labels: np.ndarray
    label_missing: np.ndarray
    label_source: np.ndarray
    timestamps: np.ndarray
    user_index: np.ndarray
    user_ids: np.ndarray
    feature_names: np.ndarray
    feature_sensors: np.ndarray
    label_names: np.ndarray

    def sensor_columns(self, sensor: str) -> np.ndarray:
        if sensor == "EF":
            return np.arange(self.features.shape[1])
        return np.flatnonzero(self.feature_sensors == sensor)

    def sensor_available(self, sensor: str) -> np.ndarray:
        if sensor == "EF":
            sensor_masks = [self.sensor_available(single_sensor) for single_sensor in SENSOR_ORDER]
            return np.column_stack(sensor_masks).all(axis=1)
        columns = self.sensor_columns(sensor)
        if sensor == "Loc":
            columns = np.asarray(
                [index for index in columns if self.feature_names[index].startswith("location:")]
            )
        return np.isfinite(self.features[:, columns]).any(axis=1)

    @property
    def core_available(self) -> np.ndarray:
        return self.paper_labeled & self.sensor_available("EF")

    @property
    def paper_labeled(self) -> np.ndarray:
        """Minutes included in the 2017 paper's 308,320 labeled examples."""

        return self.label_source != -1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_extract(
    *,
    url: str,
    archive_path: Path,
    expected_sha256: str,
    extract_dir: Path,
    member_suffix: str,
    expected_members: int,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent": "extrasensory-reproduction/0.1"}
        )
        with urllib.request.urlopen(request) as response, archive_path.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)

    actual_hash = sha256_file(archive_path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"Archive checksum mismatch: expected {expected_sha256}, got {actual_hash}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(member_suffix)]
        if len(members) != expected_members:
            raise ValueError(
                f"Expected {expected_members} files ending {member_suffix}, found {len(members)}"
            )
        archive.extractall(extract_dir)


def download_dataset(
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    original_archive_path: Path = DEFAULT_ORIGINAL_LABELS_ARCHIVE_PATH,
    original_labels_dir: Path = DEFAULT_ORIGINAL_LABELS_DIR,
) -> None:
    """Download and verify both official archives required for reproduction."""

    _download_and_extract(
        url=DATA_URL,
        archive_path=archive_path,
        expected_sha256=DATA_ARCHIVE_SHA256,
        extract_dir=raw_dir,
        member_suffix=".features_labels.csv.gz",
        expected_members=60,
    )
    _download_and_extract(
        url=ORIGINAL_LABELS_URL,
        archive_path=original_archive_path,
        expected_sha256=ORIGINAL_LABELS_ARCHIVE_SHA256,
        extract_dir=original_labels_dir,
        member_suffix=".original_labels.csv.gz",
        expected_members=60,
    )


def sensor_for_feature(feature_name: str) -> str | None:
    for sensor, prefixes in SENSOR_PREFIXES.items():
        if feature_name.startswith(prefixes):
            return sensor
    return None


def _read_header(path: Path) -> tuple[list[str], list[str]]:
    columns = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    first_label = next(index for index, column in enumerate(columns) if column.startswith("label:"))
    feature_names = columns[1:first_label]
    label_names = [column.removeprefix("label:") for column in columns[first_label:-1]]
    return feature_names, label_names


def load_raw_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    original_labels_dir: Path = DEFAULT_ORIGINAL_LABELS_DIR,
) -> ExtraSensoryDataset:
    paths = sorted(raw_dir.glob("*.features_labels.csv.gz"))
    if len(paths) != 60:
        raise FileNotFoundError(
            f"Expected 60 files in {raw_dir}; found {len(paths)}. Run the download command."
        )
    original_paths = sorted(original_labels_dir.glob("*.original_labels.csv.gz"))
    if len(original_paths) != 60:
        raise FileNotFoundError(
            f"Expected 60 files in {original_labels_dir}; found {len(original_paths)}. "
            "Run the download command."
        )

    all_feature_names, label_names = _read_header(paths[0])
    core_feature_names = [
        feature for feature in all_feature_names if sensor_for_feature(feature) is not None
    ]
    feature_sensors = [sensor_for_feature(feature) for feature in core_feature_names]

    counts = {sensor: feature_sensors.count(sensor) for sensor in SENSOR_ORDER}
    if counts != EXPECTED_SENSOR_FEATURE_COUNTS:
        raise ValueError(f"Unexpected core feature schema: {counts}")

    label_columns = [f"label:{label}" for label in label_names]
    label_lookup = {label: index for index, label in enumerate(label_names)}
    use_columns = ["timestamp", *core_feature_names, *label_columns, "label_source"]
    original_label_names = {
        *OR_INDOORS_ORIGINAL_COMPONENTS,
        *OR_OUTSIDE_ORIGINAL_COMPONENTS,
        "PHONE_IN_POCKET",
    }
    original_use_columns = [
        "timestamp",
        *(f"original_label:{label}" for label in sorted(original_label_names)),
    ]
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    missing_parts: list[np.ndarray] = []
    label_source_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    user_parts: list[np.ndarray] = []
    user_ids: list[str] = []

    feature_dtypes = {column: "float32" for column in core_feature_names}
    label_dtypes = {column: "float32" for column in label_columns}

    for user_number, path in enumerate(paths):
        user_id = path.name.split(".features_labels", 1)[0]
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=use_columns,
            dtype={**feature_dtypes, **label_dtypes},
        )
        original_path = original_labels_dir / f"{user_id}.original_labels.csv.gz"
        original = pd.read_csv(
            original_path,
            compression="gzip",
            usecols=original_use_columns,
            dtype={column: "uint8" for column in original_use_columns if column != "timestamp"},
        )
        np.testing.assert_array_equal(
            frame["timestamp"].to_numpy(dtype=np.int64),
            original["timestamp"].to_numpy(dtype=np.int64),
            err_msg=f"Timestamp mismatch for user {user_id}",
        )
        raw_labels = frame[label_columns].to_numpy(dtype=np.float32, copy=True)
        current_missing = np.isnan(raw_labels)
        paper_labels = np.nan_to_num(raw_labels, nan=0.0).astype(np.uint8)
        original_indoors = original[
            [f"original_label:{label}" for label in OR_INDOORS_ORIGINAL_COMPONENTS]
        ].any(axis=1)
        paper_labels[:, label_lookup["OR_indoors"]] = np.logical_or(
            original_indoors.to_numpy(),
            paper_labels[:, label_lookup["LOC_home"]],
        )
        paper_labels[:, label_lookup["OR_outside"]] = original[
            [f"original_label:{label}" for label in OR_OUTSIDE_ORIGINAL_COMPONENTS]
        ].any(axis=1)
        paper_labels[:, label_lookup["PHONE_IN_POCKET"]] = original[
            "original_label:PHONE_IN_POCKET"
        ].to_numpy()
        feature_parts.append(frame[core_feature_names].to_numpy(dtype=np.float32, copy=True))
        label_parts.append(paper_labels)
        # Retain the newer archive's missing-label mask for the optional
        # exclude-missing sensitivity analysis.
        missing_parts.append(current_missing)
        label_source_parts.append(frame["label_source"].to_numpy(dtype=np.int8, copy=True))
        timestamp_parts.append(frame["timestamp"].to_numpy(dtype=np.int64, copy=True))
        user_parts.append(np.full(len(frame), user_number, dtype=np.int16))
        user_ids.append(user_id)

    return ExtraSensoryDataset(
        features=np.concatenate(feature_parts),
        labels=np.concatenate(label_parts),
        label_missing=np.concatenate(missing_parts),
        label_source=np.concatenate(label_source_parts),
        timestamps=np.concatenate(timestamp_parts),
        user_index=np.concatenate(user_parts),
        user_ids=np.asarray(user_ids, dtype="U36"),
        feature_names=np.asarray(core_feature_names, dtype="U120"),
        feature_sensors=np.asarray(feature_sensors, dtype="U5"),
        label_names=np.asarray(label_names, dtype="U40"),
    )


def save_dataset(dataset: ExtraSensoryDataset, cache_path: Path = DEFAULT_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=dataset.features,
        labels=dataset.labels,
        label_missing=dataset.label_missing,
        label_source=dataset.label_source,
        timestamps=dataset.timestamps,
        user_index=dataset.user_index,
        user_ids=dataset.user_ids,
        feature_names=dataset.feature_names,
        feature_sensors=dataset.feature_sensors,
        label_names=dataset.label_names,
    )


def load_cached_dataset(cache_path: Path = DEFAULT_CACHE_PATH) -> ExtraSensoryDataset:
    with np.load(cache_path, allow_pickle=False) as arrays:
        return ExtraSensoryDataset(
            features=arrays["features"],
            labels=arrays["labels"],
            label_missing=arrays["label_missing"],
            label_source=arrays["label_source"],
            timestamps=arrays["timestamps"],
            user_index=arrays["user_index"],
            user_ids=arrays["user_ids"],
            feature_names=arrays["feature_names"],
            feature_sensors=arrays["feature_sensors"],
            label_names=arrays["label_names"],
        )


def load_or_prepare(
    cache_path: Path = DEFAULT_CACHE_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    original_labels_dir: Path = DEFAULT_ORIGINAL_LABELS_DIR,
) -> ExtraSensoryDataset:
    if cache_path.exists():
        return load_cached_dataset(cache_path)
    dataset = load_raw_dataset(raw_dir, original_labels_dir)
    save_dataset(dataset, cache_path)
    return dataset


def load_official_folds(
    user_ids: Iterable[str],
    folds_dir: Path = DEFAULT_FOLDS_DIR,
) -> list[dict[str, np.ndarray]]:
    user_ids_array = np.asarray(list(user_ids))
    user_to_index = {user: index for index, user in enumerate(user_ids_array)}
    folds: list[dict[str, np.ndarray]] = []

    for fold in range(5):
        split: dict[str, np.ndarray] = {}
        for partition in ("train", "test"):
            uuids: list[str] = []
            for platform in ("iphone", "android"):
                path = folds_dir / f"fold_{fold}_{partition}_{platform}_uuids.txt"
                uuids.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
            unknown = sorted(set(uuids) - set(user_to_index))
            if unknown:
                raise ValueError(f"Fold {fold} contains unknown users: {unknown}")
            split[partition] = np.asarray([user_to_index[uuid] for uuid in uuids], dtype=np.int16)
        folds.append(split)

    all_users = set(range(len(user_ids_array)))
    for fold, split in enumerate(folds):
        train_users = set(split["train"].tolist())
        test_users = set(split["test"].tolist())
        if train_users & test_users:
            raise ValueError(f"Fold {fold} has train/test user overlap")
        if train_users | test_users != all_users:
            raise ValueError(f"Fold {fold} does not cover all users")
        if len(train_users) != 48 or len(test_users) != 12:
            raise ValueError(f"Fold {fold} is not a 48/12 split")

    test_assignments = np.concatenate([split["test"] for split in folds])
    if sorted(test_assignments.tolist()) != list(range(len(user_ids_array))):
        raise ValueError("Each user must occur in exactly one test fold")
    return folds


def dataset_profile(
    dataset: ExtraSensoryDataset,
    folds: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    core_mask = dataset.core_available
    paper_labeled = dataset.paper_labeled
    label_missing_rates = dataset.label_missing[paper_labeled].mean(axis=0)
    per_sensor = {}
    for sensor in SENSOR_ORDER:
        mask = dataset.sensor_available(sensor) & paper_labeled
        per_sensor[sensor] = {
            "features": int(np.sum(dataset.feature_sensors == sensor)),
            "available_examples": int(mask.sum()),
            "available_users": int(np.unique(dataset.user_index[mask]).size),
            "missing_rate": float(1.0 - mask.sum() / paper_labeled.sum()),
        }

    fold_rows = []
    for fold_index, split in enumerate(folds):
        test_mask = np.isin(dataset.user_index, split["test"]) & core_mask
        fold_rows.append(
            {
                "fold": fold_index,
                "train_users": len(split["train"]),
                "test_users": len(split["test"]),
                "core_test_examples": int(test_mask.sum()),
                "core_test_users": int(np.unique(dataset.user_index[test_mask]).size),
            }
        )

    return {
        "examples": len(dataset.timestamps),
        "paper_labeled_examples": int(paper_labeled.sum()),
        "additional_unlabeled_examples": int((~paper_labeled).sum()),
        "users": len(dataset.user_ids),
        "labels": len(dataset.label_names),
        "core_features": len(dataset.feature_names),
        "core_complete_examples": int(core_mask.sum()),
        "core_complete_users": int(np.unique(dataset.user_index[core_mask]).size),
        "paper_label_reconstruction": list(RECONSTRUCTED_PAPER_LABELS),
        "timestamp_min_utc": pd.to_datetime(
            dataset.timestamps.min(), unit="s", utc=True
        ).isoformat(),
        "timestamp_max_utc": pd.to_datetime(
            dataset.timestamps.max(), unit="s", utc=True
        ).isoformat(),
        "duplicate_user_timestamps": int(
            pd.DataFrame({"user": dataset.user_index, "timestamp": dataset.timestamps})
            .duplicated()
            .sum()
        ),
        "sensor_coverage": per_sensor,
        "label_missing_rate": {
            label: float(rate)
            for label, rate in zip(dataset.label_names.tolist(), label_missing_rates)
        },
        "folds": fold_rows,
    }


def write_profile(profile: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
