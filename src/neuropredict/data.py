from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TimeSeriesSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_timestamps: pd.Series
    validation_timestamps: pd.Series
    test_timestamps: pd.Series
    test_anomaly_labels: np.ndarray
    validation_anomaly_labels: np.ndarray
    scaler: StandardScaler


def load_series(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    expected = {"timestamp", "value"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, got {set(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "value"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    if len(df) < 500:
        raise ValueError("Dataset is unexpectedly small.")

    return df


def load_anomaly_windows(labels_path: Path, dataset_key: str):
    payload = json.loads(labels_path.read_text())
    windows = payload.get(dataset_key, [])

    return [
        (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
        for start, end in windows
    ]


def make_anomaly_labels(timestamps: pd.Series, windows) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=np.int64)

    for start, end in windows:
        mask = (timestamps >= start) & (timestamps <= end)
        labels[mask.to_numpy()] = 1

    return labels


def split_and_scale(df, labels):
    # Chronological, anomaly-aware split.
    # First anomaly: 1396-1596
    # Second anomaly: 2559-2759
    train_end = 1396
    validation_start = 1597
    test_start = 2559
    test_end = 2760

    train_raw = df["value"].to_numpy()[:train_end]
    validation_raw = df["value"].to_numpy()[validation_start:test_start]
    test_raw = df["value"].to_numpy()[test_start:test_end]

    # Fit scaler ONLY on normal training data.
    normal_train = train_raw[labels[:train_end] == 0]

    scaler = StandardScaler()
    scaler.fit(normal_train.reshape(-1, 1))

    def scale(values):
        return scaler.transform(values.reshape(-1, 1)).ravel()

    return TimeSeriesSplits(
        train=scale(train_raw),
        validation=scale(validation_raw),
        test=scale(test_raw),
        train_timestamps=df["timestamp"].iloc[:train_end].reset_index(drop=True),
        validation_timestamps=df["timestamp"].iloc[
            validation_start:test_start
        ].reset_index(drop=True),
        test_timestamps=df["timestamp"].iloc[
            test_start:test_end
        ].reset_index(drop=True),
        validation_anomaly_labels=labels[
            validation_start:test_start
        ],
        test_anomaly_labels=labels[test_start:test_end],
        scaler=scaler,
    )


def make_windows(values: np.ndarray, sequence_length: int):
    if len(values) <= sequence_length:
        raise ValueError("Not enough observations for the sequence length.")

    x, y = [], []

    for i in range(len(values) - sequence_length):
        x.append(values[i:i + sequence_length])
        y.append(values[i + sequence_length])

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )