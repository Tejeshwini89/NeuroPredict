import numpy as np
import pandas as pd

from src.neuropredict.data import make_anomaly_labels, make_windows


def test_make_windows():
    x, y = make_windows(
        np.arange(10, dtype=float),
        sequence_length=3,
    )
    assert x.shape == (7, 3)
    assert y.shape == (7,)
    assert np.array_equal(
        y,
        np.array([3, 4, 5, 6, 7, 8, 9]),
    )


def test_anomaly_labels():
    timestamps = pd.Series(
        pd.date_range(
            "2024-01-01",
            periods=5,
            freq="h",
            tz="UTC",
        )
    )
    windows = [(timestamps.iloc[1], timestamps.iloc[2])]
    labels = make_anomaly_labels(timestamps, windows)
    assert labels.tolist() == [0, 1, 1, 0, 0]
