from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from .data import (
    load_anomaly_windows,
    load_series,
    make_anomaly_labels,
    split_and_scale,
)
from .model import TransformerForecaster


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"


def predict_segment(model, context, segment, sequence_length):
    """Predict a segment using preceding observations as context."""
    model.eval()

    values = np.concatenate([context, segment])
    predictions = []

    with torch.no_grad():
        for i in range(sequence_length, len(values)):
            x = torch.tensor(
                values[i - sequence_length:i],
                dtype=torch.float32,
            ).unsqueeze(0)

            predictions.append(float(model(x).item()))

    return np.asarray(predictions)


def robust_threshold(residuals, false_positive_rate=0.01):
    """
    Estimate an anomaly threshold from normal validation residuals.

    Uses the empirical quantile of the validation residual distribution.
    The validation set contains no labeled anomalies, so threshold
    selection is based only on normal behavior.
    """
    quantile = 1.0 - false_positive_rate

    return float(
        np.quantile(
            residuals,
            quantile,
        )
    )


def main():
    df = load_series(
        RAW / "ec2_cpu_utilization_53ea38.csv"
    )

    windows = load_anomaly_windows(
        RAW / "combined_windows.json",
        "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    )

    labels = make_anomaly_labels(
        df["timestamp"],
        windows,
    )

    splits = split_and_scale(df, labels)

    sequence_length = 48

    model = TransformerForecaster()

    model.load_state_dict(
        torch.load(
            ARTIFACTS / "transformer_forecaster.pt",
            map_location="cpu",
        )
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    validation_predictions = predict_segment(
        model,
        splits.train[-sequence_length:],
        splits.validation,
        sequence_length,
    )

    validation_actual = splits.validation

    validation_residuals = np.abs(
        validation_actual - validation_predictions
    )

    # Validation contains only normal observations.
    # Select threshold using normal validation behavior.
    false_positive_rate = 0.01

    threshold = robust_threshold(
        validation_residuals,
        false_positive_rate,
    )

    validation_flags = (
        validation_residuals >= threshold
    ).astype(int)

    validation_false_positives = int(
        validation_flags.sum()
    )

    validation_rate = (
        validation_false_positives
        / len(validation_residuals)
    )

    # ==================================================
    # HELD-OUT TEST
    # ==================================================

    test_predictions = predict_segment(
        model,
        splits.validation[-sequence_length:],
        splits.test,
        sequence_length,
    )

    test_actual = splits.test
    test_labels = splits.test_anomaly_labels

    test_residuals = np.abs(
        test_actual - test_predictions
    )

    anomaly_pred = (
        test_residuals >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            anomaly_pred,
            average="binary",
            zero_division=0,
        )
    )

    # ==================================================
    # FORECASTING METRICS
    # ==================================================

    transformer_mae = float(
        mean_absolute_error(
            test_actual,
            test_predictions,
        )
    )

    transformer_rmse = float(
        np.sqrt(
            mean_squared_error(
                test_actual,
                test_predictions,
            )
        )
    )

    # Naive persistence baseline.
    baseline = np.concatenate(
        [
            splits.validation[-1:],
            splits.test[:-1],
        ]
    )

    baseline_mae = float(
        mean_absolute_error(
            test_actual,
            baseline,
        )
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                test_actual,
                baseline,
            )
        )
    )

    # ==================================================
    # RESULTS
    # ==================================================

    print("\n=== Forecasting ===")
    print(
        f"Transformer MAE : "
        f"{transformer_mae:.6f}"
    )
    print(
        f"Transformer RMSE: "
        f"{transformer_rmse:.6f}"
    )
    print(
        f"Naive MAE       : "
        f"{baseline_mae:.6f}"
    )
    print(
        f"Naive RMSE      : "
        f"{baseline_rmse:.6f}"
    )

    print("\n=== Residual anomaly detection ===")
    print(
        f"Threshold method       : "
        f"Validation 99th percentile"
    )
    print(
        f"Validation threshold   : "
        f"{threshold:.6f}"
    )
    print(
        f"Validation FP rate     : "
        f"{validation_rate:.4f}"
    )
    print(
        f"Test anomalies         : "
        f"{int(test_labels.sum())}"
    )
    print(
        f"Predicted anomalies    : "
        f"{int(anomaly_pred.sum())}"
    )
    print(
        f"Precision              : "
        f"{precision:.4f}"
    )
    print(
        f"Recall                 : "
        f"{recall:.4f}"
    )
    print(
        f"F1                     : "
        f"{f1:.4f}"
    )

    results = {
        "transformer_mae": transformer_mae,
        "transformer_rmse": transformer_rmse,
        "naive_mae": baseline_mae,
        "naive_rmse": baseline_rmse,
        "threshold_method": "validation_99th_percentile",
        "validation_false_positive_rate": validation_rate,
        "anomaly_threshold": threshold,
        "test_anomalies": int(test_labels.sum()),
        "predicted_anomalies": int(
            anomaly_pred.sum()
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    (
        ARTIFACTS / "evaluation.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()