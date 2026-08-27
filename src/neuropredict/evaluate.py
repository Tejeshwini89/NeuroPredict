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

from .config import config
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


def predict_segment(
    model,
    context,
    segment,
    sequence_length,
):
    """Predict a segment using preceding observations as context."""
    model.eval()

    values = np.concatenate([context, segment])
    predictions = []

    with torch.no_grad():
        for i in range(
            sequence_length,
            len(values),
        ):
            x = torch.tensor(
                values[i - sequence_length:i],
                dtype=torch.float32,
            ).unsqueeze(0)

            predictions.append(
                float(model(x).item())
            )

    return np.asarray(predictions)


def threshold_from_quantile(
    scores,
    quantile,
):
    """Calculate an empirical anomaly threshold."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "threshold_quantile must be between 0 and 1."
        )

    return float(
        np.quantile(
            scores,
            quantile,
        )
    )


def rolling_max_score(
    residuals,
    window,
):
    """
    Convert point-wise residuals into a causal rolling
    maximum anomaly score.

    A score at time t represents the largest residual
    observed from t-window+1 through t.

    This preserves temporal ordering and does not use
    future observations.
    """
    if window < 1:
        raise ValueError(
            "score window must be at least 1."
        )

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    scores = np.empty_like(residuals)

    for i in range(len(residuals)):
        start = max(
            0,
            i - window + 1,
        )

        scores[i] = np.max(
            residuals[start:i + 1]
        )

    return scores


def apply_temporal_persistence(
    scores,
    threshold,
    persistence,
):
    """
    Require a minimum number of consecutive threshold
    violations before declaring an anomaly.
    """
    if persistence < 1:
        raise ValueError(
            "persistence must be at least 1."
        )

    above_threshold = scores >= threshold

    anomaly_flags = np.zeros(
        len(scores),
        dtype=int,
    )

    consecutive = 0

    for i, is_above in enumerate(
        above_threshold
    ):
        if is_above:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= persistence:
            start = i - persistence + 1

            anomaly_flags[
                start:i + 1
            ] = 1

    return anomaly_flags


def validation_fpr(
    scores,
    threshold,
    persistence,
):
    """Calculate false-positive rate on normal validation data."""
    predictions = apply_temporal_persistence(
        scores,
        threshold,
        persistence,
    )

    return float(
        predictions.sum()
        / len(predictions)
    )


def evaluate_detector(
    test_scores,
    test_labels,
    threshold,
    persistence,
):
    """Evaluate one detector configuration on held-out test data."""
    predictions = apply_temporal_persistence(
        test_scores,
        threshold,
        persistence,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    return (
        float(precision),
        float(recall),
        float(f1),
        int(predictions.sum()),
    )


def main():
    dataset_config = config["dataset"]
    model_config = config["model"]
    anomaly_config = config["anomaly"]

    data_file = dataset_config["file"]

    sequence_length = int(
        model_config["sequence_length"]
    )

    configured_quantile = float(
        anomaly_config["threshold_quantile"]
    )

    configured_persistence = int(
        anomaly_config.get(
            "persistence",
            1,
        )
    )

    max_validation_fpr = float(
        anomaly_config.get(
            "max_validation_fpr",
            0.02,
        )
    )

    # ==================================================
    # DATA
    # ==================================================

    df = load_series(
        RAW / Path(data_file).name
    )

    windows = load_anomaly_windows(
        RAW / "combined_windows.json",
        data_file,
    )

    labels = make_anomaly_labels(
        df["timestamp"],
        windows,
    )

    splits = split_and_scale(
        df,
        labels,
    )

    # ==================================================
    # MODEL
    # ==================================================

    model = TransformerForecaster(
        d_model=int(
            model_config["d_model"]
        ),
        nhead=int(
            model_config["nhead"]
        ),
        num_layers=int(
            model_config["num_layers"]
        ),
        dropout=float(
            model_config["dropout"]
        ),
    )

    model.load_state_dict(
        torch.load(
            ARTIFACTS / "transformer_forecaster.pt",
            map_location="cpu",
        )
    )

    model.eval()

    # ==================================================
    # VALIDATION PREDICTIONS
    # ==================================================

    validation_predictions = predict_segment(
        model,
        splits.train[-sequence_length:],
        splits.validation,
        sequence_length,
    )

    validation_actual = splits.validation

    validation_residuals = np.abs(
        validation_actual
        - validation_predictions
    )

    # ==================================================
    # HELD-OUT TEST PREDICTIONS
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
        test_actual
        - test_predictions
    )

    # ==================================================
    # DETECTOR SEARCH
    # ==================================================

    threshold_quantiles = [
        0.900,
        0.910,
        0.920,
        0.930,
        0.940,
        0.950,
        0.960,
        0.970,
        0.980,
        0.990,
        0.995,
        0.999,
    ]

    if configured_quantile not in threshold_quantiles:
        threshold_quantiles.append(
            configured_quantile
        )

    threshold_quantiles = sorted(
        set(threshold_quantiles),
        reverse=True,
    )

    score_windows = [
        1,
        3,
        5,
        10,
    ]

    persistence_values = [
        1,
        2,
        3,
        5,
    ]

    if configured_persistence not in persistence_values:
        persistence_values.append(
            configured_persistence
        )

    persistence_values = sorted(
        set(persistence_values)
    )

    selection_results = []

    for score_window in score_windows:

        validation_scores = rolling_max_score(
            validation_residuals,
            score_window,
        )

        for quantile in threshold_quantiles:

            threshold = threshold_from_quantile(
                validation_scores,
                quantile,
            )

            for persistence in persistence_values:

                fpr = validation_fpr(
                    validation_scores,
                    threshold,
                    persistence,
                )

                selection_results.append(
                    {
                        "score_window": int(
                            score_window
                        ),
                        "quantile": float(
                            quantile
                        ),
                        "persistence": int(
                            persistence
                        ),
                        "threshold": float(
                            threshold
                        ),
                        "validation_fpr": float(
                            fpr
                        ),
                    }
                )

    # ==================================================
    # VALIDATION-ONLY SELECTION
    # ==================================================

    eligible = [
        result
        for result in selection_results
        if result["validation_fpr"]
        <= max_validation_fpr
    ]

    if not eligible:
        raise RuntimeError(
            "No detector configuration satisfies "
            f"max_validation_fpr="
            f"{max_validation_fpr:.4f}."
        )

    # Prefer the configuration that uses the available
    # validation false-positive budget most effectively.
    #
    # Tie-breaking:
    # 1. Highest allowed validation FPR
    # 2. Larger temporal score window
    # 3. Lower quantile
    # 4. Lower persistence
    selected = max(
        eligible,
        key=lambda result: (
            result["validation_fpr"],
            result["score_window"],
            -result["quantile"],
            -result["persistence"],
        ),
    )

    selected_score_window = selected[
        "score_window"
    ]

    selected_quantile = selected[
        "quantile"
    ]

    selected_persistence = selected[
        "persistence"
    ]

    selected_threshold = selected[
        "threshold"
    ]

    selected_validation_fpr = selected[
        "validation_fpr"
    ]

    # ==================================================
    # FINAL TEST EVALUATION
    # ==================================================

    selected_test_scores = rolling_max_score(
        test_residuals,
        selected_score_window,
    )

    (
        precision,
        recall,
        f1,
        predicted_anomalies,
    ) = evaluate_detector(
        selected_test_scores,
        test_labels,
        selected_threshold,
        selected_persistence,
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
    # CONFIGURED DETECTOR
    # ==================================================

    configured_score_window = int(
        anomaly_config.get(
            "score_window",
            1,
        )
    )

    configured_validation_scores = (
        rolling_max_score(
            validation_residuals,
            configured_score_window,
        )
    )

    configured_threshold = (
        threshold_from_quantile(
            configured_validation_scores,
            configured_quantile,
        )
    )

    configured_validation_fpr = (
        validation_fpr(
            configured_validation_scores,
            configured_threshold,
            configured_persistence,
        )
    )

    configured_test_scores = rolling_max_score(
        test_residuals,
        configured_score_window,
    )

    (
        configured_precision,
        configured_recall,
        configured_f1,
        configured_predicted,
    ) = evaluate_detector(
        configured_test_scores,
        test_labels,
        configured_threshold,
        configured_persistence,
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

    print(
        "\n=== Validation Detector Selection ==="
    )

    print(
        f"Maximum validation FPR : "
        f"{max_validation_fpr:.4f}"
    )

    print(
        f"Selected score window   : "
        f"{selected_score_window}"
    )

    print(
        f"Selected quantile       : "
        f"{selected_quantile:.3f}"
    )

    print(
        f"Selected persistence    : "
        f"{selected_persistence}"
    )

    print(
        f"Selected threshold      : "
        f"{selected_threshold:.6f}"
    )

    print(
        f"Validation FPR          : "
        f"{selected_validation_fpr:.4f}"
    )

    print(
        "\n=== Final Held-Out Test ==="
    )

    print(
        f"Test anomalies          : "
        f"{int(test_labels.sum())}"
    )

    print(
        f"Predicted anomalies     : "
        f"{predicted_anomalies}"
    )

    print(
        f"Precision               : "
        f"{precision:.4f}"
    )

    print(
        f"Recall                  : "
        f"{recall:.4f}"
    )

    print(
        f"F1                     : "
        f"{f1:.4f}"
    )

    print(
        "\n=== Configured Detector ==="
    )

    print(
        f"Configured score window : "
        f"{configured_score_window}"
    )

    print(
        f"Configured quantile     : "
        f"{configured_quantile:.3f}"
    )

    print(
        f"Configured persistence  : "
        f"{configured_persistence}"
    )

    print(
        f"Configured threshold    : "
        f"{configured_threshold:.6f}"
    )

    print(
        f"Configured validation FPR: "
        f"{configured_validation_fpr:.4f}"
    )

    print(
        f"Configured precision    : "
        f"{configured_precision:.4f}"
    )

    print(
        f"Configured recall       : "
        f"{configured_recall:.4f}"
    )

    print(
        f"Configured F1           : "
        f"{configured_f1:.4f}"
    )

    # ==================================================
    # SAVE RESULTS
    # ==================================================

    results = {
        "transformer_mae": transformer_mae,
        "transformer_rmse": transformer_rmse,
        "naive_mae": baseline_mae,
        "naive_rmse": baseline_rmse,

        "threshold_method": (
            "validation_empirical_quantile"
        ),

        "score_method": (
            "causal_rolling_max_residual"
        ),

        "selection_strategy": (
            "maximize_validation_fpr_under_budget"
        ),

        "max_validation_fpr": (
            max_validation_fpr
        ),

        "selected_score_window": (
            selected_score_window
        ),

        "selected_threshold_quantile": (
            selected_quantile
        ),

        "selected_persistence": (
            selected_persistence
        ),

        "selected_threshold": (
            selected_threshold
        ),

        "selected_validation_fpr": (
            selected_validation_fpr
        ),

        "test_anomalies": int(
            test_labels.sum()
        ),

        "predicted_anomalies": (
            predicted_anomalies
        ),

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "configured_score_window": (
            configured_score_window
        ),

        "configured_threshold_quantile": (
            configured_quantile
        ),

        "configured_persistence": (
            configured_persistence
        ),

        "configured_threshold": (
            configured_threshold
        ),

        "configured_validation_fpr": (
            configured_validation_fpr
        ),

        "configured_precision": (
            configured_precision
        ),

        "configured_recall": (
            configured_recall
        ),

        "configured_f1": (
            configured_f1
        ),

        "validation_selection_candidates": (
            selection_results
        ),
    }

    (
        ARTIFACTS / "evaluation.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print(
        "\nEvaluation results saved to "
        "artifacts/evaluation.json"
    )


if __name__ == "__main__":
    main()