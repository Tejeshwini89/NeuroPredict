from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import config
from .detector import PersistenceDetector
from .model import TransformerForecaster
from .servicenow import ServiceNowClient


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")


MODEL_PATH = ROOT / "artifacts" / "transformer_forecaster.pt"
SCALER_PATH = ROOT / "artifacts" / "scaler.pt"
EVAL_PATH = ROOT / "artifacts" / "evaluation.json"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="NeuroPredict API",
    version="0.4.0",
    description=(
        "Transformer-based predictive AIOps inference, "
        "stateful anomaly detection, and ServiceNow "
        "incident automation API."
    ),
)


# --------------------------------------------------
# Runtime state
# --------------------------------------------------

_model = None
_scaler = None
_detector = None
_detector_config = None


# --------------------------------------------------
# Request / response models
# --------------------------------------------------

class PredictionRequest(BaseModel):
    values: list[float] = Field(
        min_length=48,
        description="Latest telemetry values. The last 48 values are used.",
    )

    observed_next_value: float | None = Field(
        default=None,
        description="Observed next telemetry value.",
    )

    create_incident: bool = Field(
        default=False,
        description=(
            "Create a ServiceNow incident when a "
            "persistent anomaly is confirmed."
        ),
    )


class PredictionResponse(BaseModel):
    prediction: float

    residual: float | None = None
    anomaly_score: float | None = None
    anomaly: bool

    threshold: float | None = None
    persistence: int | None = None
    score_window: int | None = None
    consecutive_anomalies: int | None = None

    incident_created: bool = False
    incident_number: str | None = None
    incident_error: str | None = None


# --------------------------------------------------
# Load model, scaler, and detector
# --------------------------------------------------

def get_components():
    global _model
    global _scaler
    global _detector
    global _detector_config

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts missing. Run training first.",
        )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    if _model is None:
        _model = TransformerForecaster(
            d_model=int(config["model"]["d_model"]),
            nhead=int(config["model"]["nhead"]),
            num_layers=int(config["model"]["num_layers"]),
            dropout=float(config["model"]["dropout"]),
        )

        _model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu",
            )
        )

        _model.eval()

    # --------------------------------------------------
    # Load scaler
    # --------------------------------------------------

    if _scaler is None:
        _scaler = torch.load(
            SCALER_PATH,
            map_location="cpu",
            weights_only=False,
        )

    # --------------------------------------------------
    # Load selected detector configuration
    # --------------------------------------------------

    if _detector_config is None:

        if not EVAL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Evaluation results missing. "
                    "Run evaluation before starting inference."
                ),
            )

        evaluation = json.loads(
            EVAL_PATH.read_text(
                encoding="utf-8"
            )
        )

        required_keys = [
            "selected_threshold",
            "selected_persistence",
            "selected_score_window",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in evaluation
        ]

        if missing_keys:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Evaluation results are missing required "
                    f"detector configuration: {missing_keys}"
                ),
            )

        _detector_config = {
            "threshold": float(
                evaluation["selected_threshold"]
            ),
            "persistence": int(
                evaluation["selected_persistence"]
            ),
            "score_window": int(
                evaluation["selected_score_window"]
            ),
        }

    # --------------------------------------------------
    # Initialize stateful detector
    # --------------------------------------------------

    if _detector is None:
        _detector = PersistenceDetector(
            threshold=_detector_config["threshold"],
            persistence=_detector_config["persistence"],
        )

    return (
        _model,
        _scaler,
        _detector,
        _detector_config,
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    model_ready = (
        MODEL_PATH.exists()
        and SCALER_PATH.exists()
    )

    evaluation_ready = EVAL_PATH.exists()

    return {
        "status": "ok",
        "model_ready": model_ready,
        "evaluation_ready": evaluation_ready,
        "servicenow_configured": all(
            [
                "SERVICENOW_INSTANCE" in os.environ,
                "SERVICENOW_USER" in os.environ,
                "SERVICENOW_PASSWORD" in os.environ,
            ]
        ),
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    (
        model,
        scaler,
        detector,
        detector_config,
    ) = get_components()

    mean = float(
        scaler["mean"]
    )

    scale = float(
        scaler["scale"]
    )

    threshold = float(
        detector_config["threshold"]
    )

    persistence = int(
        detector_config["persistence"]
    )

    score_window = int(
        detector_config["score_window"]
    )

    sequence_length = int(
        config["model"]["sequence_length"]
    )

    # --------------------------------------------------
    # Scale latest telemetry
    # --------------------------------------------------

    latest_values = request.values[
        -sequence_length:
    ]

    scaled = [
        (value - mean) / scale
        for value in latest_values
    ]

    x = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0)

    # --------------------------------------------------
    # Transformer prediction
    # --------------------------------------------------

    with torch.no_grad():
        scaled_prediction = float(
            model(x).item()
        )

    prediction = (
        scaled_prediction * scale
        + mean
    )

    # --------------------------------------------------
    # Forecast-only mode
    # --------------------------------------------------

    if request.observed_next_value is None:
        return PredictionResponse(
            prediction=prediction,
            anomaly=False,
            threshold=threshold,
            persistence=persistence,
            score_window=score_window,
            consecutive_anomalies=(
                detector.consecutive_anomalies
            ),
        )

    # --------------------------------------------------
    # Residual anomaly score
    # --------------------------------------------------

    residual = abs(
        request.observed_next_value
        - prediction
    )

    anomaly_score = (
        residual / max(scale, 1e-8)
    )

    # --------------------------------------------------
    # Stateful persistence detection
    # --------------------------------------------------

    anomaly = detector.update(
        anomaly_score
    )

    # --------------------------------------------------
    # ServiceNow automation
    # --------------------------------------------------

    incident_created = False
    incident_number = None
    incident_error = None

    if anomaly and request.create_incident:

        try:
            client = ServiceNowClient()

            severity = (
                "1"
                if anomaly_score >= 10
                else "2"
                if anomaly_score >= 5
                else "3"
            )

            short_description = (
                "NeuroPredict: EC2 CPU anomaly detected"
            )

            description = (
                "NeuroPredict predictive AIOps alert.\n\n"
                f"Predicted CPU value: "
                f"{prediction:.4f}\n"
                f"Observed CPU value: "
                f"{request.observed_next_value:.4f}\n"
                f"Residual: "
                f"{residual:.4f}\n"
                f"Anomaly score: "
                f"{anomaly_score:.4f}\n"
                f"Detection threshold: "
                f"{threshold:.4f}\n"
                f"Persistence requirement: "
                f"{persistence}\n"
                f"Consecutive anomalies: "
                f"{detector.consecutive_anomalies}\n"
                f"Score window: "
                f"{score_window}\n"
                f"Model: TransformerForecaster\n"
                f"Sequence length: "
                f"{sequence_length}\n"
                f"Dataset: "
                f"{config['dataset']['file']}\n"
            )

            result = client.create_incident(
                short_description=short_description,
                description=description,
                severity=severity,
            )

            incident = result.get(
                "result",
                {}
            )

            incident_number = incident.get(
                "number"
            )

            incident_created = True

        except Exception as exc:
            incident_error = str(exc)

    return PredictionResponse(
        prediction=prediction,
        residual=residual,
        anomaly_score=anomaly_score,
        anomaly=anomaly,
        threshold=threshold,
        persistence=persistence,
        score_window=score_window,
        consecutive_anomalies=(
            detector.consecutive_anomalies
        ),
        incident_created=incident_created,
        incident_number=incident_number,
        incident_error=incident_error,
    )