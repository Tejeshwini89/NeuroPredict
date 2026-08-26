from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import TransformerForecaster
from .servicenow import ServiceNowClient


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "transformer_forecaster.pt"
SCALER_PATH = ROOT / "artifacts" / "scaler.pt"
EVAL_PATH = ROOT / "artifacts" / "evaluation.json"


app = FastAPI(
    title="NeuroPredict API",
    version="0.2.0",
    description="Transformer-based predictive AIOps inference and ServiceNow incident automation API.",
)


_model = None
_scaler = None
_threshold = None


class PredictionRequest(BaseModel):
    values: list[float] = Field(
        min_length=48,
        description="Latest 48 raw telemetry values.",
    )

    observed_next_value: float | None = Field(
        default=None,
        description="Observed next telemetry value.",
    )

    create_incident: bool = Field(
        default=False,
        description="Create a ServiceNow incident when an anomaly is detected.",
    )


class PredictionResponse(BaseModel):
    prediction: float
    residual: float | None = None
    anomaly_score: float | None = None
    anomaly: bool

    incident_created: bool = False
    incident_number: str | None = None
    incident_error: str | None = None


def get_components():
    global _model, _scaler, _threshold

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts missing. Run training first.",
        )

    if _model is None:
        _model = TransformerForecaster()

        _model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu",
            )
        )

        _model.eval()

    if _scaler is None:
        _scaler = torch.load(
            SCALER_PATH,
            map_location="cpu",
            weights_only=False,
        )

    if _threshold is None and EVAL_PATH.exists():
        _threshold = float(
            json.loads(
                EVAL_PATH.read_text()
            )["anomaly_threshold"]
        )

    return _model, _scaler, _threshold


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": (
            MODEL_PATH.exists()
            and SCALER_PATH.exists()
        ),
        "servicenow_configured": all(
            [
                "SERVICENOW_INSTANCE" in __import__("os").environ,
                "SERVICENOW_USER" in __import__("os").environ,
                "SERVICENOW_PASSWORD" in __import__("os").environ,
            ]
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    model, scaler, threshold = get_components()

    mean = float(scaler["mean"])
    scale = float(scaler["scale"])

    # --------------------------------------------------
    # Scale latest telemetry
    # --------------------------------------------------

    latest_values = request.values[-48:]

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
        )

    # --------------------------------------------------
    # Residual anomaly detection
    # --------------------------------------------------

    residual = abs(
        request.observed_next_value
        - prediction
    )

    anomaly_score = (
        residual / max(scale, 1e-8)
    )

    anomaly = (
        threshold is not None
        and anomaly_score >= threshold
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
                f"Predicted CPU value: {prediction:.4f}\n"
                f"Observed CPU value: {request.observed_next_value:.4f}\n"
                f"Residual: {residual:.4f}\n"
                f"Anomaly score: {anomaly_score:.4f}\n"
                f"Detection threshold: {threshold:.4f}\n"
                f"Model: TransformerForecaster\n"
                f"Sequence length: 48\n"
                f"Dataset: ec2_cpu_utilization_53ea38\n"
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
        incident_created=incident_created,
        incident_number=incident_number,
        incident_error=incident_error,
    )