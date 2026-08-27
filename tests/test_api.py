from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.neuropredict.api import app


client = TestClient(app)


NORMAL_VALUES = [
    10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57,
]


def reset_detector():
    """Reset the API detector state between tests."""
    import src.neuropredict.api as api_module

    api_module._detector = None


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert data["evaluation_ready"] is True


def test_predict_normal_value_does_not_create_incident():
    reset_detector()

    response = client.post(
        "/predict",
        json={
            "values": NORMAL_VALUES,
            "observed_next_value": 2.0,
            "create_incident": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["anomaly"] is False
    assert data["incident_created"] is False
    assert data["incident_number"] is None
    assert data["incident_error"] is None
    assert data["consecutive_anomalies"] == 0


def test_predict_persistent_anomaly_is_detected():
    reset_detector()

    payload = {
        "values": NORMAL_VALUES,
        "observed_next_value": 10.0,
        "create_incident": False,
    }

    first_response = client.post(
        "/predict",
        json=payload,
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert first_data["anomaly"] is False
    assert first_data["consecutive_anomalies"] == 1

    second_response = client.post(
        "/predict",
        json=payload,
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert second_data["anomaly"] is True
    assert second_data["consecutive_anomalies"] == 2


def test_predict_anomaly_creates_mocked_servicenow_incident():
    reset_detector()

    mock_client = MagicMock()

    mock_client.create_incident.return_value = {
        "result": {
            "number": "INC-MOCK-0001",
        }
    }

    payload = {
        "values": NORMAL_VALUES,
        "observed_next_value": 10.0,
        "create_incident": True,
    }

    with patch(
        "src.neuropredict.api.ServiceNowClient",
        return_value=mock_client,
    ):
        first_response = client.post(
            "/predict",
            json=payload,
        )

        assert first_response.status_code == 200
        assert first_response.json()["anomaly"] is False
        assert first_response.json()["incident_created"] is False

        second_response = client.post(
            "/predict",
            json=payload,
        )

    assert second_response.status_code == 200

    data = second_response.json()

    assert data["anomaly"] is True
    assert data["incident_created"] is True
    assert data["incident_number"] == "INC-MOCK-0001"
    assert data["incident_error"] is None

    mock_client.create_incident.assert_called_once()