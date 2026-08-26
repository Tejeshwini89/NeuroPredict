# NeuroPredict — AI-Powered Predictive AIOps Platform

NeuroPredict is an end-to-end predictive AIOps system for infrastructure telemetry.

Pipeline:
1. Download real AWS CloudWatch CPU telemetry from the Numenta Anomaly Benchmark (NAB).
2. Validate, clean, scale, and window the time series.
3. Train a Transformer-based next-step forecaster.
4. Compare it with a naive persistence baseline.
5. Convert forecast residuals into anomaly scores.
6. Evaluate forecasting and anomaly-detection performance.
7. Serve inference through FastAPI.
8. Provide a ServiceNow Table API adapter for incident creation.
9. Include Docker and GitHub Actions CI scaffolding.

## Architecture

```text
AWS CloudWatch telemetry
        |
        v
 Data validation
        |
        v
 Normal-window filtering
        |
        v
 Sliding sequences
        |
        v
 Transformer forecaster
        |
        v
 Forecast residual / anomaly score
        |
        +------> FastAPI inference
        |
        +------> ServiceNow incident adapter
```

## Dataset

This project uses:
`realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`

from the Numenta Anomaly Benchmark. NAB documents its realAWSCloudwatch
directory as real AWS server metrics collected through Amazon CloudWatch.

The data is downloaded at runtime and is not committed to Git.

## Run locally

Recommended Python: 3.11 or 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python -m src.neuropredict.train
python -m src.neuropredict.evaluate
uvicorn src.neuropredict.api:app --reload
```

API docs:
`http://127.0.0.1:8000/docs`

## ServiceNow

Set these environment variables before using the adapter:

```text
SERVICENOW_INSTANCE
SERVICENOW_USER
SERVICENOW_PASSWORD
SERVICENOW_TABLE
```

Credentials must never be committed.

## Engineering decisions

- Transformer forecasting models temporal dependencies.
- Naive persistence is retained as a baseline.
- Known anomaly windows are excluded from the normal training region.
- Forecast residuals provide an interpretable anomaly signal.
- FastAPI separates model serving from training.
- Docker and CI improve reproducibility.

## Roadmap

- Multivariate CPU/network/disk telemetry
- Probabilistic forecasting intervals
- MLflow/model registry
- AWS SageMaker training and deployment
- Prometheus-based monitoring and drift detection
- Asynchronous ServiceNow incident creation
- Production dashboard

## Data attribution

Numenta Anomaly Benchmark:
https://github.com/numenta/NAB

See the upstream repository for dataset and license details.
