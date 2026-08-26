# NeuroPredict - AI-Powered Predictive AIOps Platform

NeuroPredict is an end-to-end predictive AIOps platform that uses Transformer-based time-series forecasting to anticipate infrastructure behavior, detect abnormal deviations, and automatically create ServiceNow incidents.

The system is designed around a practical AIOps workflow:

**Telemetry -> Forecast -> Residual Analysis -> Anomaly Detection -> Incident Automation**

---

## Why NeuroPredict?

Traditional infrastructure monitoring often reacts after a metric crosses a static threshold.

NeuroPredict takes a different approach:

1. Learn the normal temporal behavior of infrastructure telemetry.
2. Forecast the next expected metric value.
3. Compare the prediction with the observed value.
4. Convert the forecast error into an interpretable anomaly score.
5. Automatically create a ServiceNow incident when the deviation is significant.

This makes the system useful for **predictive monitoring and automated IT operations** rather than simple threshold-based alerting.

---

## Architecture

```text
                    Real Infrastructure Telemetry
                              |
                              v
                    Data Validation & Cleaning
                              |
                              v
                    Train / Validation / Test
                              |
                              v
                    Normal-Window Filtering
                              |
                              v
                     Sliding Time Windows
                              |
                              v
                  Transformer Time-Series Model
                              |
                              v
                     Next-Step Forecast
                              |
                              v
                 +------------+-------------+
                 |                          |
                 v                          v
          Forecast Evaluation        Residual Analysis
                                           |
                                           v
                                    Anomaly Score
                                           |
                                           v
                                    Anomaly Decision
                                           |
                                           v
                                      FastAPI API
                                           |
                                           v
                              ServiceNow Table API
                                           |
                                           v
                                  Incident Created