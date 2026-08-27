from __future__ import annotations


class PersistenceDetector:
    """
    Stateful anomaly detector.

    An observation becomes a confirmed anomaly only when
    the anomaly condition remains true for the configured
    number of consecutive observations.
    """

    def __init__(
        self,
        threshold: float,
        persistence: int = 1,
    ):
        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than 0."
            )

        if persistence < 1:
            raise ValueError(
                "persistence must be at least 1."
            )

        self.threshold = float(threshold)
        self.persistence = int(persistence)

        self._consecutive_anomalies = 0

    @property
    def consecutive_anomalies(self) -> int:
        """Return the current consecutive anomaly count."""
        return self._consecutive_anomalies

    def reset(self) -> None:
        """Reset the consecutive anomaly counter."""
        self._consecutive_anomalies = 0

    def update(self, anomaly_score: float) -> bool:
        """
        Process one anomaly score.

        Returns True only when the configured persistence
        requirement has been satisfied.
        """

        if anomaly_score < 0:
            raise ValueError(
                "anomaly_score cannot be negative."
            )

        if anomaly_score >= self.threshold:
            self._consecutive_anomalies += 1
        else:
            self._consecutive_anomalies = 0

        if (
            self._consecutive_anomalies
            >= self.persistence
        ):
            return True

        return False