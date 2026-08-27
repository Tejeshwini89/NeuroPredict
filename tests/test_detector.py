from src.neuropredict.detector import PersistenceDetector


def test_single_anomaly_with_persistence_one():
    detector = PersistenceDetector(
        threshold=1.0,
        persistence=1,
    )

    assert detector.update(1.5) is True


def test_persistence_requires_consecutive_anomalies():
    detector = PersistenceDetector(
        threshold=1.0,
        persistence=2,
    )

    assert detector.update(1.5) is False
    assert detector.update(1.8) is True


def test_normal_score_resets_persistence():
    detector = PersistenceDetector(
        threshold=1.0,
        persistence=2,
    )

    assert detector.update(1.5) is False
    assert detector.update(0.5) is False
    assert detector.update(1.5) is False
    assert detector.update(1.6) is True


def test_reset_clears_state():
    detector = PersistenceDetector(
        threshold=1.0,
        persistence=2,
    )

    detector.update(1.5)

    detector.reset()

    assert detector.consecutive_anomalies == 0
    assert detector.update(1.5) is False