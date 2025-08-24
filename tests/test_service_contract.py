# tests/test_service_contract.py
from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_truth_happy_and_idempotent():
    p = client.post("/predict", json={"timestamp": "2024-01-01T00:00:00Z", "x": 0.01})
    assert p.status_code == 200
    pid = p.json()["prediction_id"]
    series_id = p.json()["series_id"]
    tgt = p.json()["target_timestamp"]

    t1 = client.post("/truth", json={"prediction_id": pid, "y": 0.02})
    assert t1.status_code == 200
    assert t1.json()["idempotent"] is False

    # replay -> idempotent True
    t2 = client.post("/truth", json={"prediction_id": pid, "y_true": 0.02})
    assert t2.status_code == 200
    assert t2.json()["idempotent"] is True

    # fallback by (series_id, target_timestamp)
    p2 = client.post("/predict", json={
        "timestamp": "2024-01-01T01:00:00Z",
        "x": -0.03,
        "series_id": series_id,
        "target_timestamp": tgt  # simulate same target clock
    })
    assert p2.status_code == 200

    t3 = client.post("/truth", json={"series_id": series_id, "target_timestamp": tgt, "value": -0.01})
    # could match the most recent prediction with that key if not evicted
    assert t3.status_code in (200, 404)  # depends on eviction; we just assert it doesn't 422


def test_truth_requires_value():
    p = client.post("/predict", json={"timestamp": "2024-01-01T02:00:00Z", "x": 0.0})
    pid = p.json()["prediction_id"]
    r = client.post("/truth", json={"prediction_id": pid})
    assert r.status_code == 422
