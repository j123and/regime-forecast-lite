from __future__ import annotations

from fastapi.testclient import TestClient

from service.app import app


def test_healthz():
    c = TestClient(app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    c = TestClient(app)
    payload = {
        "timestamp": "2024-01-01T00:00:00Z",
        "x": 1.23,
        "covariates": {"rv": 0.01},
    }
    r = c.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "y_hat" in body
    assert "latency_ms" in body
