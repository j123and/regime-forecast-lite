# tests/test_eviction_and_nan.py
import importlib
import os


def _reload_app(env: dict):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    import service.app as appmod
    importlib.reload(appmod)
    return appmod

def test_pending_eviction_causes_404():
    appmod = _reload_app({"PENDING_CAP": 1})
    from fastapi.testclient import TestClient

    client = TestClient(appmod.app)
    p1 = client.post("/predict", json={"timestamp": "t0", "x": 0.1})
    assert p1.status_code == 200
    pid1 = p1.json()["prediction_id"]

    p2 = client.post("/predict", json={"timestamp": "t1", "x": 0.2})
    assert p2.status_code == 200

    # pid1 should be evicted from pending map
    t = client.post("/truth", json={"prediction_id": pid1, "y": 0.1})
    assert t.status_code == 404

def test_nan_input_422():
    appmod = _reload_app({})
    from fastapi.testclient import TestClient

    client = TestClient(appmod.app)
    # send NaN as string; service coerces and rejects
    r = client.post("/predict", json={"timestamp": "t", "x": "NaN"})
    assert r.status_code == 422
