# tests/test_rate_limit.py
import importlib
import os

from fastapi.testclient import TestClient


def _reload_app(env: dict):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    import service.app as appmod
    importlib.reload(appmod)
    return appmod

def test_rate_limit_simple():
    appmod = _reload_app({"RATE_LIMIT_PER_MINUTE": 2})
    client = TestClient(appmod.app)
    # 1st ok
    assert client.post("/predict", json={"timestamp": "t0", "x": 0.0}).status_code == 200
    # 2nd ok
    assert client.post("/predict", json={"timestamp": "t1", "x": 0.0}).status_code == 200
    # 3rd should hit 429
    assert client.post("/predict", json={"timestamp": "t2", "x": 0.0}).status_code == 429

def test_api_key_guard():
    appmod = _reload_app({"API_KEY": "secret"})
    client = TestClient(appmod.app)
    # no key -> 401
    r1 = client.post("/predict", json={"timestamp": "t", "x": 0.0})
    assert r1.status_code == 401
    # with key -> ok
    r2 = client.post("/predict", headers={"x-api-key": "secret"}, json={"timestamp": "t", "x": 0.0})
    assert r2.status_code == 200
