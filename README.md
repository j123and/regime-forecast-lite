# regime-forecast (MVP)

Dev quickstart:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[dev]"
    pre-commit install
    uvicorn service.app:app --reload --port 8000

Endpoints:
- GET /healthz
- GET /metrics
- POST /predict
