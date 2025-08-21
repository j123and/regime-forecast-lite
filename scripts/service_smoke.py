from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from service.app import app


def main() -> None:
    client = TestClient(app)

    t0 = datetime.now(UTC).replace(microsecond=0).isoformat()
    # Simple synthetic tick
    predict_payload: dict[str, Any] = {
        "timestamp": t0,
        "x": 0.01,
        "covariates": {"rv": 0.01, "ewm_vol": 0.01, "ac1": 0.0, "z": 0.0},
        "series_id": "demo",
        "target_timestamp": (
            datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=1)
        ).isoformat(),
    }

    p = client.post("/predict", json=predict_payload)
    p.raise_for_status()
    p_json = p.json()
    pred_id = p_json["prediction_id"]

    # Feed truth for the predicted tick
    truth_payload: dict[str, Any] = {"prediction_id": pred_id, "y": 0.0125}
    t = client.post("/truth", json=truth_payload)
    t.raise_for_status()
    t_json = t.json()

    out = {"predict": p_json, "truth": t_json}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
