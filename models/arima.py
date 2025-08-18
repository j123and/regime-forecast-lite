from typing import Any, Tuple, Dict

class ARIMAModel:
    def __init__(self) -> None:
        pass

    def predict_update(self, tick: Dict[str, Any], feats: Dict[str, float]) -> Tuple[float, Dict]:
        y_hat = float(tick["x"])  # naive baseline
        return y_hat, {}
