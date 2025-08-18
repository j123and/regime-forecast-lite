from .types import Tick, Features

class FeatureExtractor:
    """MVP feature extractor. Expand with rolling/EWMA/autocorr later."""
    def __init__(self) -> None:
        pass

    def update(self, tick: Tick) -> Features:
        x = float(tick["x"])
        feats: Features = {"z": x}
        feats.update(tick.get("covariates", {}))  # safe merge
        return feats
