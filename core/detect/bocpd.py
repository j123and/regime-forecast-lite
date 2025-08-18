from .types import Features, DetectorOut

class BOCPD:
    """Stub with same interface as the real BOCPD; replace logic later."""
    def __init__(self, threshold: float = 0.7, cooldown: int = 5) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._cool = 0

    def update(self, x: float, feats: Features) -> DetectorOut:
        score = min(1.0, abs(float(feats.get("z", 0.0))) / 2.0)
        label = "high_vol" if score > 0.5 else "low_vol"
        if score >= self.threshold:
            self._cool = self.cooldown
        if self._cool > 0 and label == "high_vol":
            self._cool -= 1
        return {"regime_label": label, "regime_score": float(score), "meta": {"cp_prob": float(score)}}
