from __future__ import annotations

import math

from ..types import DetectorOut, Features


def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    if kappa <= 0.0 or alpha <= 0.0 or beta <= 0.0 or not math.isfinite(x):
        return -1e12
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    inv = 1.0 + ((x - mu) ** 2) / (nu * scale2)
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * (math.log(math.pi * nu) + math.log(scale2))
        - ((nu + 1.0) / 2.0) * math.log(inv)
    )

def _update_nig(mu: float, kappa: float, alpha: float, beta: float, x: float) -> tuple[float, float, float, float]:
    kappa_n = kappa + 1.0
    mu_n = (kappa * mu + x) / kappa_n
    alpha_n = alpha + 0.5
    beta_n = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa_n
    return mu_n, kappa_n, alpha_n, beta_n

def _logsumexp(values: list[float]) -> float:
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))

class BOCPD:
    """
    Bayesian Online Change Point Detection (Student-t emissions).
    CHANGE uses prior predictive; GROWTH uses run-length predictive.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        cooldown: int = 5,
        hazard: float = 1 / 200,
        rmax: int = 400,
        mu0: float = 0.0,
        kappa0: float = 1e-3,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        vol_threshold: float = 0.02,
    ) -> None:
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.h = float(hazard)
        self.r_cap = int(rmax)
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.params: list[tuple[float, float, float, float]] = [self.prior]
        self.pr: list[float] = [1.0]
        self._cool = 0
        self.vol_threshold = float(vol_threshold)

    def update(self, x: float, feats: Features) -> DetectorOut:
        x = float(x)
        error = False

        r = min(self.r_cap, len(self.pr) + 1)
        if len(self.params) < r:
            self.params += [self.prior] * (r - len(self.params))

        try:
            loglik = [_student_t_logpdf(x, *self.params[i]) for i in range(len(self.pr))]
            log_pr = [math.log(p) if p > 0 else -1e12 for p in self.pr]
            log1mh = math.log(max(1.0 - self.h, 1e-12))
            logh = math.log(max(self.h, 1e-12))

            new_logpr = [-1e12] * r
            loglik_prior = _student_t_logpdf(x, *self.prior)
            lsum_pr = _logsumexp(log_pr)
            new_logpr[0] = logh + loglik_prior + lsum_pr

            for i in range(len(self.pr)):
                if i + 1 < r:
                    term = log_pr[i] + log1mh + loglik[i]
                    a, b = new_logpr[i + 1], term
                    if a > b:
                        new_logpr[i + 1] = a + math.log1p(math.exp(b - a))
                    else:
                        new_logpr[i + 1] = b + math.log1p(math.exp(a - b))

            lz = _logsumexp(new_logpr)
            if not math.isfinite(lz):
                raise FloatingPointError("normalization failed")

            self.pr = [math.exp(v - lz) for v in new_logpr]
            cp_prob = self.pr[0]

            new_params: list[tuple[float, float, float, float]] = [None] * r  # type: ignore[list-item]
            new_params[0] = _update_nig(*self.prior, x)
            for i in range(1, r):
                prev = self.params[i - 1] if (i - 1) < len(self.params) else self.prior
                new_params[i] = _update_nig(*prev, x)
            self.params = new_params

        except Exception:
            self.pr = [1.0]
            self.params = [self.prior]
            cp_prob = self.h
            error = True

        if cp_prob >= self.threshold:
            self._cool = self.cooldown
        recent_cp = self._cool > 0
        if self._cool > 0:
            self._cool -= 1

        vol = float(feats.get("ewm_vol", 0.0))
        if vol == 0.0:
            vol = float(feats.get("rv", 0.0))
        label = "high_vol" if vol > self.vol_threshold else "low_vol"

        return {
            "regime_label": label,
            "regime_score": float(cp_prob),
            "meta": {"cp_prob": float(cp_prob), "recent_cp": recent_cp, "error": error},
        }
