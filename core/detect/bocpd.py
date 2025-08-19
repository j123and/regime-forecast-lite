from __future__ import annotations

import math

from ..types import DetectorOut, Features


def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Predictive Student-t for Normal-Inverse-Gamma prior (unknown mean/variance)."""
    if kappa <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        return -1e9
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
    """One-step posterior update for Normal-Inverse-Gamma."""
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
    Bayesian Online Change Point Detection (Adams & MacKay) with Student-t emissions.

    CHANGE uses PRIOR predictive; GROWTH uses run-length predictive.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        cooldown: int = 5,
        hazard: float = 1 / 200,
        rmax: int = 400,
        # NIG prior
        mu0: float = 0.0,
        kappa0: float = 1e-3,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        # regime labelling
        vol_threshold: float = 0.02,
    ) -> None:
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self.h = float(hazard)
        self.R = int(rmax)
        self.prior = (mu0, kappa0, alpha0, beta0)

        self.params: list[tuple[float, float, float, float]] = [self.prior]
        self.pr: list[float] = [1.0]  # p(r_t = r)
        self._cool = 0
        self.vol_threshold = float(vol_threshold)

    def update(self, x: float, feats: Features) -> DetectorOut:
        x = float(x)
        error = False

        # cap run-length array
        r_cap = min(self.R, len(self.pr) + 1)
        if len(self.params) < r_cap:
            self.params += [self.prior] * (r_cap - len(self.params))

        # predictive loglik for each existing run length (growth)
        loglik = [_student_t_logpdf(x, *self.params[r]) for r in range(len(self.pr))]

        # log-domain transitions
        log_pr = [math.log(p) if p > 0 else -1e9 for p in self.pr]
        log1mh = math.log(max(1.0 - self.h, 1e-12))
        logh = math.log(max(self.h, 1e-12))

        # unnormalized new log pmf
        new_logpr = [-1e9] * r_cap

        # CHANGE: r -> 0  (prior predictive)
        loglik_prior = _student_t_logpdf(x, *self.prior)
        lsum_pr = _logsumexp(log_pr)
        new_logpr[0] = logh + loglik_prior + lsum_pr

        # GROWTH: r -> r+1
        for r in range(len(self.pr)):
            if r + 1 < r_cap:
                term = log_pr[r] + log1mh + loglik[r]
                a, b = new_logpr[r + 1], term
                new_logpr[r + 1] = a + math.log1p(math.exp(b - a)) if a > b else b + math.log1p(math.exp(a - b))

        # normalize
        lz = _logsumexp(new_logpr)
        if not math.isfinite(lz):
            # numeric guard: reset to prior; treat as degraded
            self.pr = [1.0]
            self.params = [self.prior]
            cp_prob = self.h
            error = True
        else:
            self.pr = [math.exp(v - lz) for v in new_logpr]
            cp_prob = self.pr[0]

        # posterior params for next step — build without None sentinels
        new_params: list[tuple[float, float, float, float]] = []
        new_params.append(_update_nig(*self.prior, x))
        for r in range(1, r_cap):
            prev = self.params[r - 1] if (r - 1) < len(self.params) else self.prior
            new_params.append(_update_nig(*prev, x))
        self.params = new_params

        # cooldown / hysteresis side-effect
        if cp_prob >= self.threshold:
            self._cool = self.cooldown
        if self._cool > 0:
            self._cool -= 1

        # regime from vol proxy
        vol = float(feats.get("ewm_vol", 0.0) or feats.get("rv", 0.0))
        label = "high_vol" if vol > self.vol_threshold else "low_vol"

        # run-length summaries
        r_map = int(max(range(len(self.pr)), key=lambda i: self.pr[i]))  # MAP run length
        r_mean = float(sum(i * p for i, p in enumerate(self.pr)))

        return {
            "regime_label": label,
            "regime_score": float(cp_prob),
            "meta": {"cp_prob": float(cp_prob), "r_map": float(r_map), "r_mean": r_mean, "error": error},
        }
