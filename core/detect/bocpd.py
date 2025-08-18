# core/detect/bocpd.py
import math
from typing import List, Tuple
from ..types import Features, DetectorOut

def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """
    Predictive Student-t for Normal-Inverse-Gamma prior (unknown mean/variance)
    nu = 2*alpha; scale^2 = beta*(kappa+1)/(alpha*kappa)
    """
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

def _update_nig(mu: float, kappa: float, alpha: float, beta: float, x: float) -> Tuple[float, float, float, float]:
    """One-step posterior update for Normal-Inverse-Gamma."""
    kappa_n = kappa + 1.0
    mu_n = (kappa * mu + x) / kappa_n
    alpha_n = alpha + 0.5
    beta_n = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa_n
    return mu_n, kappa_n, alpha_n, beta_n

def _logsumexp(values: List[float]) -> float:
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))

class BOCPD:
    """
    Bayesian Online Change Point Detection (Adams & MacKay) with Student-t emissions.

    Key detail:
      - GROWTH (r -> r+1) uses the run-length predictive p(x_t | params[r])
      - CHANGE (r -> 0) uses the PRIOR predictive p(x_t | prior), NOT the run-length predictive

    If you (incorrectly) use the run-length predictive for CHANGE as well,
    P(r_t = 0 | x_{1:t}) collapses toward the hazard h, which is the bug you saw.
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

        # state
        self.params: List[Tuple[float, float, float, float]] = [self.prior]  # params[r]
        self.pr: List[float] = [1.0]  # run-length pmf: p(r_t = r)
        self._cool = 0
        self.vol_threshold = float(vol_threshold)

    def update(self, x: float, feats: Features) -> DetectorOut:
        x = float(x)

        # ensure arrays up to R
        R = min(self.R, len(self.pr) + 1)
        if len(self.params) < R:
            self.params += [self.prior] * (R - len(self.params))

        # predictive log-likelihood for each existing run length (for GROWTH)
        loglik = [_student_t_logpdf(x, *self.params[r]) for r in range(len(self.pr))]

        # log domain for stability
        log_pr = [math.log(p) if p > 0 else -1e9 for p in self.pr]
        log1mh = math.log(max(1.0 - self.h, 1e-12))
        logh = math.log(max(self.h, 1e-12))

        # --- new run-length log-probs (unnormalized) ---
        new_logpr = [-1e9] * R

        # CHANGE: r -> 0  (use PRIOR predictive)
        loglik_prior = _student_t_logpdf(x, *self.prior)
        lsum_pr = _logsumexp(log_pr)  # log(sum_r p(r))
        new_logpr[0] = logh + loglik_prior + lsum_pr

        # GROWTH: r -> r+1  (use run-length predictive)
        for r in range(len(self.pr)):
            if r + 1 < R:
                term = log_pr[r] + log1mh + loglik[r]
                a, b = new_logpr[r + 1], term
                if a > b:
                    new_logpr[r + 1] = a + math.log1p(math.exp(b - a))
                else:
                    new_logpr[r + 1] = b + math.log1p(math.exp(a - b))

        # normalize
        lZ = _logsumexp(new_logpr)
        if not math.isfinite(lZ):
            # numeric guard: reset to prior
            self.pr = [1.0]
            self.params = [self.prior]
            cp_prob = 1.0
        else:
            self.pr = [math.exp(v - lZ) for v in new_logpr]
            cp_prob = self.pr[0]

        # update posterior params for next step
        new_params: List[Tuple[float, float, float, float]] = [None] * R  # type: ignore
        new_params[0] = _update_nig(*self.prior, x)  # start new segment with x
        for r in range(1, R):
            prev = self.params[r - 1] if (r - 1) < len(self.params) else self.prior
            new_params[r] = _update_nig(*prev, x)
        self.params = new_params

        # hysteresis via cooldown when cp_prob spikes
        if cp_prob >= self.threshold:
            self._cool = self.cooldown
        if self._cool > 0:
            self._cool -= 1

        # regime label from vol proxy; score is cp_prob
        vol = float(feats.get("ewm_vol", 0.0) or feats.get("rv", 0.0))
        label = "high_vol" if vol > self.vol_threshold else "low_vol"

        return {
            "regime_label": label,
            "regime_score": float(cp_prob),
            "meta": {"cp_prob": float(cp_prob)},
        }
