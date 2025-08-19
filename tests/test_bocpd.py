from statistics import pstdev

from core.detect.bocpd import BOCPD
from core.features import FeatureExtractor


def test_bocpd_detects_mean_shift_with_small_delay():
    fe = FeatureExtractor(win=20, rv_win=20, ewm_alpha=0.2)
    det = BOCPD(
        threshold=0.2,   # cooldown only
        hazard=1/50,     # modest hazard; baseline cp ~ 0.02
        rmax=200,
        mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=0.1,
        vol_threshold=1e9,
    )

    xs = [0.0]*80 + [0.5]*40
    t0 = 80
    cp = []
    for t, x in enumerate(xs):
        feats = fe.update({"timestamp": str(t), "x": x, "covariates": {}})
        cp.append(det.update(x, feats)["meta"]["cp_prob"])

    pre = cp[:t0]
    post = cp[t0:]

    # Adaptive threshold: pre-shift max plus a margin tied to variance
    pre_max = max(pre)
    pre_sig = pstdev(pre) if len(pre) > 1 else 0.0
    margin = max(2.5 * pre_sig, 0.015)   # ~noise band; floor guards tiny-variance cases
    thr = pre_max + margin

    alarm = None
    for i, v in enumerate(post):
        if v >= thr:
            alarm = t0 + i
            break

    assert alarm is not None, "No change detected"
    assert (alarm - t0) <= 10, f"Delay too large: {alarm - t0}"
