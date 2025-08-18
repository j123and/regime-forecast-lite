import random
from core.conformal import OnlineConformal

def test_online_conformal_coverage_stationary():
    random.seed(0)
    oc = OnlineConformal(window=1000, decay=1.0)
    # generate iid normal(0,1), predict 0
    y = [random.gauss(0.0, 1.0) for _ in range(2000)]
    hits = 0
    alpha = 0.1  # 90% nominal
    # online update; evaluate coverage after warm-up
    warm = 300
    for t, yt in enumerate(y):
        ql, qh = oc.interval(0.0, alpha=alpha)
        if t >= warm and (ql <= yt <= qh):
            hits += 1
        oc.update(0.0, yt)
    n = len(y) - warm
    cover = hits / max(1, n)
    assert abs(cover - (1 - alpha)) <= 0.08  # loose bound for small-sample online behavior
