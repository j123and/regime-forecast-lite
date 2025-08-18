from core.features import FeatureExtractor

def test_features_stream_constant():
    fe = FeatureExtractor(win=3, rv_win=3, ewm_alpha=0.5)
    for i in range(5):
        out = fe.update({"timestamp": str(i), "x": 0.0, "covariates": {}})
        for k in ("z", "ewm_vol", "ac1", "rv"):
            assert k in out
            assert isinstance(out[k], float)
        assert abs(out["z"]) < 1e-9
        assert out["ewm_vol"] >= 0.0
        assert out["rv"] >= 0.0
