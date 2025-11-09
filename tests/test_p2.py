import numpy as np
from src.p2 import P2Quantile

def test_p2_median_converges_on_normal():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=200_000)
    p2 = P2Quantile(0.5)
    for v in x:
        p2.update(float(v))
    assert p2.ready()
    est = p2.value()
    ref = float(np.quantile(x, 0.5, method="linear"))
    # Should be close; tolerance depends on n and distribution
    assert abs(est - ref) < 0.04 * np.std(x)

