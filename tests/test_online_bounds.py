import numpy as np
from src.online_bounds import OnlineBounds

def test_sup_monotone():
    ob = OnlineBounds()
    xs = [3, 1, 4, 2, 10, 6]
    sups = []
    for v in xs:
        ob.update(v)
        sups.append(ob.sup)
    assert all(sups[i] <= sups[i+1] for i in range(len(sups)-1))
    assert sups[-1] == 10

def test_welford_mean_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    ob = OnlineBounds()
    for v in x:
        ob.update(float(v))
    assert abs(ob.mean - float(np.mean(x))) < 1e-3
    assert abs(ob.std() - float(np.std(x, ddof=1))) < 1e-3

