# stream-bounds

Streaming Real-Analysis Stats in O(1): inf/sup, Welford mean/var, P2 quantiles, and Linf envelopes for sensor telemetry.
Five-marker P2 is implemented with canonical 1-based positions and correct marker updates (n[k:] += 1).
The online tracker reports latency and keeps constant-sized state.

----------------------------------------------------------------

## Why this matters

You often cannot store the full stream (telemetry, embedded, ops). You still need:
- Bounds: inf, sup (exist by completeness; running sup_T is monotone increasing).
- Robust spread: Linf envelope radius max_t |x_t - c_t| around a center c_t (mean vs median).
- Quantiles: percentiles for thresholds/alerts without O(n) memory.
- Low latency: O(1) per-sample updates.

This repo shows batch vs streaming trade-offs with rigorous, reproducible code.

----------------------------------------------------------------

## Algorithms (one-liners)

- Welford mean/var: numerically stable, O(1) state, single pass.
- P2 (Jain & Chlamtac, 1985): O(1) space/time streaming estimator for one quantile using 5 markers.
  Implementation details that matter: (i) 1-based marker positions, (ii) increment markers >= k, (iii) parabolic prediction with monotonicity guard.
- Linf envelopes: max_t |x_t - center_t|. We track two: around running mean and around running median (P2).
  The median envelope becomes defined once the P2 median is ready.

----------------------------------------------------------------

## Repo layout

```
stream-bounds/
|-- src/
|   |-- p2.py                 # P2 estimator (fixed; 1-based positions; n[k:] += 1)
|   |-- online_bounds.py      # Online O(1) tracker (dual Linf envelopes)
|   `-- main.py               # CLI: batch vs streaming comparison
|-- tests/
|   |-- test_p2.py            # P2 unit test
|   `-- test_online_bounds.py # sup monotonicity; Welford vs numpy
|-- notebooks/
|   `-- demo.ipynb            # Plots: P2 error vs n/outliers; latency histogram
|-- benchmarks/
|   `-- microbench.py         # (optional) batch timing harness
|-- .github/workflows/ci.yml  # pytest on push/PR
|-- pyproject.toml            # deps + ruff/black/pytest config
|-- requirements.txt          # alternative to pyproject deps
|-- README.md                 # you are here
|-- LICENSE
`-- .gitignore
```

----------------------------------------------------------------

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python >= 3.9, NumPy >= 1.24.

----------------------------------------------------------------

## Quickstart

```
python -m src.main --n 200000 --seed 7 --outlier-rate 0.005 --q 0.1 --q 0.5 --q 0.9
```

Outputs
- Batch (np.quantile) reference vs P2 estimates
- Linf envelopes online (mean and median)
- Latency: avg and p95 per-sample
- Memory notes

----------------------------------------------------------------

## CLI flags

```
--n               Number of samples (default 200000)
--seed            RNG seed (default 7)
--outlier-rate    Fraction of rare spikes (default 0.005)
--q               Quantile to track (repeatable; default 0.1, 0.5, 0.9)
```

----------------------------------------------------------------

## Benchmarks (example expectations)

- Welford mean/std matches NumPy within ~1e-3 on 50k+ samples.
- P2(0.5) typically within < 0.04*sigma on Normal(mu, sigma^2) at n = 2e5.
- Latency on a modern laptop: tens of microseconds per sample (measure yours).
- Linf(median) is less sensitive to spikes than Linf(mean).

See `notebooks/demo.ipynb` to reproduce and plot error vs n and vs outlier rate.

----------------------------------------------------------------

## Pitfalls and gotchas

- P2 indexing: Use 1-based positions [1,2,3,4,5] and update markers >= k.
- Monotonicity: Fall back to linear prediction when parabolic breaks monotonicity; clamp if needed.
- Definition drift: Do not switch envelope centers midstream; track both from t = 1 (median undefined until ready).
- "Exact" quantiles vocabulary: np.quantile(method="linear") is the standard reference, not an order-statistic exact pick.

----------------------------------------------------------------

## Tests and CI

```
pytest
```
CI runs pytest on push/PR. Add more distributional tests if you extend P2 to multiple quantiles.

----------------------------------------------------------------

## References

- R. Jain and I. Chlamtac, "The P2 Algorithm for Dynamic Calculation of Quantiles and Histograms Without Storing Observations," Communications of the ACM, 28(10):1076-1085, 1985.
- T. Chan et al., "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances," The American Statistician, 1998 (Welford derivations).

