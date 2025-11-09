# main.py
import time, sys, math, argparse
from dataclasses import dataclass
import numpy as np
from src.online_bounds import OnlineBounds

def offline_stats(x: np.ndarray, quantiles=(0.1, 0.5, 0.9)):
    t0 = time.perf_counter()
    x = np.asarray(x)
    inf_ = float(np.min(x))
    sup_ = float(np.max(x))
    mean_ = float(np.mean(x))
    var_ = float(np.var(x, ddof=1)) if len(x) > 1 else float('nan')
    qs = {q: float(np.quantile(x, q, method="linear")) for q in quantiles}
    env_mean = float(np.max(np.abs(x - mean_)))
    med_ = qs[0.5] if 0.5 in qs else float(np.quantile(x, 0.5))
    env_median = float(np.max(np.abs(x - med_)))
    t1 = time.perf_counter()
    return {
        "time_s": t1 - t0,
        "inf": inf_, "sup": sup_, "mean": mean_, "std": math.sqrt(var_) if var_==var_ else float('nan'),
        "quantiles": qs,
        "envelope_mean": env_mean,
        "envelope_median": env_median,
        "space_note": "Stores all n samples (O(n) memory)"
    }

def online_stats(x: np.ndarray, quantiles=(0.1, 0.5, 0.9)):
    ob = OnlineBounds(track_quantiles=quantiles)
    lat = []
    for v in x:
        lat.append(ob.update(float(v)))
    qs = {q: ob.q(q) for q in quantiles}
    return {
        "avg_latency_ms": 1e3 * float(np.mean(lat)),
        "p95_latency_ms": 1e3 * float(np.percentile(lat, 95)),
        "inf": ob.inf, "sup": ob.sup, "mean": ob.mean, "std": ob.std(),
        "quantiles": qs,
        "envelope_mean": ob.max_abs_dev_mean,
        "envelope_median": ob.max_abs_dev_median,  # may be NaN if median never got ready
        "space_note": "Keeps constant-sized state (O(1) memory)"
    }

def make_signal(n=200000, seed=7, outlier_rate=0.005):
    rng = np.random.default_rng(seed)
    base = 50 + 2*rng.standard_normal(n) + 0.5*np.sin(np.linspace(0, 12*np.pi, n))
    x = base.copy()
    k = int(outlier_rate * n)
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        x[idx] += rng.choice([+50, -30], size=k)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000, help="number of samples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outlier-rate", type=float, default=0.005)
    ap.add_argument("--q", type=float, action="append", default=[0.1, 0.5, 0.9],
                    help="quantile(s) to track")
    args = ap.parse_args()

    x = make_signal(n=args.n, seed=args.seed, outlier_rate=args.outlier_rate)
    print(f"\nGenerated stream: n={len(x):,} (seed={args.seed}, outliers≈{args.outlier_rate*100:.2f}%)")

    off = offline_stats(x, quantiles=tuple(args.q))
    on  = online_stats(x, quantiles=tuple(args.q))

    def fmtq(qs): return ", ".join([f"q{int(100*q):02d}={qs[q]:.4f}" for q in sorted(qs)])

    print("\nOFFLINE (batch, reference) — O(n) time, O(n) memory (stores data):")
    print(f"  time: {off['time_s']:.4f}s")
    print(f"  inf/sup: {off['inf']:.4f} / {off['sup']:.4f}")
    print(f"  mean±std: {off['mean']:.4f} ± {off['std']:.4f}")
    print(f"  quantiles (np.quantile): {fmtq(off['quantiles'])}")
    print(f"  L∞ envelope (mean):   {off['envelope_mean']:.4f}")
    print(f"  L∞ envelope (median): {off['envelope_median']:.4f}")

    print("\nONLINE (streaming, constant space) — O(1) per sample:")
    print(f"  avg latency per sample: {on['avg_latency_ms']:.6f} ms (p95 {on['p95_latency_ms']:.6f} ms)")
    print(f"  inf/sup: {on['inf']:.4f} / {on['sup']:.4f}")
    print(f"  mean±std: {on['mean']:.4f} ± {on['std']:.4f}")
    print(f"  quantiles (P²): {fmtq(on['quantiles'])}")
    print(f"  L∞ envelope_online (mean):   {on['envelope_mean']:.4f}")
    print(f"  L∞ envelope_online (median): {on['envelope_median']:.4f}")  # may print 'nan'
    print(f"  space: {on['space_note']}")

    deltas = {q: abs(on['quantiles'][q] - off['quantiles'][q]) for q in off['quantiles']}
    print("\n|P² − batch quantile| (absolute error):")
    for q in sorted(deltas):
        print(f"  q{int(100*q):02d}: {deltas[q]:.6f}")

    print("\nNotes:")
    print("  • P² uses five markers; median envelope becomes defined once the median marker is initialized.")
    print("  • Batch quantiles via np.quantile(method='linear') are a reference, not order-statistic exact.")
    print("  • Online L∞ envelopes use running centers (mean, P² median).")
    print("  • For reproducibility: tweak --seed and --outlier-rate.")

if __name__ == "__main__":
    main()

