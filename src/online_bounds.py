# online_bounds.py
import time, math
from dataclasses import dataclass
import numpy as np
from src.p2 import P2Quantile

@dataclass
class OnlineBounds:
    track_quantiles: tuple = (0.1, 0.5, 0.9)

    def __post_init__(self):
        self.n = 0
        self.inf = float('inf')
        self.sup = float('-inf')
        self.mean = 0.0
        self.m2 = 0.0

        # L∞ envelopes (max |x - center_t|) for two centers
        self.max_abs_dev_mean = 0.0
        self.max_abs_dev_median = float('nan')  # undefined until median is ready
        self._median_ready_once = False

        self.qmap = {q: P2Quantile(q) for q in self.track_quantiles}

    def update(self, x: float) -> float:
        t0 = time.perf_counter()

        # inf/sup
        if x < self.inf: self.inf = x
        if x > self.sup: self.sup = x

        # Welford mean/var
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

        # Update quantiles (P²)
        for est in self.qmap.values():
            est.update(x)

        # Envelopes
        # mean-centered envelope always defined
        self.max_abs_dev_mean = max(self.max_abs_dev_mean, abs(x - self.mean))

        # median-centered envelope defined once median estimator is ready
        med_ready = (0.5 in self.qmap) and self.qmap[0.5].ready()
        if med_ready:
            med = self.qmap[0.5].value()
            if not self._median_ready_once:
                self.max_abs_dev_median = 0.0
                self._median_ready_once = True
            self.max_abs_dev_median = max(self.max_abs_dev_median, abs(x - med))

        t1 = time.perf_counter()
        return (t1 - t0)

    def std(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else float('nan')

    def q(self, qval: float):
        return self.qmap[qval].value() if qval in self.qmap else float('nan')

