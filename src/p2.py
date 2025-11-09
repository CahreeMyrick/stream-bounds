# p2.py
import numpy as np

class P2Quantile:
    """
    Jain & Chlamtac (1985) P² one-quantile streaming estimator.
    Keeps 5 markers (heights x[0..4], positions n[0..4]) in O(1) space/time.
    Uses 1-based marker positions internally (n starts at [1,2,3,4,5]).
    """
    def __init__(self, q: float):
        if not (0 < q < 1):
            raise ValueError("q must be in (0,1)")
        self.q = q
        self._boot = []   # first 5 samples
        self.x = None     # marker heights (5,)
        self.n = None     # marker positions (5,) 1-based floats
        self.np = None    # desired positions (5,)
        self.dn = None    # desired increments (5,)

    def ready(self) -> bool:
        return self.x is not None

    def value(self) -> float:
        if not self.ready():
            return float('nan')
        return float(self.x[2])  # middle marker is the q-quantile

    def update(self, v: float) -> None:
        # Bootstrap the first 5 samples
        if self.x is None:
            self._boot.append(v)
            if len(self._boot) < 5:
                return
            self._boot.sort()
            self.x  = np.array(self._boot, dtype=float)
            # 1-based positions
            self.n  = np.array([1, 2, 3, 4, 5], dtype=float)
            self.np = np.array([1, 1 + 2*self.q, 1 + 4*self.q, 3 + 2*self.q, 5], dtype=float)
            self.dn = np.array([0, self.q/2, self.q, (1+self.q)/2, 1], dtype=float)
            self._boot = None
            return

        # Find k so that x[k-1] <= v < x[k] in 1-based terms.
        # With 0-based arrays and clipping, we let k in [1..4]
        k = int(np.clip(np.searchsorted(self.x, v), 1, 4))
        if v < self.x[0]:
            self.x[0] = v
            k = 1
        elif v > self.x[4]:
            self.x[4] = v
            k = 4

        # Increment marker positions for markers >= k (NOT the lower ones)
        self.n[k:] += 1
        # Update desired positions
        self.np += self.dn

        # Adjust interior markers 1..3 toward desired positions
        for i in (1, 2, 3):
            d = self.np[i] - self.n[i]
            # move at most one step if possible
            if (d >= 1 and (self.n[i+1] - self.n[i]) > 1) or (d <= -1 and (self.n[i-1] - self.n[i]) < -1):
                s = 1 if d >= 1 else -1

                # Parabolic prediction
                num = (self.n[i] - self.n[i-1] + s) * (self.x[i+1] - self.x[i]) / (self.n[i+1] - self.n[i]) \
                    + (self.n[i+1] - self.n[i] - s) * (self.x[i] - self.x[i-1]) / (self.n[i] - self.n[i-1])
                xip = self.x[i] + s * num / (self.n[i+1] - self.n[i-1])

                # Monotonicity guard -> linear
                if not (self.x[i-1] < xip < self.x[i+1]):
                    xip = self.x[i] + s * (self.x[i+s] - self.x[i]) / (self.n[i+s] - self.n[i])

                # Update
                self.x[i] = xip
                self.n[i] += s

                # Optional hard clamp (extra safety)
                if self.x[i] <= self.x[i-1]:
                    self.x[i] = np.nextafter(self.x[i-1], np.inf)
                if self.x[i] >= self.x[i+1]:
                    self.x[i] = np.nextafter(self.x[i+1], -np.inf)

