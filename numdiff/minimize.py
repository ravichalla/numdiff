"""Delta-debugging minimizer: shrink a failing input to a small reproducer."""
from __future__ import annotations

from typing import Callable

import numpy as np


def ddmin(x: np.ndarray, is_bad: Callable[[np.ndarray], bool], max_tests: int = 400) -> np.ndarray:
    """Return a (locally) minimal subsequence of x for which is_bad(x) is still True."""
    assert is_bad(x), "input must fail to begin with"
    n, tests = 2, 0
    while len(x) >= 2 and tests < max_tests:
        chunk = max(1, len(x) // n)
        reduced = False
        for start in range(0, len(x), chunk):
            cand = np.concatenate([x[:start], x[start + chunk:]])
            if len(cand) == 0:
                continue
            tests += 1
            if is_bad(cand):
                x, n, reduced = cand, max(n - 1, 2), True
                break
        if not reduced:
            if chunk == 1:
                break
            n = min(len(x), n * 2)
    return x
