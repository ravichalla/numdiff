import numpy as np
from numdiff import core
from numdiff.minimize import ddmin


def test_ulp_dist_basics():
    a = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    assert core.ulp_dist(a, a).max() == 0
    assert core.ulp_dist(np.float32([1.0]), np.nextafter(np.float32([1.0]), np.float32([2.0])))[0] == 1
    assert core.ulp_dist(np.float32([np.nan]), np.float32([1.0]))[0] > 1e9


def test_baseline_is_bit_exact_with_itself():
    r = core.evaluate("sum", "naive", core.BASELINE_CONFIG, "uniform", 1024, reps=1)
    assert r.bit_exact_vs_baseline and r.status == "PASS"


def test_fastmath_changes_bits_but_stays_accurate():
    r = core.evaluate("sum", "naive", "O3_fastmath", "cancel", 4096, reps=1)
    assert not r.bit_exact_vs_baseline and r.status == "PASS"


def test_bf16_is_flagged_lossy():
    r = core.evaluate("sum", "naive+bf16", "O3", "uniform", 4096, reps=1)
    assert r.status == "LOSSY" and r.score > 100


def test_ddmin_finds_minimal_pair():
    x = np.arange(64, dtype=np.float32)
    small = ddmin(x, lambda a: 3.0 in a and 40.0 in a)
    assert sorted(small.tolist()) == [3.0, 40.0]
