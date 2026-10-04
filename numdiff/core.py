"""Build, run, reference-compute and compare. Everything else (CLI, agent) sits on top of this."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "kernels.cpp"
BUILD = ROOT / "build"
EPS32 = 2.0 ** -24

# kernel -> source variants (must match the registry in kernels.cpp)
VARIANTS = {
    "sum": ["naive", "unroll4", "pairwise"],
    "dot": ["naive", "unroll4", "pairwise"],
    "softmax": ["naive", "online"],
    "layernorm": ["naive", "welford"],
    "prefix": ["naive", "blocked"],
}
REDUCTIONS = {"sum", "dot", "prefix"}  # judged by condition-aware error, not raw ULPs
BASELINE_CONFIG = "O0"
BASELINE_VARIANT = "naive"

# compiler config name -> flags. "ir_*" configs go through clang -> opt -> clang (needs LLVM).
CONFIGS = {
    "O0": ["-O0"],
    "O3": ["-O3"],
    "O3_fastmath": ["-O3", "-ffast-math"],
    "O3_fma": ["-O3", "-march=native", "-ffp-contract=fast"],
}
IR_CONFIGS = {  # name -> (frontend flags, opt pipeline)
    "ir_default_O3": ([], "default<O3>"),
    "ir_default_O3_fast": (["-ffast-math"], "default<O3>"),
}
DISTS = ["uniform", "wide", "cancel"]


# ----------------------------------------------------------------------------- build
def _tool(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def llvm_available() -> bool:
    return bool(_tool("clang++") and _tool("opt"))


def all_configs() -> list[str]:
    return list(CONFIGS) + (list(IR_CONFIGS) if llvm_available() else [])


def build(config: str, force: bool = False) -> Path:
    BUILD.mkdir(exist_ok=True)
    exe = BUILD / f"runner_{config}"
    if exe.exists() and not force and exe.stat().st_mtime >= SRC.stat().st_mtime:
        return exe
    if config in CONFIGS:
        cxx = os.environ.get("CXX") or _tool("g++", "clang++", "c++")
        cmd = [cxx, "-std=c++17", *CONFIGS[config], str(SRC), "-o", str(exe)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    elif config in IR_CONFIGS:
        fe_flags, pipeline = IR_CONFIGS[config]
        clang, opt = _tool("clang++"), _tool("opt")
        if not (clang and opt):
            raise RuntimeError("LLVM (clang++ and opt) not found; IR configs unavailable")
        ll0, ll1 = BUILD / f"{config}.0.ll", BUILD / f"{config}.1.ll"
        subprocess.run([clang, "-std=c++17", "-O0", "-Xclang", "-disable-O0-optnone", *fe_flags,
                        "-S", "-emit-llvm", str(SRC), "-o", str(ll0)], check=True, capture_output=True, text=True)
        subprocess.run([opt, f"-passes={pipeline}", "-S", str(ll0), "-o", str(ll1)],
                       check=True, capture_output=True, text=True)
        subprocess.run([clang, "-O0", str(ll1), "-o", str(exe)], check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"unknown config {config}")
    return exe


# ----------------------------------------------------------------------------- inputs
def gen_input(kernel: str, dist: str, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = 2 * n if kernel == "dot" else n
    if dist == "uniform":
        x = rng.uniform(-1, 1, m)
    elif dist == "wide":  # log-uniform magnitude 1e-3..1e3, random sign
        x = 10.0 ** rng.uniform(-3, 3, m) * rng.choice([-1.0, 1.0], m)
    elif dist == "cancel":  # big values that cancel in pairs + small noise
        h = m // 2
        v = rng.uniform(1, 1e4, h)
        x = np.concatenate([v, -v, np.zeros(m - 2 * h)])
        rng.shuffle(x)
        x = x + rng.uniform(-1e-2, 1e-2, m)
    else:
        raise ValueError(dist)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------- reference (float64)
def reference(kernel: str, x32: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (float64 reference output, per-element scale used to normalise absolute error)."""
    x = x32.astype(np.float64)
    if kernel == "sum":
        return np.array([math.fsum(x)]), np.array([np.abs(x).sum()])
    if kernel == "dot":
        n = len(x) // 2
        p = x[:n] * x[n:]  # exact: 24+24 bit products fit in float64
        return np.array([math.fsum(p)]), np.array([np.abs(p).sum()])
    if kernel == "prefix":
        return np.cumsum(x), np.cumsum(np.abs(x))
    if kernel == "softmax":
        e = np.exp(x - x.max())
        r = e / e.sum()
        return r, np.full(len(r), r.max())  # scale by the largest output
    if kernel == "layernorm":
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        r = (x - mu) / np.sqrt(var + 1e-5)
        return r, np.full(len(r), np.abs(r).max())
    raise ValueError(kernel)


# ----------------------------------------------------------------------------- metrics
def _ordered(a: np.ndarray) -> np.ndarray:
    i = a.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(2 ** 31) - i, i)  # monotone map of float32 bit patterns


def ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(_ordered(a) - _ordered(b)).astype(np.float64)
    bad = ~np.isfinite(a) | ~np.isfinite(b)
    return np.where(bad, 2.0 ** 31, d)


def tolerance(kernel: str, n: int) -> tuple[str, float]:
    # norm_err = max |y - ref| / (eps32 * scale). Raw ULPs are misleading near zero (cancellation,
    # tiny softmax outputs, denormals), so ULPs are reported but pass/fail uses the scaled error.
    # Reductions: scale = sum of |terms|; softmax/layernorm: scale = largest |output|.
    return "norm_err", (2.0 if kernel in REDUCTIONS else 4.0) * math.sqrt(n)


# ----------------------------------------------------------------------------- run
def run_one(config: str, kernel: str, variant: str, x32: np.ndarray, reps: int = 1) -> tuple[np.ndarray, float]:
    exe = build(config)
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        x32.astype("<f4").tofile(fin)
        r = subprocess.run([str(exe), f"{kernel}/{variant}", str(fin), str(fout), str(reps)],
                           check=True, capture_output=True, text=True)
        t = float(r.stdout.strip().split("=")[1])
        return np.fromfile(fout, dtype="<f4"), t


@dataclass
class Row:
    kernel: str
    variant: str
    config: str
    dist: str
    n: int
    lossy: bool
    max_ulp: float
    norm_err: float
    bit_exact_vs_baseline: bool
    max_ulp_vs_baseline: float
    time_ns: float
    status: str  # PASS | FAIL | LOSSY
    score: float  # the metric used for pass/fail and regression tracking

    @property
    def key(self) -> str:
        return f"{self.kernel}/{self.variant}@{self.config}|{self.dist}|n={self.n}"


def evaluate(kernel: str, variant: str, config: str, dist: str, n: int, seed: int = 0,
             reps: int = 3, x32: np.ndarray | None = None) -> Row:
    x32 = gen_input(kernel, dist, n, seed) if x32 is None else x32
    lossy = variant.endswith("+bf16")
    base_variant = variant.removesuffix("+bf16")
    y, t = run_one(config, kernel, variant, x32, reps)
    y_base, _ = run_one(BASELINE_CONFIG, kernel, BASELINE_VARIANT, x32, 1)
    ref64, scale = reference(kernel, x32)
    ref32 = ref64.astype(np.float32)

    max_ulp = float(ulp_dist(y, ref32).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        ne = np.abs(y.astype(np.float64) - ref64) / (EPS32 * np.maximum(scale, 1e-300))
    norm_err = float(np.nanmax(ne)) if np.isfinite(ne).all() else float("inf")
    metric, tol = tolerance(kernel, len(x32) // (2 if kernel == "dot" else 1))
    score = norm_err
    status = "LOSSY" if lossy else ("PASS" if score <= tol else "FAIL")
    return Row(kernel, variant, config, dist, n, lossy, max_ulp, norm_err,
               bool(np.array_equal(y.view(np.uint32), y_base.view(np.uint32))) if y.shape == y_base.shape else False,
               float(ulp_dist(y, y_base).max()) if y.shape == y_base.shape else float("inf"),
               t, status, float(score))


def run_matrix(kernels: list[str], configs: list[str], dists: list[str], n: int,
               seed: int = 0, reps: int = 3, bf16: bool = False) -> list[Row]:
    rows = []
    for k in kernels:
        for v in VARIANTS[k] + ([f"{VARIANTS[k][0]}+bf16"] if bf16 else []):
            for c in configs:
                for d in dists:
                    rows.append(evaluate(k, v, c, d, n, seed, reps))
    return rows


# ----------------------------------------------------------------------------- baselines / regressions
def save_baseline(rows: list[Row], path: str | Path) -> None:
    Path(path).write_text(json.dumps({r.key: r.score for r in rows if not r.lossy}, indent=1))


def regressions(rows: list[Row], path: str | Path, factor: float = 2.0, slack: float = 1.0) -> list[tuple[str, float, float]]:
    old = json.loads(Path(path).read_text())
    return [(r.key, old[r.key], r.score) for r in rows
            if r.key in old and r.score > factor * old[r.key] + slack]


def rows_to_json(rows: list[Row]) -> str:
    return json.dumps([{**asdict(r), "key": r.key} for r in rows], indent=1)


def rows_to_markdown(rows: list[Row]) -> str:
    out = ["| kernel/variant | config | dist | max ULP | norm err | bit-exact vs base | max ULP vs base | time (µs) | status |",
           "|---|---|---|---:|---:|:--:|---:|---:|:--:|"]
    for r in rows:
        ne = f"{r.norm_err:.3g}"
        out.append(f"| {r.kernel}/{r.variant} | {r.config} | {r.dist} | {r.max_ulp:.3g} | {ne} | "
                   f"{'yes' if r.bit_exact_vs_baseline else 'no'} | {r.max_ulp_vs_baseline:.3g} | {r.time_ns/1e3:.1f} | {r.status} |")
    return "\n".join(out)
