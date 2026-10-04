# numdiff

A small differential-testing harness for numerical kernels, with a verifier-in-the-loop search on top.
It compares each kernel across **compiler configurations** (`-O0`, `-O3`, `-ffast-math`, FMA contraction,
optional LLVM-IR pass pipelines) and **source-level variants** (naive, unrolled, pairwise, online, Welford,
blocked), checks every result against a float64 reference, and reports where results diverge.

Scope: a small CPU harness, not a production compiler or a GPU framework.

## Quick start
```bash
pip install numpy pytest          # anthropic is optional (LLM proposer)
python -m numdiff.cli run -n 65536 --bf16            # full matrix -> out/results.{json,md}
python -m numdiff.cli run --save-baseline out/baseline.json
python -m numdiff.cli run --baseline out/baseline.json   # exit 1 on regression or FAIL
python -m numdiff.cli minimize sum naive O3_fastmath --dist cancel -n 256 --ulp-threshold 2
python -m numdiff.cli agent sum --steps 8                # heuristic proposer
ANTHROPIC_API_KEY=... python -m numdiff.cli agent sum --llm   # Claude as proposer
python -m pytest -q tests
```

## What is measured
- **max ULP** vs the float64 reference rounded to fp32 (informational: misleading near zero).
- **norm err** = max |y - ref| / (eps32 * scale). Scale is sum|terms| for sum/dot/prefix (a condition-aware
  bound) and the largest |output| for softmax/layernorm. **Pass/fail uses this.** Tolerance is
  2*sqrt(n) for reductions and 4*sqrt(n) otherwise (statistical bound for recursive summation; tune it).
- **bit-exact vs baseline** (`-O0`, naive variant) and **ULP vs baseline**: this is the differential part.
- `+bf16` variants round inputs to bfloat16 first and are reported as `LOSSY` (no pass/fail).
- Inputs: `uniform`, `wide` (log-uniform magnitudes 1e-3..1e3), `cancel` (large values cancelling in pairs).

## Components
| file | role |
|---|---|
| `src/kernels.cpp` | sum, dot, softmax, layernorm, prefix-sum, each with 2-3 source variants |
| `numdiff/core.py` | build matrix, input generation, float64 references, metrics, baselines/regressions |
| `numdiff/minimize.py` | delta-debugging minimizer: shrinks a diverging input to a small reproducer |
| `numdiff/agent.py` | proposer -> verifier -> reward loop over a closed menu; logs JSONL |
| `numdiff/cli.py` | `run`, `minimize`, `agent` |

## The agent loop
A proposer picks a `(variant, config)` from a fixed menu. The harness builds it, checks correctness on all
three distributions and times it. Reward = speedup over baseline if correct, else -1. Each step is logged to
`out/agent_<kernel>.jsonl` and fed back to the proposer as history. The menu is closed, so the model chooses
among transformations and never executes code it wrote.

## Findings from my runs (n = 65,536, g++ 13, x86-64): fill in with your own numbers
- `-ffast-math` changed the bits of sum/dot/softmax/layernorm vs `-O0` while staying within tolerance; it
  often *reduced* error (vectorised partial sums) rather than increasing it.
- Raw ULP error is meaningless on the `cancel` distribution (results near 0); scaled error is the useful signal.
- Welford layernorm was not more accurate than two-pass in fp32 on these inputs.
- The 18-70x "speedups" in the agent log are inflated: the `-O0` baseline can't inline `std::vector` accessors.
  Compare against `-O3` for meaningful numbers.

## Not done / caveats
- LLVM-IR configs (`ir_default_O3`, `ir_default_O3_fast`: clang -> `opt -passes=...` -> clang) are implemented
  but were **not exercised** (no LLVM in my sandbox). They auto-enable when `clang++` and `opt` are on PATH.
- No GPU path. Natural extension: a CUDA backend for the same kernels, diffed against these CPU results.
- Timing is a single-process min-of-reps measurement, not a rigorous benchmark.
- Ideas: MPFR reference, more kernels (GEMM tile, attention), fp16 emulation, real equivalence checking
  (e.g. Alive2 on IR pairs), and coverage-guided input generation.
