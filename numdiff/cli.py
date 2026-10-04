from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import core
from .agent import LLMProposer, RandomProposer, run_agent
from .minimize import ddmin


def cmd_run(a):
    kernels = a.kernels or list(core.VARIANTS)
    configs = a.configs or core.all_configs()
    rows = core.run_matrix(kernels, configs, a.dists, a.n, a.seed, a.reps, a.bf16)
    md = core.rows_to_markdown(rows)
    print(md)
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "results.json").write_text(core.rows_to_json(rows))
    (Path(a.out) / "results.md").write_text(md + "\n")
    if a.save_baseline:
        core.save_baseline(rows, a.save_baseline)
        print(f"\nbaseline saved to {a.save_baseline}")
    code = 0
    if a.baseline:
        regs = core.regressions(rows, a.baseline)
        for key, old, new in regs:
            print(f"REGRESSION {key}: {old:.3g} -> {new:.3g}")
        code |= bool(regs)
    fails = [r for r in rows if r.status == "FAIL"]
    for r in fails:
        print(f"FAIL {r.key}: score={r.score:.3g}")
    return int(code or bool(fails))


def cmd_minimize(a):
    kind, tol = core.tolerance(a.kernel, a.n)
    x = core.gen_input(a.kernel, a.dist, a.n, a.seed)

    def is_bad(arr: np.ndarray) -> bool:
        if a.kernel == "dot" and len(arr) % 2:
            return False  # dot needs an even-length input (a ++ b)
        r = core.evaluate(a.kernel, a.variant, a.config, a.dist, len(arr), reps=1, x32=arr.astype(np.float32))
        return r.max_ulp_vs_baseline > a.ulp_threshold

    if not is_bad(x):
        print("input does not exceed the threshold; nothing to minimize")
        return 1
    small = ddmin(x, is_bad)
    print(f"minimized {len(x)} -> {len(small)} elements")
    np.set_printoptions(precision=9, linewidth=100)
    print(small)
    Path(a.out).mkdir(exist_ok=True)
    small.astype("<f4").tofile(Path(a.out) / f"repro_{a.kernel}_{a.variant}_{a.config}.bin")
    return 0


def cmd_agent(a):
    proposer = LLMProposer(seed=a.seed) if a.llm else RandomProposer(a.seed)
    Path(a.out).mkdir(exist_ok=True)
    hist = run_agent(a.kernel, a.steps, a.n, proposer, Path(a.out) / f"agent_{a.kernel}.jsonl", seed=a.seed)
    best = max((h for h in hist if h["status"] == "PASS"), key=lambda h: h["speedup"], default=None)
    for h in hist:
        print(f"{h['step']:>3} {h['variant']:<9} {h['config']:<14} {h['status']:<4} speedup={h['speedup']:<7} reward={h['reward']}")
    print("\nbest correct candidate:", best)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="numdiff")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the kernel x variant x config x distribution matrix")
    r.add_argument("--kernels", nargs="*", choices=list(core.VARIANTS))
    r.add_argument("--configs", nargs="*")
    r.add_argument("--dists", nargs="*", default=core.DISTS)
    r.add_argument("-n", type=int, default=1 << 16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--bf16", action="store_true", help="also run a bf16-rounded-input variant (reported as LOSSY)")
    r.add_argument("--out", default="out")
    r.add_argument("--save-baseline")
    r.add_argument("--baseline", help="compare against a saved baseline; exit 1 on regression")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("minimize", help="shrink an input that makes a variant diverge from the baseline")
    m.add_argument("kernel", choices=list(core.VARIANTS))
    m.add_argument("variant")
    m.add_argument("config")
    m.add_argument("--dist", default="cancel")
    m.add_argument("-n", type=int, default=256)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--ulp-threshold", type=float, default=1000.0)
    m.add_argument("--out", default="out")
    m.set_defaults(fn=cmd_minimize)

    g = sub.add_parser("agent", help="verifier-in-the-loop search for the fastest correct variant")
    g.add_argument("kernel", choices=list(core.VARIANTS))
    g.add_argument("--steps", type=int, default=8)
    g.add_argument("-n", type=int, default=1 << 18)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--llm", action="store_true", help="use Claude as the proposer (needs ANTHROPIC_API_KEY)")
    g.add_argument("--out", default="out")
    g.set_defaults(fn=cmd_agent)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
