"""Verifier-in-the-loop search over compiler/source transformations.

A proposer (LLM or heuristic) suggests (variant, config) pairs from a fixed menu. The harness is the
verifier: it builds, runs on several input distributions, checks numerical correctness against a float64
reference and times the result. Reward = speedup over baseline if correct, else a penalty. Every step is
logged as JSONL, which doubles as the memory handed back to the proposer.

The menu is closed on purpose: the model never gets to run arbitrary code, only to pick transformations.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from . import core


def menu(kernel: str, configs: list[str]) -> list[dict]:
    return [{"variant": v, "config": c} for v in core.VARIANTS[kernel] for c in configs]


class RandomProposer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, kernel, options, history):
        tried = {(h["variant"], h["config"]) for h in history}
        fresh = [o for o in options if (o["variant"], o["config"]) not in tried]
        return self.rng.choice(fresh or options)


class LLMProposer:
    """Asks Claude for the next transformation. Needs `pip install anthropic` and ANTHROPIC_API_KEY.
    Falls back to random if the reply is not a valid menu entry."""

    def __init__(self, model: str | None = None, seed: int = 0):
        import anthropic  # imported lazily so the rest of the project has no hard dependency
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("NUMDIFF_MODEL", "claude-sonnet-5-5")
        self.fallback = RandomProposer(seed)

    def propose(self, kernel, options, history):
        prompt = (
            f"You are tuning the '{kernel}' kernel. Goal: maximise speedup while staying numerically correct.\n"
            f"Options (choose exactly one, do not repeat a tried pair):\n{json.dumps(options)}\n\n"
            f"History (variant, config, status, speedup, max_ulp, reward):\n"
            + "\n".join(json.dumps({k: h[k] for k in ("variant", "config", "status", "speedup", "max_ulp", "reward")})
                        for h in history)
            + '\n\nReply with ONLY a JSON object like {"variant": "...", "config": "...", "why": "..."}.'
        )
        try:
            msg = self.client.messages.create(model=self.model, max_tokens=300,
                                              messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            for o in options:
                if o["variant"] == obj.get("variant") and o["config"] == obj.get("config"):
                    return {**o, "why": obj.get("why", "")}
        except Exception as e:  # network, parse, or SDK errors: never crash the loop
            print(f"[llm proposer] falling back to random: {e}")
        return self.fallback.propose(kernel, options, history)


def verify(kernel: str, variant: str, config: str, n: int, dists: list[str], seed: int) -> dict:
    """Run one candidate on every distribution. Correct only if all pass; time is the mean."""
    rows = [core.evaluate(kernel, variant, config, d, n, seed, reps=5) for d in dists]
    ok = all(r.status == "PASS" for r in rows)
    return {"ok": ok, "time_ns": sum(r.time_ns for r in rows) / len(rows),
            "max_ulp": max(r.max_ulp for r in rows),
            "worst_score": max(r.score for r in rows),
            "bit_exact": all(r.bit_exact_vs_baseline for r in rows)}


def run_agent(kernel: str, steps: int, n: int, proposer, log_path: str | Path,
              dists: list[str] | None = None, seed: int = 0) -> list[dict]:
    dists = dists or core.DISTS
    configs = core.all_configs()
    options = menu(kernel, configs)
    base = verify(kernel, core.BASELINE_VARIANT, core.BASELINE_CONFIG, n, dists, seed)
    history: list[dict] = []
    with open(log_path, "w") as log:
        for step in range(steps):
            p = proposer.propose(kernel, options, history)
            res = verify(kernel, p["variant"], p["config"], n, dists, seed)
            speedup = base["time_ns"] / res["time_ns"] if res["time_ns"] > 0 else 0.0
            reward = speedup if res["ok"] else -1.0
            entry = {"step": step, "kernel": kernel, "variant": p["variant"], "config": p["config"],
                     "why": p.get("why", ""), "status": "PASS" if res["ok"] else "FAIL",
                     "speedup": round(speedup, 3), "max_ulp": res["max_ulp"],
                     "worst_score": res["worst_score"], "bit_exact": res["bit_exact"], "reward": round(reward, 3)}
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
    return history
