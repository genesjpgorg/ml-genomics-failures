"""Summarize a graded eval: per-arm rates, per-task table, and paired bootstrap differences vs the baseline arm.

usage: evals/.venv/bin/python evals/report.py --out evals/results/pilot

The unit of independence is the task, not the run: repetitions are averaged within each task first, and confidence
intervals come from resampling tasks (paired across arms, since every arm sees the same tasks).
Writes report.md and summary.json in --out.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_NAMES = {"0": "baseline", "1": "book as text", "2": "MCP, unprompted", "3": "MCP, prompted"}


def load(out: Path) -> list[dict]:
    rows = []
    for rj in sorted((out / "runs").glob("*/arm*/rep*/run.json")):
        jj = rj.with_name("judge.json")
        if not jj.exists():
            continue
        run, judge = json.loads(rj.read_text()), json.loads(jj.read_text())
        rows.append({**{k: run[k] for k in ("task", "arm", "rep", "cost_usd", "duration_s", "turns", "contaminated", "is_error", "timed_out")},
                     "mcp_calls": len(run["mcp_calls"]), "mcp_tools": run["mcp_calls"], **judge})  # fmt: skip
    return rows


def rate(values: list) -> float | None:
    vals = [float(bool(v)) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def per_task(rows: list[dict], metric: str, flawed: bool | None) -> dict[str, dict[str, float]]:
    """task -> arm -> mean of metric over reps (tasks filtered by flawed status)."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (flawed is None or r["flawed"] == flawed) and r.get(metric) is not None:
            acc[r["task"]][r["arm"]].append(float(bool(r[metric])))
    return {t: {a: float(np.mean(v)) for a, v in arms.items()} for t, arms in acc.items()}


def paired_bootstrap(table: dict, arm: str, base: str = "0", n: int = 10000, seed: int = 0) -> dict | None:
    tasks = [t for t in table if arm in table[t] and base in table[t]]
    if len(tasks) < 2:
        return None
    d = np.array([table[t][arm] - table[t][base] for t in tasks])
    boots = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return {
        "diff": round(float(d.mean()), 3),
        "ci95": [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])],
        "n_tasks": len(tasks),
    }


def fmt(x: float | None, pct: bool = True) -> str:
    return "–" if x is None else (f"{x:.0%}" if pct else f"{x:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = load(a.out)
    if not rows:
        raise SystemExit("no graded runs; run judge.py first")
    config = json.loads((a.out / "config.json").read_text())
    arms = sorted({r["arm"] for r in rows})
    summary: dict = {"config": config, "n_runs": len(rows), "arms": {}}
    lines = [
        f"# Eval report: {a.out.name}",
        "",
        (
            f"Model under test: `{config['model']}` · judge: `{rows[0].get('judge_model')}` · code: `{config['git']}` · "
            f"{len(rows)} graded runs ({len({r['task'] for r in rows})} tasks × {len(arms)} arms × {config['reps']} reps)"
        ),
        "",
        "## By arm",
        "",
        "| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        fl, cl = [r for r in rs if r["flawed"]], [r for r in rs if not r["flawed"]]
        s = {
            "n": len(rs),
            "verdict_correct": rate([r["verdict_correct"] for r in rs]),
            "verdict_correct_flawed": rate([r["verdict_correct"] for r in fl]),
            "verdict_correct_clean": rate([r["verdict_correct"] for r in cl]),
            "detected": rate([r.get("detected") for r in fl]),
            "quantified": rate([r.get("quantified") for r in fl]),
            "fix": rate([r.get("fix") for r in fl]),
            "false_alarm": rate([r.get("false_alarm") for r in cl]),
            "used_mcp": rate([r["mcp_calls"] > 0 for r in rs]),
            "mcp_calls_per_run": float(np.mean([r["mcp_calls"] for r in rs])),
            "cost_per_run": float(np.mean([r["cost_usd"] or 0 for r in rs])),
            "time_per_run_s": float(np.mean([r["duration_s"] or 0 for r in rs])),
            "contaminated": sum(r["contaminated"] for r in rs),
            "errors": sum(r["is_error"] or r["timed_out"] or ("error" in r) for r in rs),
        }
        summary["arms"][arm] = s
        lines.append(
            f"| {arm} {ARM_NAMES.get(arm, '')} | {fmt(s['verdict_correct'])} | {fmt(s['detected'])} | {fmt(s['quantified'])} | "
            f"{fmt(s['fix'])} | {fmt(s['false_alarm'])} | {fmt(s['used_mcp'])} | {s['mcp_calls_per_run']:.1f} | "
            f"${s['cost_per_run']:.2f} | {s['time_per_run_s']:.0f}s |"
        )
    lines += ["", "## Difference from baseline (paired over tasks, 95% bootstrap CI)", "",
              "| Metric | " + " | ".join(f"arm {x} − arm 0" for x in arms if x != "0") + " |",
              "|---|" + "---|" * (len(arms) - 1)]  # fmt: skip
    summary["vs_baseline"] = {}
    for metric, flawed, label in [
        ("verdict_correct", None, "Verdict correct (all tasks)"),
        ("detected", True, "Detected planted flaw (flawed tasks)"),
        ("quantified", True, "Quantified it (flawed tasks)"),
        ("false_alarm", False, "False alarm (clean tasks)"),
    ]:
        table = per_task(rows, metric, flawed)
        cells = []
        for arm in arms:
            if arm == "0":
                continue
            b = paired_bootstrap(table, arm)
            summary["vs_baseline"].setdefault(metric, {})[arm] = b
            cells.append("–" if b is None else f"{b['diff']:+.0%} [{b['ci95'][0]:+.0%}, {b['ci95'][1]:+.0%}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## By task", "", "Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.", "",
              "| Task | Expected | " + " | ".join(f"arm {x}" for x in arms) + " |", "|---|---|" + "---|" * len(arms)]  # fmt: skip
    vc = per_task(rows, "verdict_correct", None)
    det = per_task(rows, "detected", True)
    fa = per_task(rows, "false_alarm", False)
    for t in sorted(vc):
        flawed = t in det
        cells = []
        for arm in arms:
            second = det.get(t, {}).get(arm) if flawed else fa.get(t, {}).get(arm)
            cells.append(f"{fmt(vc[t].get(arm))} / {'det' if flawed else 'FA'} {fmt(second)}")
        exp = "flawed" if flawed else "clean"
        lines.append(f"| {t} | {exp} | " + " | ".join(cells) + " |")
    tool_use = defaultdict(int)
    for r in rows:
        for name in r["mcp_tools"]:
            tool_use[name] += 1
    if tool_use:
        lines += [
            "",
            "## MCP tool calls",
            "",
            ", ".join(f"`{k}` {v}" for k, v in sorted(tool_use.items(), key=lambda kv: -kv[1])),
        ]
    flagged = [r for r in rows if r["contaminated"] or r["is_error"] or r["timed_out"] or "error" in r]
    if flagged:
        lines += ["", "## Flagged runs", ""] + [
            f"- {r['task']} arm{r['arm']} rep{r['rep']}: "
            + ", ".join(k for k in ("contaminated", "is_error", "timed_out") if r[k]) + (f" judge error: {r['error'][:80]}" if "error" in r else "")
            for r in flagged
        ]  # fmt: skip
    lines += [
        "",
        "Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.",
    ]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
