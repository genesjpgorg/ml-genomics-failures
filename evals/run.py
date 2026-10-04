"""Run the eval: every task x arm x repetition as one headless Claude Code session.

usage: evals/.venv/bin/python evals/run.py --out evals/results/pilot [--arms 0,1,3] [--reps 2] [--tasks T01,C02]
                                           [--model claude-sonnet-5-5] [--backend claude|devin] [--parallel 4]

Arms:
  0  baseline: no book, no MCP
  1  the book as text: dist/AGENTS.md appended to the system prompt (claude backend only)
  2  MCP server available, not mentioned in the prompt
  3  MCP server available, and the prompt asks to audit with it

Backends:
  claude  headless Claude Code in --restricted mode (default)
  devin   headless `devin -p` sessions (Devin CLI). Each run gets a workspace copy with a project-scope
          .devin/mcp_config.json that overrides the account plugin: disabled for arm 0, this repo's local
          build for arms 2 and 3. The trajectory is exported in ATIF and parsed for tool calls.
          Devin exposes MCP through generic mcp_* tools, so an unprompted arm-2 agent must discover the
          server itself: mcp_list_servers -> mcp_list_tools -> mcp_call_tool.

Isolation: each run gets a fresh copy of the task workspace in a temp directory outside this repo, Claude Code in
--restricted mode (user/project settings ignored, file tools confined to the workspace), only the MCP servers the
arm defines (--strict-mcp-config), no web tools, Write/Edit confined to the workspace copy, and Bash limited to
read-only commands and Python. Python can still
read any file, so runs whose tool calls touch this repo (rubrics, AGENTS.md) are flagged as contaminated.

Re-running with the same --out skips finished runs, so an interrupted eval resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVALS = Path(__file__).resolve().parent
REPO = EVALS.parent
TASKS = EVALS / "tasks"
AGENT_PYTHON_BIN = EVALS / ".venv" / "bin"
MCP_SERVER = REPO / ".venv" / "bin" / "ml-genomics-failures-mcp"

PROMPT = """A colleague wrote up the analysis in this directory (README.md and the files next to it). Review it \
before it goes into a paper: is the main claim supported by the evidence? Check the data and results yourself \
where you can: create scripts in this directory with the Write tool and run them with Python (numpy, pandas, scipy and scikit-learn are \
installed).

End your answer with a line `VERDICT: SUPPORTED` or `VERDICT: NOT SUPPORTED`, followed by the issues that matter \
most."""
# The realistic case: the agent is asked to write the analysis up, and nobody mentions auditing it.
WRITEUP = """Our analysis is in this directory (README.md and the files next to it). Draft the Results paragraph \
for our paper, with the key numbers. Python (numpy, pandas, scipy, scikit-learn) is available if you need it; create \
scripts with the Write tool."""
# The forward case: the agent produces the metric itself; nothing primes an audit.
TRAIN = """Train a classifier on the data in this directory (README.md describes it) and report its held-out \
performance. Python (numpy, pandas, scipy, scikit-learn) is available; create scripts with the Write tool."""
PROMPTS = {"review": PROMPT, "writeup": WRITEUP, "train": TRAIN}

ARM3_SUFFIX = """

Use the ml-genomics-failures MCP server to audit this: its checklist, the matching cases, and its check tools on \
the data."""

ARMS = {
    "0": {"mcp": False, "book": False, "suffix": ""},
    "1": {"mcp": False, "book": True, "suffix": ""},
    "2": {"mcp": True, "book": False, "suffix": ""},
    "3": {"mcp": True, "book": False, "suffix": ARM3_SUFFIX},
}
BASH = ["python", "python3", "ls", "head", "tail", "wc", "cat", "grep", "sort", "cut"]
CONTAMINATION = (
    "rubric.json", "ml-genomics-failures/evals", "ml-genomics-failures/dist", "AGENTS.md", "FAILURES.md",
    "make_tasks.py", "genesjpgorg",
)
WEB_TOOLS = {"web_search", "webfetch", "browser_preview"}
MCP_DISABLED = {"mcpServers": {"ml-genomics-failures": {"command": "true", "disabled": True}}}
MCP_LOCAL = {"mcpServers": {"ml-genomics-failures": {"command": str(MCP_SERVER), "args": []}}}


def claude_cmd(arm: dict, mcp_config: Path, model: str, prompt: str) -> list[str]:
    allowed = ["Read", "Write", "Edit", "Glob", "Grep", *(f"Bash({c}:*)" for c in BASH)]
    if arm["mcp"]:
        allowed.append("mcp__ml-genomics-failures")
    cmd = [
        "claude", "-p", prompt + arm["suffix"],
        "--model", model,
        "--restricted", "--tools", "Bash,Read,Write,Edit,Glob,Grep",
        "--allowedTools", *allowed,
        "--strict-mcp-config", "--mcp-config", str(mcp_config),
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
    ]  # fmt: skip
    if arm["book"]:
        cmd += ["--append-system-prompt-file", str(REPO / "dist" / "AGENTS.md")]
    return cmd


def parse_stream(lines: list[str]) -> dict:
    tools, answer, final = [], "", {}
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    tools.append({"name": c["name"], "input": c.get("input", {})})
                elif c.get("type") == "text":
                    answer = c["text"]  # keep the last assistant text as a fallback answer
        elif e.get("type") == "result":
            final = e
    return {
        "answer": final.get("result") or answer,
        "is_error": bool(final.get("is_error")) or not final,
        "api_error_status": final.get("api_error_status"),
        "cost_usd": final.get("total_cost_usd"),
        "duration_s": round(final.get("duration_ms", 0) / 1000, 1),
        "turns": final.get("num_turns"),
        "tool_calls": [t["name"] for t in tools],
        "mcp_calls": [t["name"].split("__")[-1] for t in tools if t["name"].startswith("mcp__")],
        "contaminated": any(m in json.dumps(t["input"]) for t in tools for m in CONTAMINATION),
        "tool_inputs": tools,
    }


def devin_cmd(arm: dict, model: str, prompt: str, atif: Path) -> list[str]:
    return [
        "devin", "-p", prompt + arm["suffix"],
        "--model", model,
        "--permission-mode", "dangerous",  # headless runs can't answer prompts; the workspace is disposable
        "--export", str(atif),
        "--respect-workspace-trust", "false",
    ]


def parse_atif(path: Path, stdout: str) -> dict:
    """Tool calls and the final answer from a Devin ATIF trajectory."""
    tools, answer = [], ""
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        d = {}
    for s in d.get("steps", []):
        for tc in s.get("tool_calls") or []:
            tools.append({"name": tc.get("function_name"), "input": tc.get("arguments", {})})
        if s.get("source") == "assistant" and isinstance(s.get("message"), str) and s["message"].strip():
            answer = s["message"]
    mcp = []
    for t in tools:
        if t["name"] == "mcp_call_tool":
            mcp.append(str(t["input"].get("tool_name", t["input"].get("name", "?"))))
        elif t["name"] in ("mcp_list_servers", "mcp_list_tools"):
            mcp.append(t["name"])
    blob = json.dumps([t["input"] for t in tools])
    contaminated = any(m in blob for m in CONTAMINATION) or "mcp_config" in blob
    web = [t["name"] for t in tools if t["name"] in WEB_TOOLS]
    fm = d.get("final_metrics", {})
    return {
        "answer": answer or stdout.strip(),
        "tool_calls": [t["name"] for t in tools],
        "mcp_calls": mcp,
        "contaminated": contaminated or bool(web),
        "tool_inputs": tools,
        "turns": sum(1 for s in d.get("steps", []) if s.get("tool_calls")),
        "prompt_tokens": fm.get("total_prompt_tokens"),
        "completion_tokens": fm.get("total_completion_tokens"),
    }


class LimitReached(RuntimeError):
    """The account hit a usage or rate limit: the run says nothing about the agent, so it isn't recorded."""


def run_one(
    task: str, arm_id: str, rep: int, out: Path, model: str, timeout: int,
    prompt: str = "review", backend: str = "claude",
) -> dict:
    if backend == "devin":
        return run_one_devin(task, arm_id, rep, out, model, timeout, prompt)
    run_dir = out / "runs" / task / f"arm{arm_id}" / f"rep{rep}"
    if (run_dir / "run.json").exists():
        return json.loads((run_dir / "run.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_id]
    with tempfile.TemporaryDirectory(prefix="review-") as tmp:
        ws = Path(tmp) / "analysis"
        shutil.copytree(TASKS / task / "workspace", ws)
        servers = {"ml-genomics-failures": {"command": str(MCP_SERVER), "args": []}} if arm["mcp"] else {}
        mcp_config = Path(tmp) / "mcp.json"
        mcp_config.write_text(json.dumps({"mcpServers": servers}))
        env = {**os.environ, "PATH": f"{AGENT_PYTHON_BIN}:{os.environ['PATH']}"}
        t0 = time.time()
        try:
            proc = subprocess.run(
                claude_cmd(arm, mcp_config, model, PROMPTS[prompt]),
                cwd=ws,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, timed_out = proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, timed_out = "timeout", True
    (run_dir / "transcript.jsonl").write_text(stdout)
    result = {
        "task": task,
        "arm": arm_id,
        "rep": rep,
        "model": model,
        "prompt": prompt,
        "prompt_text": PROMPTS[prompt] + arm["suffix"],
        "wall_s": round(time.time() - t0, 1),
        "timed_out": timed_out,
        "stderr_tail": stderr[-500:],
        **parse_stream(stdout.splitlines()),
    }
    if result["api_error_status"] == 429 or (result["is_error"] and "limit" in (result["answer"] or "").lower()):
        (run_dir / "transcript.jsonl").unlink()
        raise LimitReached(f"{task} arm{arm_id} rep{rep}: {result['answer']}")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2))
    return result


def run_one_devin(task: str, arm_id: str, rep: int, out: Path, model: str, timeout: int, prompt: str) -> dict:
    run_dir = out / "runs" / task / f"arm{arm_id}" / f"rep{rep}"
    if (run_dir / "run.json").exists():
        return json.loads((run_dir / "run.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_id]
    if arm["book"]:
        raise SystemExit("arm 1 (book as system prompt) isn't supported by the devin backend")
    tmp = Path(tempfile.mkdtemp(prefix="review-"))
    try:
        ws = tmp / "analysis"
        shutil.copytree(TASKS / task / "workspace", ws)
        (ws / ".devin").mkdir()
        (ws / ".devin" / "mcp_config.json").write_text(
            json.dumps(MCP_LOCAL if arm["mcp"] else MCP_DISABLED)
        )
        atif = tmp / "trajectory.json"
        env = {**os.environ, "PATH": f"{AGENT_PYTHON_BIN}:{os.environ['PATH']}"}
        t0 = time.time()
        try:
            proc = subprocess.run(
                devin_cmd(arm, model, PROMPTS[prompt], atif),
                cwd=ws,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, timed_out = proc.stdout, proc.stderr, proc.returncode != 0
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, timed_out = "timeout", True
        result = {
            "task": task,
            "arm": arm_id,
            "rep": rep,
            "model": model,
            "prompt": prompt,
            "prompt_text": PROMPTS[prompt] + arm["suffix"],
            "wall_s": round(time.time() - t0, 1),
            "timed_out": timed_out,
            "stderr_tail": stderr[-500:],
            "duration_s": round(time.time() - t0, 1),
            "cost_usd": None,
            "api_error_status": None,
            **parse_atif(atif, stdout),
        }
        result["is_error"] = timed_out or not result["answer"].strip()
        (run_dir / "transcript.jsonl").write_text(atif.read_text() if atif.exists() else stdout)
        if "usage limit" in result["answer"].lower() or "limit reached" in result["answer"].lower():
            raise LimitReached(f"{task} arm{arm_id} rep{rep}: {result['answer'][:200]}")
        (run_dir / "run.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", default="0,1,3")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tasks", default="all")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument(
        "--prompt",
        choices=sorted(PROMPTS),
        default="review",
        help="review: is the claim supported? writeup: draft the Results paragraph (no audit requested)",
    )
    p.add_argument("--backend", choices=["claude", "devin"], default="claude")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1200)
    a = p.parse_args()
    if not MCP_SERVER.exists():
        sys.exit(f"MCP server not found at {MCP_SERVER}; run `uv sync` in the repo first")
    tasks = sorted(d.name for d in TASKS.iterdir() if d.is_dir()) if a.tasks == "all" else a.tasks.split(",")
    arms = a.arms.split(",")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(
        json.dumps(
            {
                "tasks": tasks, "arms": arms, "reps": a.reps, "model": a.model, "git": sha,
                "prompt": a.prompt, "backend": a.backend,
            }, indent=2
        )
    )
    jobs = [(t, arm, r) for t in tasks for arm in arms for r in range(a.reps)]
    print(
        f"{len(jobs)} runs: {len(tasks)} tasks x arms {arms} x {a.reps} reps, model {a.model}, {a.parallel} parallel",
        flush=True,
    )
    total = 0.0
    with ThreadPoolExecutor(a.parallel) as pool:
        futs = {
            pool.submit(run_one, t, arm, r, a.out, a.model, a.timeout, a.prompt, a.backend): (t, arm, r)
            for t, arm, r in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            t, arm, r = futs[f]
            try:
                res = f.result()
            except LimitReached as e:
                for other in futs:
                    other.cancel()
                sys.exit(f"stopped: {e}. Finished runs are kept; rerun the same command to resume.")
            total += res.get("cost_usd") or 0
            flags = " ERROR" * res["is_error"] + " TIMEOUT" * res["timed_out"] + " CONTAMINATED" * res["contaminated"]
            print(
                f"[{i}/{len(jobs)}] {t} arm{arm} rep{r}: {res['duration_s']}s, ${res.get('cost_usd') or 0:.2f}, "
                f"{len(res['mcp_calls'])} MCP calls{flags}  (total ${total:.2f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
