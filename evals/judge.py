"""Grade every finished run against its task's hidden rubric, blind to the arm.

usage: evals/.venv/bin/python evals/judge.py --out evals/results/pilot [--model claude-opus-5-5] [--parallel 4]

Before grading, the answer is masked: case IDs (F01-F99), the server and tool names, and mentions of the book are
replaced, so the judge can't tell which arm produced it or reward name-dropping. The verdict is also parsed
deterministically from the answer's `VERDICT:` line; the judge's reading is kept for answers without one.
Writes judge.json next to each run.json; re-running skips graded runs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVALS = Path(__file__).resolve().parent
MASKS = [
    (re.compile(r"\bF\d{2}\b"), "[case]"),
    (re.compile(r"ml[-_]genomics[-_]failures", re.IGNORECASE), "[tool]"),
    (
        re.compile(
            r"\b(check_split_overlap|check_sequence_similarity|check_shortcuts|kmer_baseline|get_audit_checklist)\b"
        ),
        "[check]",
    ),
    (re.compile(r"book of failures", re.IGNORECASE), "[reference]"),
]

JUDGE = """You grade a reviewer's assessment of a scientific analysis against a hidden answer key. The reviewer saw \
only the analysis files, not the key. Grade what the reviewer concluded and showed, not their writing style or length.

ANSWER KEY
{key}

REVIEWER'S ANSWER
<answer>
{answer}
</answer>

Grade these fields (use null where a field doesn't apply to this task):
- "verdict": the reviewer's final verdict on the main claim: "SUPPORTED", "NOT SUPPORTED" or "UNCLEAR".
- "detected" (flawed tasks only): true if the reviewer identified the specific mechanism in the key's "mechanism" \
as a reason the claim fails (in their own words). Generic concerns ("small sample", "needs validation", "possible \
confounding") don't count unless they name this mechanism.
- "quantified" (flawed tasks only): true if the reviewer backed the mechanism with a number they measured on the \
data that roughly matches one of the key's "quantify" items.
- "fix" (flawed tasks only): true if the reviewer proposed the key's "fix" or an equivalent remedy.
- "false_alarm" (clean tasks only): true if the reviewer asserted a serious flaw the key says is absent (see \
"false_alarms"), or rejected the claim on grounds the data doesn't support. Caveats listed under \
"acceptable_caveats", or minor suggestions, are not false alarms.
- "other_valid_issue": true if the reviewer raised a real, important problem that the key doesn't mention.
- "rationale": one or two sentences.

Return only a JSON object with exactly these keys."""


def mask(text: str) -> str:
    for pattern, repl in MASKS:
        text = pattern.sub(repl, text)
    return text


def parse_verdict(answer: str) -> str | None:
    found = re.findall(r"VERDICT:\s*\**\s*(NOT SUPPORTED|SUPPORTED)", answer, re.IGNORECASE)
    return found[-1].upper() if found else None


def key_for(rubric: dict) -> str:
    keep = {k: v for k, v in rubric.items() if k not in ("id", "book_case", "class")}
    return json.dumps(keep, indent=2)


def grade(run_json: Path, model: str) -> dict:
    out = run_json.with_name("judge.json")
    if out.exists():
        return json.loads(out.read_text())
    run = json.loads(run_json.read_text())
    rubric = json.loads((EVALS / "tasks" / run["task"] / "rubric.json").read_text())
    answer = mask(run.get("answer") or "")
    result = {"parsed_verdict": parse_verdict(run.get("answer") or ""), "expected_verdict": rubric["expected_verdict"]}
    if not answer.strip():
        result |= {"verdict": "UNCLEAR", "detected": False if rubric["flawed"] else None, "error": "empty answer"}
    else:
        proc = subprocess.run(
            ["claude", "-p", JUDGE.format(key=key_for(rubric), answer=answer), "--model", model, "--tools", "",
             "--strict-mcp-config", "--output-format", "json", "--no-session-persistence"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600, check=False,
        )  # fmt: skip
        try:
            text = json.loads(proc.stdout)["result"]
            result |= json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        except (json.JSONDecodeError, KeyError, AttributeError):
            # not saved, so a rerun retries it (usage limits, transient failures)
            return {
                "error": f"unparseable judge output: {proc.stdout[-300:]} {proc.stderr[-300:]}",
                "verdict_correct": False,
            }
    result["final_verdict"] = result["parsed_verdict"] or result.get("verdict")
    result["verdict_correct"] = result["final_verdict"] == rubric["expected_verdict"]
    result["flawed"] = rubric["flawed"]
    result["judge_model"] = model
    out.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--parallel", type=int, default=4)
    a = p.parse_args()
    runs = sorted((a.out / "runs").glob("*/arm*/rep*/run.json"))
    print(f"grading {len(runs)} runs with {a.model}")
    with ThreadPoolExecutor(a.parallel) as pool:
        for path, res in zip(runs, pool.map(lambda r: grade(r, a.model), runs)):
            tag = "/".join(path.parts[-4:-1])
            if "final_verdict" not in res:
                print(f"{tag}: NOT GRADED ({res['error'][:120]})")
                continue
            print(f"{tag}: verdict {res.get('final_verdict')} ({'ok' if res['verdict_correct'] else 'WRONG'}), "
                  f"detected={res.get('detected')} false_alarm={res.get('false_alarm')}{' ERROR' if 'error' in res else ''}")  # fmt: skip


if __name__ == "__main__":
    main()
