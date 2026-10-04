# Eval report: x1-writeup-devin

Model under test: `claude-sonnet-5-5-medium` · judge: `devin:claude-opus-5-5-medium` · code: `7232b45` · 20 graded runs (5 tasks × 2 arms × 2 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 100% | 100% | 100% | 100% | 0% | 0% | 0.0 | $0.00 | 24s |
| 2 MCP, unprompted | 100% | 100% | 100% | 100% | 0% | 90% | 1.3 | $0.00 | 27s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 2 − arm 0 |
|---|---|
| Verdict correct (all tasks) | +0% [+0%, +0%] |
| Detected planted flaw (flawed tasks) | +0% [+0%, +0%] |
| Quantified it (flawed tasks) | +0% [+0%, +0%] |
| False alarm (clean tasks) | – |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 2 |
|---|---|---|---|
| C01 | clean | 100% / FA 0% | 100% / FA 0% |
| T01 | flawed | 100% / det 100% | 100% / det 100% |
| T03 | flawed | 100% / det 100% | 100% / det 100% |
| T04 | flawed | 100% / det 100% | 100% / det 100% |
| T05 | flawed | 100% / det 100% | 100% / det 100% |

## MCP tool calls

`mcp_list_tools` 9, `audit_directory` 4

## Flagged runs

- C01 arm2 rep0: contaminated
- C01 arm2 rep1: contaminated
- T04 arm2 rep0: contaminated
- T05 arm2 rep0: contaminated
- T05 arm2 rep1: contaminated

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
