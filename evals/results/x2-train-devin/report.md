# Eval report: x2-train-devin

Model under test: `claude-sonnet-5-5-medium` · judge: `devin:claude-opus-5-5-medium` · code: `7232b45` · 16 graded runs (4 tasks × 2 arms × 2 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 88% | 100% | 100% | 83% | 0% | 0% | 0.0 | $0.00 | 20s |
| 2 MCP, unprompted | 100% | 100% | 100% | 100% | 0% | 100% | 1.4 | $0.00 | 24s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 2 − arm 0 |
|---|---|
| Verdict correct (all tasks) | +12% [+0%, +38%] |
| Detected planted flaw (flawed tasks) | +0% [+0%, +0%] |
| Quantified it (flawed tasks) | +0% [+0%, +0%] |
| False alarm (clean tasks) | – |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 2 |
|---|---|---|---|
| C05 | clean | 100% / FA 0% | 100% / FA 0% |
| T09 | flawed | 50% / det 100% | 100% / det 100% |
| T10 | flawed | 100% / det 100% | 100% / det 100% |
| T11 | flawed | 100% / det 100% | 100% / det 100% |

## MCP tool calls

`mcp_list_tools` 8, `check_split_overlap` 2, `audit_directory` 1

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
