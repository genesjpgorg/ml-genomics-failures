# Eval report: x5-consistency-devin

Model under test: `claude-sonnet-5-5-medium` · judge: `devin:claude-opus-5-5-medium` · code: `622d896` · 36 graded runs (3 tasks × 2 arms × 6 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 94% | 83% | 94% | 67% | – | 0% | 0.0 | $0.00 | 21s |
| 2 MCP, unprompted | 100% | 100% | 100% | 94% | – | 100% | 1.2 | $0.00 | 21s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 2 − arm 0 |
|---|---|
| Verdict correct (all tasks) | +6% [+0%, +17%] |
| Detected planted flaw (flawed tasks) | +17% [+0%, +50%] |
| Quantified it (flawed tasks) | +6% [+0%, +17%] |
| False alarm (clean tasks) | – |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 2 |
|---|---|---|---|
| T09 | flawed | 100% / det 100% | 100% / det 100% |
| T11 | flawed | 100% / det 100% | 100% / det 100% |
| T13 | flawed | 83% / det 50% | 100% / det 100% |

## MCP tool calls

`mcp_list_tools` 18, `check_split_overlap` 4

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
