# Eval report: x3-family-devin

Model under test: `claude-sonnet-5-5-medium` · judge: `devin:claude-opus-5-5-medium` · code: `7232b45` · 8 graded runs (2 tasks × 2 arms × 2 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 100% | 100% | 100% | 100% | 0% | 0% | 0.0 | $0.00 | 17s |
| 2 MCP, unprompted | 75% | 100% | 100% | 100% | 50% | 100% | 1.0 | $0.00 | 19s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 2 − arm 0 |
|---|---|
| Verdict correct (all tasks) | -25% [-50%, +0%] |
| Detected planted flaw (flawed tasks) | – |
| Quantified it (flawed tasks) | – |
| False alarm (clean tasks) | – |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 2 |
|---|---|---|---|
| C06 | clean | 100% / FA 0% | 50% / FA 50% |
| T12 | flawed | 100% / det 100% | 100% / det 100% |

## MCP tool calls

`mcp_list_tools` 4

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
