# Eval report: smoke

Model under test: `claude-sonnet-5-5` · judge: `claude-opus-5-5` · code: `1bd90ff` · 6 graded runs (2 tasks × 3 arms × 1 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 100% | 100% | 100% | 100% | 0% | 0% | 0.0 | $0.06 | 28s |
| 1 book as text | 100% | 100% | 100% | 100% | 0% | 0% | 0.0 | $0.10 | 26s |
| 3 MCP, prompted | 100% | 100% | 100% | 100% | 0% | 100% | 2.5 | $0.11 | 35s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 1 − arm 0 | arm 3 − arm 0 |
|---|---|---|
| Verdict correct (all tasks) | +0% [+0%, +0%] | +0% [+0%, +0%] |
| Detected planted flaw (flawed tasks) | – | – |
| Quantified it (flawed tasks) | – | – |
| False alarm (clean tasks) | – | – |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 1 | arm 3 |
|---|---|---|---|---|
| C02 | clean | 100% / FA 0% | 100% / FA 0% | 100% / FA 0% |
| T01 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |

## MCP tool calls

`get_audit_checklist` 2, `fetch` 2, `check_split_overlap` 1

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
