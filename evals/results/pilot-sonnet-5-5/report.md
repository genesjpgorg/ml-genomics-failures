# Eval report: pilot-sonnet-5-5

Model under test: `claude-sonnet-5-5` · judge: `claude-opus-5-5` · code: `14633e2` · 72 graded runs (12 tasks × 3 arms × 2 reps)

## By arm

| Arm | Verdict correct | Flawed: detected | Flawed: quantified | Flawed: fix | Clean: false alarm | Used MCP | MCP calls/run | Cost/run | Time/run |
|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 100% | 100% | 100% | 100% | 0% | 0% | 0.0 | $0.06 | 30s |
| 1 book as text | 96% | 100% | 100% | 100% | 12% | 0% | 0.0 | $0.06 | 27s |
| 3 MCP, prompted | 100% | 100% | 100% | 100% | 0% | 100% | 2.9 | $0.08 | 31s |

## Difference from baseline (paired over tasks, 95% bootstrap CI)

| Metric | arm 1 − arm 0 | arm 3 − arm 0 |
|---|---|---|
| Verdict correct (all tasks) | -4% [-12%, +0%] | +0% [+0%, +0%] |
| Detected planted flaw (flawed tasks) | +0% [+0%, +0%] | +0% [+0%, +0%] |
| Quantified it (flawed tasks) | +0% [+0%, +0%] | +0% [+0%, +0%] |
| False alarm (clean tasks) | +12% [+0%, +38%] | +0% [+0%, +0%] |

## By task

Verdict correct / detected (flawed) or false alarm (clean), averaged over reps.

| Task | Expected | arm 0 | arm 1 | arm 3 |
|---|---|---|---|---|
| C01 | clean | 100% / FA 0% | 100% / FA 0% | 100% / FA 0% |
| C02 | clean | 100% / FA 0% | 100% / FA 0% | 100% / FA 0% |
| C03 | clean | 100% / FA 0% | 100% / FA 0% | 100% / FA 0% |
| C04 | clean | 100% / FA 0% | 50% / FA 50% | 100% / FA 0% |
| T01 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T02 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T03 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T04 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T05 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T06 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T07 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |
| T08 | flawed | 100% / det 100% | 100% / det 100% | 100% / det 100% |

## MCP tool calls

`get_audit_checklist` 24, `fetch` 13, `check_split_overlap` 8, `check_sequence_similarity` 8, `kmer_baseline` 8, `check_shortcuts` 7, `search` 2

Small-sample caution: with a dozen tasks, one task moves a rate by ~8 points; the CIs above reflect that.
