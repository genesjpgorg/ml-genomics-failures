# Does the book of failures make agents better reviewers?

An A/B eval: the same agent reviews small analyses that contain a planted flaw (or deliberately don't), with and
without the book, and a blind judge grades the reviews against a hidden answer key.

## Tasks (pilot: 12)

`make_tasks.py` generates them; each planted property is measured with the package's checks before a task is
written, so the answer key is verified. Domains are disguised so an agent can't match a task to a known paper.

| Task | Planted flaw | Class | Book case |
|---|---|---|---|
| T01 | Protein–protein interactions: random pair split, labels follow per-protein propensity | L | F03/F04 |
| T02 | Splice variants: labels uniform within genes, variant-level split | C/L | F09 |
| T03 | Plasma proteomics: case/control confounded with plate and run date | S | F10 |
| T04 | Promoter activity: 70% of test sequences are near-copies of training ones | L | F12 |
| T05 | Phage vs bacterial contigs: separable by GC content alone | B | F01/F08 |
| T06 | TF binding: "beats the dinucleotide-shuffled control" taken as evidence of grammar | K | F02 |
| T07 | Expression model: across-gene r used to justify personal variant predictions | A | F05 |
| T08 | Immunotherapy signature: feature selection before cross-validation | leaky preprocessing | none (tests generalization) |
| C01 | Protein-disjoint split, pair-level signal | clean | |
| C02 | Plates balanced, selection inside CV folds, real signal | clean | |
| C03 | Motif order with identical composition, controls and seeds | clean | |
| C04 | Across-gene claim limited to what was tested | clean | |

Clean controls measure false alarms: an agent that calls everything leaky would otherwise look perfect.

## Arms

| Arm | Agent gets |
|---|---|
| 0 | nothing (baseline) |
| 1 | `dist/AGENTS.md` appended to the system prompt |
| 2 | the MCP server, not mentioned |
| 3 | the MCP server, and the prompt asks to audit with it |

Every arm uses the same prompt ("is the main claim supported?", ending in `VERDICT: SUPPORTED/NOT SUPPORTED`),
model, tools and isolation (see `run.py`).

## Run

```bash
uv sync                                   # the MCP server the arms use (repo .venv)
uv venv evals/.venv && uv pip install --python evals/.venv/bin/python numpy pandas scipy scikit-learn
evals/.venv/bin/python evals/make_tasks.py
evals/.venv/bin/python evals/run.py    --out evals/results/pilot --arms 0,1,3 --reps 2
evals/.venv/bin/python evals/judge.py  --out evals/results/pilot
evals/.venv/bin/python evals/report.py --out evals/results/pilot   # -> report.md, summary.json
```

`run.py` and `judge.py` resume where they stopped. Results are not committed (`evals/results/` is ignored);
commit a `report.md` when it's worth keeping.

## Grading

`judge.py` masks case IDs, tool and server names before grading, so the judge can't tell the arms apart. It scores
the verdict, whether the planted mechanism was detected, quantified on the data and fixed, false alarms on clean
tasks, and other valid issues. The verdict is also parsed from the `VERDICT:` line. Hand-check a sample of
judgements before trusting the numbers.

`report.py` averages repetitions within each task and bootstraps over tasks (paired across arms): the task is the
unit of independence, not the run.

## Findings so far

**Baseline Sonnet 5.5 detects every single-flaw task; the MCP's edge is completeness.** Across review, write-up
and train-and-report prompts, the baseline detects every planted mechanism (`x1`–`x4`, 44 graded `devin -p`
runs): single leaks, a hidden clustering column, confounded batches. On tasks with one flaw there is no
outcome gap. Where a difference appears is consistency and completeness on harder tasks (`x5-consistency`,
36 runs): on T13 (two simultaneous flaws) the baseline found the obvious confound but treated the leaky split
as a caveat in 3 of 6 runs — full detection 50% vs 100% with the MCP, fixes 67% vs 94%. The tool's output
framing matters: `check_split_overlap` reports the entity-memorization score as a failure, where an ad-hoc
pandas check reads as a caveat.

**Costs.** ~30% more tokens per run, and one false alarm on a clean task in x3 (the "everything looks like a
pitfall" failure mode also seen when the book was text for Haiku 4.5). **Unprompted adoption is real**: ~95%
of arm-2 runs call the tools without being asked, half of them the one-call `audit_directory` — the server
instructions do change what agents do first.

## Known limits

- Isolation is not a sandbox: an agent's Python could read files outside its workspace. Runs whose tool calls touch
  this repo (rubrics, `AGENTS.md`) are flagged as contaminated in the report.
- The explicit review prompt already primes scepticism; arm 2 with an implicit prompt ("write the results
  section") is the harder, more realistic test.
