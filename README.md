# ml-genomics-failures

A **book of failures** for machine learning in genomics: documented cases where a model looked good but had
learned a shortcut, leaked test data, or was never compared to a simple baseline. Each case gives the red flags in
the numbers, the test that exposes the problem and the fix. An eight-question audit procedure ties them together.

It's written for AI agents (Claude, Codex, Cursor, Devin, ChatGPT and others) to run before trusting a metric or
claiming a model "learned biology", and it reads fine for people too.

| Code | Pitfall class |
|---|---|
| L | Leakage through shared entities (genes, elements, species, individuals, homologs) |
| S | Shortcut / confounder (distance, GC, gene degree, phylogeny, batch, ancestry) |
| A | Wrong axis of generalization (across genes vs across individuals, lineage vs phenotype) |
| B | Missing or weak baseline |
| C | Circular evaluation |
| K | Control that isn't a null |

The current cases are listed in [`skills/ml-genomics-failures/FAILURES.md`](skills/ml-genomics-failures/FAILURES.md).

## Use it in an agent

All outputs are generated from the same YAML, so every agent sees the same content.

| Agent | What to use |
|---|---|
| Claude Code | Copy or symlink the skill: `ln -s "$PWD/skills/ml-genomics-failures" ~/.claude/skills/` (all projects) or into a project's `.claude/skills/` |
| Claude apps, and other agents that support [Agent Skills](https://agentskills.io) (e.g. Codex, Cursor, GitHub Copilot) | Install the [`skills/ml-genomics-failures/`](skills/ml-genomics-failures/) folder as a skill, per that agent's docs (for Claude apps, upload it as a zip) |
| Devin | Add [`dist/AGENTS.md`](dist/AGENTS.md) as Knowledge, or commit it to the repo as `AGENTS.md` |
| ChatGPT | Upload [`dist/AGENTS.md`](dist/AGENTS.md) to a Project or custom GPT as knowledge |
| Anything else | Put [`dist/AGENTS.md`](dist/AGENTS.md) in the prompt, or read [`dist/failures.json`](dist/failures.json) |

An MCP server (search and fetch cases, the audit as a prompt, and executable checks such as split-overlap and
k-mer baselines) is planned and will read `dist/failures.json`.

## Layout

```
cases/F01.yaml …          one case per file: the source of truth
checklist.yaml            pitfall classes, audit procedure, skill description
build.py                  validates the YAML and generates everything below
skills/ml-genomics-failures/SKILL.md     generated: Agent Skill (audit + case index)
skills/ml-genomics-failures/FAILURES.md  generated: full cases
dist/AGENTS.md            generated: everything in one file
dist/failures.json        generated: machine-readable
```

## Add a case

1. Copy a case to `cases/F<next>.yaml` and fill in every field: `id`, `title`, `domain`, `classes`, `claim`,
   `mechanism`, `red_flags`, `test`, `fix`, `sources` (each with `citation`, optional `url`), `status`
   (`state: published`, or `state: internal` with `date` and `pending`).
2. Run `uv run build.py` and commit the YAML together with the regenerated files. CI runs
   `uv run build.py --check` and fails if they're out of date.

Only write claims the cited source supports, and link the source. Mark unpublished cases `internal` and update
them when the deciding experiment has run.
