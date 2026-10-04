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

## MCP server

Any MCP client can use the book through tools instead of reading files:

| Tool / prompt | What it does |
|---|---|
| `search(query)` | Cases by free text, case ID or class code, best first |
| `fetch(id)` | One full case as markdown, plus metadata |
| `list_cases(pitfall_class?, domain?)` | Filtered case summaries |
| `get_audit_checklist()` | The 8-question audit, pitfall classes and case index |
| prompt `audit_experiment(description)` | The audit applied to your experiment |
| resource `failures://book` | The whole book as one markdown document |
| `check_split_overlap(columns, train, test, label?)` | Entities (genes, enhancers, species, individuals) shared by train and test, and the score of memorizing each entity's label (F03, F04, F09) |
| `check_sequence_similarity(train_fasta, test_fasta, test_scores?)` | Each test sequence's nearest training sequence by k-mer containment, and performance by identity to training (F12) |
| `check_shortcuts(table, label, features, group_column?)` | Cross-validated score of each candidate confounder (batch, GC, distance, ancestry...) predicting the label alone (F10, F11, F01) |
| `kmer_baseline(train, test)` | GC-only and k-mer composition baselines for sequence → label (F01, F02, F08) |

The `check_*` tools and `kmer_baseline` measure pitfalls on your own data. Pass `model_score` and each result
reports what share of your model's gain over chance the trivial baseline already reaches. They take CSV/TSV or
FASTA inline; a local (stdio) server also accepts file paths, while a hosted one never reads files. They use only
numpy: about 5 s for 12 Mbp of sequence. For proteins or alignment-level identity, use MMseqs2.

`search` and `fetch` use the result shape ChatGPT connectors expect, so one server works everywhere.

**Local clients (stdio).** Needs [uv](https://docs.astral.sh/uv/).

```bash
# Claude Code
claude mcp add ml-genomics-failures -- uvx --from git+https://github.com/genesjpgorg/ml-genomics-failures ml-genomics-failures-mcp
```

Claude Desktop, Cursor and most other clients take the same command in their JSON config:

```json
{
  "mcpServers": {
    "ml-genomics-failures": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/genesjpgorg/ml-genomics-failures", "ml-genomics-failures-mcp"]
    }
  }
}
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.ml-genomics-failures]
command = "uvx"
args = ["--from", "git+https://github.com/genesjpgorg/ml-genomics-failures", "ml-genomics-failures-mcp"]
```

**Web agents (ChatGPT, Devin, claude.ai).** These need a public HTTPS endpoint. Run the server with
`ml-genomics-failures-mcp --http --host 0.0.0.0 --port 8000` (add `--stateless` for serverless or multi-replica
hosting) behind HTTPS, and add `https://<host>/mcp` as a connector. A hosted endpoint isn't set up yet.


## Layout

```
cases/F01.yaml …          one case per file: the source of truth
checklist.yaml            pitfall classes, audit procedure, skill description
build.py                  validates the YAML and generates everything below
skills/ml-genomics-failures/SKILL.md     generated: Agent Skill (audit + case index)
skills/ml-genomics-failures/FAILURES.md  generated: full cases
dist/AGENTS.md            generated: everything in one file
dist/failures.json        generated: machine-readable
src/ml_genomics_failures/ MCP server package (render.py is shared with build.py; data/ is generated)
tests/                    MCP server tests (`uv run pytest`)
```

## Add a case

1. Copy a case to `cases/F<next>.yaml` and fill in every field: `id`, `title`, `domain`, `classes`, `claim`,
   `mechanism`, `red_flags`, `test`, `fix`, `sources` (each with `citation`, optional `url`), `status`
   (`state: published`, or `state: internal` with `date` and `pending`).
2. Run `uv run build.py` and commit the YAML together with the regenerated files. CI runs
   `uv run build.py --check` (fails if they're out of date), `ruff` and the tests.

Only write claims the cited source supports, and link the source. Mark unpublished cases `internal` and update
them when the deciding experiment has run.
