"""MCP server for the book of failures: search and fetch cases, the audit checklist, and an audit prompt.

usage: ml-genomics-failures-mcp                         # stdio (Claude Code, Cursor, Codex CLI, ...)
       ml-genomics-failures-mcp --http --port 8000      # Streamable HTTP at http://HOST:PORT/mcp

`search` and `fetch` follow the shape ChatGPT connectors expect ({"results": [{id, title, url}]} and
{id, title, text, url, metadata}), so the same server works there and in every other MCP client.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import cache
from importlib.resources import files
from typing import Any, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import __version__, checks
from .render import audit_section, index_table, render_case

REPO_URL = "https://github.com/genesjpgorg/ml-genomics-failures"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
# File-path inputs are read only when the server runs locally (stdio). A hosted server takes data inline, so a
# caller can never make it read its own filesystem.
ALLOW_PATHS = False
# search weights per field: a hit in the title counts more than one buried in the mechanism
WEIGHTS = {"title": 3, "domain": 2, "claim": 1, "mechanism": 1, "red_flags": 1, "test": 1, "fix": 1}


class SearchHit(TypedDict):
    id: str
    title: str
    url: str


class SearchResults(TypedDict):
    results: list[SearchHit]


class Case(TypedDict):
    id: str
    title: str
    text: str
    url: str
    metadata: dict


class CaseSummary(TypedDict):
    id: str
    title: str
    classes: list[str]
    domain: str
    status: str
    url: str


@cache
def book() -> dict:
    return json.loads(files(__package__).joinpath("data/failures.json").read_text(encoding="utf-8"))


def cases() -> dict[str, dict]:
    return {c["id"]: c for c in book()["cases"]}


def case_url(case_id: str) -> str:
    return f"{REPO_URL}/blob/main/cases/{case_id}.yaml"


def summary(c: dict) -> CaseSummary:
    return {
        "id": c["id"],
        "title": c["title"],
        "classes": c["classes"],
        "domain": c["domain"],
        "status": c["status"]["state"],
        "url": case_url(c["id"]),
    }


def _words(text: str) -> list[str]:
    """Lower-case words of 2+ characters (single letters like the "k" of "k-mer" would match class codes)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


def rank(query: str) -> list[dict]:
    """Cases scored by weighted word overlap with the query; class codes (L, S, A, B, C, K) and case IDs match
    exactly. An empty query returns every case in ID order."""
    if not query.strip():
        return list(cases().values())
    codes = {c["code"] for c in book()["checklist"]["classes"]}
    names = {c["code"]: _words(c["name"]) for c in book()["checklist"]["classes"]}
    q_words = set(_words(query))
    q_codes = {t for t in query.split() if t in codes}  # case-sensitive: "a" or "c" in prose isn't a class
    scored = []
    for c in cases().values():
        score = 10 * (c["id"].lower() in q_words) + 3 * len(q_codes & set(c["classes"]))
        for field, w in WEIGHTS.items():
            value = " ".join(c[field]) if isinstance(c[field], list) else c[field]
            score += w * len(q_words & set(_words(value)))
        score += 2 * sum(len(q_words & set(names[k])) for k in c["classes"])
        if score:
            scored.append((score, c["id"], c))
    return [c for _, _, c in sorted(scored, key=lambda t: (-t[0], t[1]))]


def checklist_markdown() -> str:
    b = book()
    return "\n\n".join(
        [
            f"# {b['checklist']['title']}",
            b["checklist"]["summary"],
            audit_section(b["checklist"], 2),
            "## Cases",
            index_table(b["cases"]),
            "Fetch a case by ID before citing it.",
        ]
    )


mcp = MCPServer(
    name="ml-genomics-failures",
    title="Book of failures: ML in genomics",
    description="Documented pitfalls in ML for genomics, with red flags and the tests that expose them.",
    instructions=(
        "Use this server before trusting a metric or claiming that a genomics model learned biology. Start with "
        "get_audit_checklist (or the audit_experiment prompt) and answer every question for the experiment at hand. "
        "Use search or list_cases to find matching cases, and fetch a case before citing it by ID (e.g. F03). Treat "
        "cases as hypotheses to test on the user's run, not verdicts, and don't add claims about a paper that the "
        "case doesn't contain. When the user has data (splits, sequences, a feature table, sequences with labels), "
        "run the check_* tools and kmer_baseline to measure the pitfalls instead of only describing them."
    ),
    website_url=REPO_URL,
    version=__version__,
)


@mcp.tool(annotations=READ_ONLY)
def search(query: str) -> SearchResults:
    """Search the book of failures (ML pitfalls in genomics) by free text, case ID or class code (L leakage,
    S shortcut/confounder, A wrong axis of generalization, B missing baseline, C circular evaluation, K control
    that isn't a null). Returns matching case IDs, titles and URLs, best first; fetch an ID for the full case."""
    return {"results": [{"id": c["id"], "title": c["title"], "url": case_url(c["id"])} for c in rank(query)]}


@mcp.tool(annotations=READ_ONLY)
def fetch(id: str) -> Case:  # "id", not "case_id": the name ChatGPT connectors expect
    """Full case by ID (e.g. "F03"): domain, pitfall classes, the claim, what went wrong, red flags, the test that
    exposes it, the fix, sources and status."""
    c = cases().get(id.strip().upper())
    if c is None:
        raise ToolError(f"unknown case {id!r}; known: {', '.join(cases())}")
    return {
        "id": c["id"],
        "title": c["title"],
        "text": render_case(c, 2),
        "url": case_url(c["id"]),
        "metadata": {"classes": c["classes"], "domain": c["domain"], "status": c["status"]},
    }


@mcp.tool(annotations=READ_ONLY)
def list_cases(pitfall_class: str | None = None, domain: str | None = None) -> list[CaseSummary]:
    """List cases, optionally filtered by pitfall class code (L, S, A, B, C, K) and/or a word in the domain
    (e.g. "single-cell", "variant", "regulatory")."""
    out = list(cases().values())
    if pitfall_class:
        out = [c for c in out if pitfall_class.strip().upper() in c["classes"]]
    if domain:
        out = [c for c in out if domain.strip().lower() in c["domain"].lower()]
    return [summary(c) for c in out]


@mcp.tool(annotations=READ_ONLY)
def get_audit_checklist() -> str:
    """The audit procedure (8 questions), the pitfall classes and the case index, as markdown. Run it on any ML
    experiment, benchmark or claim in genomics before trusting the reported metric."""
    return checklist_markdown()


# --- executable checks --------------------------------------------------------------------------------------


def _load(text: str | None, path: str | None, name: str) -> str:
    if (text is None) == (path is None):
        raise ToolError(f"give exactly one of {name} (inline) or {name}_path")
    if path is not None:
        if not ALLOW_PATHS:
            raise ToolError("file paths are only read by a local (stdio) server; pass the data inline instead")
        return checks.read_text(path)
    return text


def _run(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except checks.CheckError as e:
        raise ToolError(str(e)) from e


@mcp.tool(annotations=READ_ONLY)
def check_split_overlap(
    columns: list[str],
    train: str | None = None,
    test: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    label: str | None = None,
    task: str = "auto",
    model_score: float | None = None,
) -> dict[str, Any]:
    """Leakage check (class L; cases F03, F04, F09): how many test rows share an entity (gene, enhancer, promoter,
    species, individual...) with training, per entity column. With `label`, also scores an entity-memorization
    baseline: predict each test row from the training labels of its own entities, with no features. Compare it
    with the model's score (`model_score`, same metric: AUROC for binary labels, accuracy for multiclass, Spearman
    for numeric); if it gets close, the model may be memorizing per-entity label rates.

    Give each input inline (CSV/TSV or FASTA text) or, when the server runs locally, as a file path (.gz is fine).
    Returns a plain-language summary that names the matching cases, plus the numbers."""
    tr = checks.parse_csv(_load(train, train_path, "train"), "train")
    te = checks.parse_csv(_load(test, test_path, "test"), "test")
    return _run(checks.split_overlap, tr, te, columns, label, task, model_score)


@mcp.tool(annotations=READ_ONLY)
def check_sequence_similarity(
    train_fasta: str | None = None,
    test_fasta: str | None = None,
    train_fasta_path: str | None = None,
    test_fasta_path: str | None = None,
    k: int = 15,
    scale: int | None = None,
    test_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Homology leakage check (class L; case F12) for DNA/RNA: for every test sequence, its nearest training
    sequence by k-mer containment, with an estimated identity, binned (>=0.99, 0.95-0.99, ...). Pass `test_scores`
    ({test id: per-example score or 0/1 correct}) to see the model's performance by identity to training. Large
    inputs are subsampled automatically (FracMinHash `scale`). For proteins or alignment-level identity use
    MMseqs2.

    Give each input inline (CSV/TSV or FASTA text) or, when the server runs locally, as a file path (.gz is fine).
    Returns a plain-language summary that names the matching cases, plus the numbers."""
    tr = checks.parse_fasta(_load(train_fasta, train_fasta_path, "train_fasta"), "train_fasta")
    te = checks.parse_fasta(_load(test_fasta, test_fasta_path, "test_fasta"), "test_fasta")
    return _run(checks.sequence_similarity, tr, te, k, scale, test_scores)


@mcp.tool(annotations=READ_ONLY)
def check_shortcuts(
    label: str,
    features: list[str],
    table: str | None = None,
    table_path: str | None = None,
    task: str = "auto",
    group_column: str | None = None,
    n_folds: int = 5,
    model_score: float | None = None,
) -> dict[str, Any]:
    """Shortcut / confounder check (class S; cases F10, F11, F01): cross-validated score of predicting `label`
    from each candidate feature alone (batch, plate, GC, distance, sequencing depth, ancestry, family, gene...).
    One row per example. Categorical features predict from same-category training rows; numeric ones are binned.
    `group_column` keeps groups (e.g. gene, individual) within one fold, like an entity-level split. With
    `model_score` (same metric: AUROC binary, accuracy multiclass, Spearman numeric), each feature's share of the
    model's gain over chance is reported.

    Give each input inline (CSV/TSV or FASTA text) or, when the server runs locally, as a file path (.gz is fine).
    Returns a plain-language summary that names the matching cases, plus the numbers."""
    rows = checks.parse_csv(_load(table, table_path, "table"), "table")
    return _run(checks.shortcuts, rows, label, features, task, group_column, n_folds, model_score)


@mcp.tool(annotations=READ_ONLY)
def kmer_baseline(
    train: str | None = None,
    test: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    sequence_column: str = "sequence",
    label_column: str = "label",
    id_column: str = "id",
    k: int = 5,
    task: str = "auto",
    model_score: float | None = None,
) -> dict[str, Any]:
    """Composition baseline (class B; cases F01, F02, F08) for any sequence -> label model: predicts each test
    label from the nearest training sequences by GC content alone and by k-mer frequencies. Inputs are CSV/TSV
    with id, label and sequence columns. If composition reaches most of the model's score (`model_score`, same
    metric), the model adds little beyond composition, and composition-preserving controls such as token
    shuffling are not nulls.

    Give each input inline (CSV/TSV or FASTA text) or, when the server runs locally, as a file path (.gz is fine).
    Returns a plain-language summary that names the matching cases, plus the numbers."""

    def triples(text: str, name: str) -> list[tuple[str, str, str]]:
        rows = checks.parse_csv(text, name)
        need = [label_column, sequence_column]
        try:
            checks.require_columns(rows, need, name)
        except checks.CheckError as e:
            raise ToolError(str(e)) from e
        return [(r.get(id_column, str(i)), r[label_column], r[sequence_column]) for i, r in enumerate(rows)]

    tr = triples(_load(train, train_path, "train"), "train")
    te = triples(_load(test, test_path, "test"), "test")
    return _run(checks.kmer_baseline, tr, te, k, task, model_score)


@mcp.prompt(title="Audit an ML-in-genomics experiment")
def audit_experiment(description: str) -> str:
    """Audit an experiment, benchmark or paper claim against the book of failures."""
    return (
        f"{checklist_markdown()}\n\n---\n\nAudit this experiment with the procedure above:\n\n{description}\n\n"
        "Answer each of the 8 questions with numbers from the experiment where available, and say what is unknown. "
        "Name the cases that match (fetch them first) and, for each, the check that would confirm or rule it out. "
        "End with the claim the evidence supports today and the single most informative next experiment."
    )


@mcp.resource(
    "failures://book",
    title="Book of failures (full)",
    mime_type="text/markdown",
    description="Checklist and all cases.",
)
def full_book() -> str:
    return checklist_markdown() + "\n\n" + "\n\n".join(render_case(c, 2) for c in book()["cases"])


def http_app(allowed_hosts: list[str] | None = None, public: bool = False):
    """Stateless Streamable HTTP ASGI app at /mcp, for hosting (Modal, uvicorn, any ASGI server).

    ``public=True`` turns off DNS-rebinding protection: it guards servers on localhost or private networks, and a
    public, read-only, unauthenticated endpoint that never reads files gains nothing from it. Otherwise only
    ``allowed_hosts`` (default: localhost) may be used in the Host header."""
    if public:
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    elif allowed_hosts:
        security = TransportSecuritySettings(
            allowed_hosts=allowed_hosts, allowed_origins=[f"https://{h}" for h in allowed_hosts]
        )
    else:
        security = None
    return mcp.streamable_http_app(stateless_http=True, json_response=True, transport_security=security)


def main() -> None:
    p = argparse.ArgumentParser(prog="ml-genomics-failures-mcp", description=__doc__.split("\n")[0])
    p.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--allowed-host", action="append", help="Host header to accept (repeatable), e.g. mcp.example.org")
    p.add_argument("--public", action="store_true", help="accept any Host header (public, read-only deployments)")
    a = p.parse_args()
    global ALLOW_PATHS
    ALLOW_PATHS = not a.http
    if a.http:
        import uvicorn

        uvicorn.run(http_app(a.allowed_host, a.public), host=a.host, port=a.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
