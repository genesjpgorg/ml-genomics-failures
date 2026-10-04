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
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__
from .render import audit_section, index_table, render_case

REPO_URL = "https://github.com/genesjpgorg/ml-genomics-failures"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
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
        "case doesn't contain."
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


def main() -> None:
    p = argparse.ArgumentParser(prog="ml-genomics-failures-mcp", description=__doc__.split("\n")[0])
    p.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--stateless", action="store_true", help="stateless HTTP (for serverless/multi-replica hosting)")
    a = p.parse_args()
    if a.http:
        mcp.run("streamable-http", host=a.host, port=a.port, stateless_http=a.stateless)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
