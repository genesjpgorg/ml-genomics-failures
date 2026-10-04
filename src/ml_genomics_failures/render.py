"""Markdown rendering of the checklist and cases, shared by build.py and the MCP server.

Works on plain dicts (the parsed YAML, or dist/failures.json), so it needs no YAML library at runtime.
"""

from __future__ import annotations


def classes_table(checklist: dict) -> str:
    rows = [f"| **{c['code']}** | {c['name']} | {c['test']} |" for c in checklist["classes"]]
    return "\n".join(["| Code | Class | One-line test |", "|---|---|---|", *rows])


def audit_section(checklist: dict, level: int) -> str:
    h = "#" * level
    steps = [
        f"{i}. **{s['title']}{'' if s['title'][-1] in '.?!' else '.'}** {s['text']}"
        for i, s in enumerate(checklist["audit"]["steps"], 1)
    ]
    usage = [f"- {u}" for u in checklist["usage"]]
    return "\n".join(
        [
            f"{h} Pitfall classes",
            "",
            classes_table(checklist),
            "",
            f"{h} Audit procedure",
            "",
            checklist["audit"]["intro"],
            "",
            *steps,
            "",
            f"{h} How to use the cases",
            "",
            *usage,
        ]
    )


def index_table(cases: list[dict]) -> str:
    rows = [f"| {c['id']} | {c['title']} | {', '.join(c['classes'])} | {c['status']['state']} |" for c in cases]
    return "\n".join(["| ID | Case | Class | Status |", "|---|---|---|---|", *rows])


def status_text(s: dict) -> str:
    if s["state"] == "internal":
        return f"internal ({s['date']}). Pending: {s['pending']}"
    return s["state"]


def render_case(c: dict, level: int) -> str:
    sources = [f"  - {s['citation']}" + (f" {s['url']}" if s.get("url") else "") for s in c["sources"]]
    flags = [f"  - {f}" for f in c["red_flags"]]
    return "\n".join(
        [
            f"{'#' * level} {c['id']}. {c['title']}",
            "",
            f"- **Domain:** {c['domain']}",
            f"- **Class:** {', '.join(c['classes'])}",
            f"- **Claim:** {c['claim']}",
            f"- **What went wrong:** {c['mechanism']}",
            "- **Red flags:**",
            *flags,
            f"- **Test that exposes it:** {c['test']}",
            f"- **Fix:** {c['fix']}",
            "- **Sources:**",
            *sources,
            f"- **Status:** {status_text(c['status'])}",
        ]
    )
