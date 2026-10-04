"""The MCP server, over a real in-process MCP client session."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import Client

from ml_genomics_failures import server

ROOT = Path(__file__).parents[1]


def call(tool: str, args: dict | None = None):
    async def go():
        async with Client(server.mcp, raise_exceptions=True) as c:
            return await c.call_tool(tool, args or {})

    return asyncio.run(go())


def structured(result):
    return result.structured_content.get("result", result.structured_content)


def test_bundled_data_is_current():
    assert subprocess.run([sys.executable, str(ROOT / "build.py"), "--check"], cwd=ROOT, check=False).returncode == 0


def test_tools_listed():
    async def go():
        async with Client(server.mcp) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            prompts = {p.name for p in (await c.list_prompts()).prompts}
            return tools, prompts

    tools, prompts = asyncio.run(go())
    assert {"search", "fetch", "list_cases", "get_audit_checklist"} <= tools
    assert "audit_experiment" in prompts


@pytest.mark.parametrize(
    "query, expected",
    [
        ("enhancer promoter interaction", "F03"),
        ("gene regulatory network unseen genes", "F04"),
        ("token shuffle control", "F02"),
        ("perturbation linear baseline", "F06"),
        ("F09", "F09"),
    ],
)
def test_search_ranks_expected_case_first(query, expected):
    results = structured(call("search", {"query": query}))["results"]
    assert results[0]["id"] == expected
    assert results[0]["url"].endswith(f"/cases/{expected}.yaml")


def test_search_class_code_is_case_sensitive():
    k = {r["id"] for r in structured(call("search", {"query": "K"}))["results"]}
    assert k == {"F02"}


def test_fetch_and_unknown_id():
    doc = structured(call("fetch", {"id": "f05"}))
    assert doc["id"] == "F05" and "across individuals" in doc["text"] and doc["metadata"]["classes"] == ["A"]
    err = call("fetch", {"id": "F999"})
    assert err.is_error and "unknown case 'F999'" in err.content[0].text


def test_list_cases_filters():
    ids = [c["id"] for c in structured(call("list_cases", {"pitfall_class": "b"}))]
    assert ids == ["F06", "F07", "F08"]
    assert [c["id"] for c in structured(call("list_cases", {"domain": "single-cell"}))] == ["F04", "F07", "F10"]


def test_checklist_and_prompt():
    text = call("get_audit_checklist").content[0].text
    assert "Unit of independence" in text and "F12" in text

    async def go():
        async with Client(server.mcp) as c:
            return await c.get_prompt("audit_experiment", {"description": "my model predicts X"})

    msg = asyncio.run(go()).messages[0].content.text
    assert "my model predicts X" in msg and "Audit procedure" in msg


def test_bundled_json_matches_dist():
    a = json.loads((ROOT / "dist/failures.json").read_text())
    b = json.loads((ROOT / "src/ml_genomics_failures/data/failures.json").read_text())
    assert a == b
