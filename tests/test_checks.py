"""Executable checks on synthetic data with a planted answer: each must find what was planted, and nothing more."""

import asyncio
import random

import numpy as np
import pytest
from mcp import Client

from ml_genomics_failures import checks, server


def rand_seq(rng: random.Random, n: int, gc: float = 0.5) -> str:
    return "".join(rng.choices("GCAT", weights=[gc / 2, gc / 2, (1 - gc) / 2, (1 - gc) / 2], k=n))


def mutate(rng: random.Random, s: str, rate: float) -> str:
    return "".join(rng.choice("ACGT".replace(c, "")) if rng.random() < rate else c for c in s)


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def to_csv(rows: list[dict]) -> str:
    cols = list(rows[0])
    return "\n".join([",".join(cols), *(",".join(str(r[c]) for c in cols) for r in rows)])


# --- split overlap -------------------------------------------------------------------------------------------


def ep_pairs(seed: int = 0) -> list[dict]:
    """Enhancer-promoter pairs whose label depends only on how 'interactive' each element is: no pair-level
    signal, so the only way to score above chance is to memorize elements."""
    rng = random.Random(seed)
    enh = {f"e{i}": rng.random() for i in range(40)}
    pro = {f"p{i}": rng.random() for i in range(40)}
    rows = []
    for e, pe in enh.items():
        for p, pp in rng.sample(sorted(pro.items()), 10):
            rows.append({"enhancer": e, "promoter": p, "label": int(rng.random() < pe * pp * 2)})
    return rows


def test_split_overlap_random_split_leaks_and_memorization_scores():
    rows = ep_pairs()
    random.Random(1).shuffle(rows)
    train, test = rows[:300], rows[300:]
    out = checks.split_overlap(train, test, ["enhancer", "promoter"], label="label", model_score=0.80)
    assert out["test_rows_with_any_seen_entity"] == 1.0
    base = out["entity_memorization_baseline"]
    assert base["task"] == "binary" and base["score"] > 0.7
    assert base["share_of_model_gain"] > 0.6
    assert "F03" in out["summary"]


def test_split_overlap_entity_disjoint_split_is_clean():
    rows = ep_pairs()
    held_e = {f"e{i}" for i in range(30, 40)}
    held_p = {f"p{i}" for i in range(30, 40)}
    train = [r for r in rows if r["enhancer"] not in held_e and r["promoter"] not in held_p]
    test = [r for r in rows if r["enhancer"] in held_e and r["promoter"] in held_p]
    out = checks.split_overlap(train, test, ["enhancer", "promoter"], label="label")
    assert out["test_rows_with_any_seen_entity"] == 0.0
    assert out["entity_memorization_baseline"]["score"] == 0.5  # every test row falls back to the global rate
    assert "entity-disjoint" in out["summary"]


def test_split_overlap_missing_column():
    with pytest.raises(checks.CheckError, match="no column"):
        checks.split_overlap([{"a": "1"}], [{"a": "2"}], ["gene"])


# --- sequence similarity -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seqs():
    rng = random.Random(0)
    train = {f"t{i}": rand_seq(rng, 2000) for i in range(30)}
    test = {f"copy{i}": mutate(rng, train[f"t{i}"], 0.01) for i in range(5)}
    test |= {"rc": revcomp(train["t7"])}
    test |= {f"new{i}": rand_seq(rng, 2000) for i in range(6)}
    return train, test


def test_sequence_similarity_finds_near_copies(seqs):
    train, test = seqs
    out = checks.sequence_similarity(train, test)
    by_id = {p["id"]: p for p in out["most_similar"]}
    for i in range(5):
        assert by_id[f"copy{i}"]["nearest_train"] == f"t{i}"
        assert 0.97 < by_id[f"copy{i}"]["identity"] <= 1.0  # 1% mutations -> ~0.99
    assert by_id["rc"]["nearest_train"] == "t7" and by_id["rc"]["identity"] == 1.0  # strand-independent
    bins = {b["identity"]: b["n_test"] for b in out["by_identity"]}
    assert bins[">=0.99"] + bins["0.95-0.99"] == 6
    assert bins["<0.80 or unrelated"] == 6
    assert "6/12" in out["summary"] and "F12" in out["summary"]


def test_sequence_similarity_scores_by_identity_and_subsampling(seqs):
    train, test = seqs
    scores = {t: (1.0 if not t.startswith("new") else 0.0) for t in test}
    out = checks.sequence_similarity(train, test, test_scores=scores, scale=4)
    assert out["scale"] == 4
    assert out["score_vs_identity_spearman"] > 0.8
    assert out["by_identity"][-1]["mean_test_score"] == 0.0


def test_parse_fasta_rejects_garbage():
    with pytest.raises(checks.CheckError, match="before the first"):
        checks.parse_fasta("ACGT\n>x\nACGT", "test")


# --- shortcuts -----------------------------------------------------------------------------------------------


def batch_table(seed: int = 0, n: int = 400) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = int(rng.random() < 0.5)
        batch = f"b{y}" if rng.random() < 0.9 else f"b{1 - y}"  # processing batch confounded with the label
        rows.append({"label": y, "batch": batch, "noise": rng.normal(), "gene": f"g{i % 40}"})
    return rows


def test_shortcuts_flags_confounded_batch():
    out = checks.shortcuts(batch_table(), "label", ["noise", "batch"], model_score=0.92)
    top = out["features"][0]
    assert top["feature"] == "batch" and top["score"] > 0.85 and top["share_of_model_gain"] > 0.8
    noise = next(f for f in out["features"] if f["feature"] == "noise")
    assert noise["kind"] == "numeric" and abs(noise["score"] - 0.5) < 0.1
    assert "F10" in out["summary"]


def test_shortcuts_entity_memorization_disappears_with_grouped_folds():
    rng = np.random.default_rng(1)
    rate = {f"g{i}": rng.random() for i in range(40)}  # each gene has its own label rate (cf. F04, F09)
    rows = [{"label": int(rng.random() < rate[f"g{i % 40}"]), "gene": f"g{i % 40}"} for i in range(800)]
    random_folds = checks.shortcuts(rows, "label", ["gene"])
    grouped = checks.shortcuts(rows, "label", ["gene"], group_column="gene")
    assert random_folds["features"][0]["score"] > 0.7
    assert grouped["features"][0]["score"] == 0.5 and grouped["features"][0]["test_rows_with_seen_category"] == 0


def test_shortcuts_balanced_category_reports_no_shortcut():
    rows = [{"label": i % 2, "plate": f"p{(i // 2) % 4}"} for i in range(120)]  # every plate exactly 50/50
    out = checks.shortcuts(rows, "label", ["plate"])
    assert out["features"][0]["score"] < 0.5 and "note" in out["features"][0]
    assert out["summary"].startswith("No single feature predicts the label beyond chance")


def test_shortcuts_regression_and_multiclass():
    rng = np.random.default_rng(2)
    gc = rng.random(300)
    reg = [{"y": f"{g * 10 + rng.normal(0, 0.5):.3f}", "gc": f"{g:.4f}"} for g in gc]
    assert checks.shortcuts(reg, "y", ["gc"])["features"][0]["score"] > 0.9
    fam = [{"family": f"F{i % 3}", "site": f"s{i % 3}"} for i in range(90)]
    out = checks.shortcuts(fam, "family", ["site"])
    assert out["task"] == "multiclass" and out["features"][0]["score"] == 1.0


# --- duplicate rows ------------------------------------------------------------------------------------------


def test_duplicate_rows_finds_reprocessed_samples():
    rng = np.random.default_rng(3)
    cols = [f"f{i}" for i in range(10)]
    train = [{c: float(v) for c, v in zip(cols, row)} for row in rng.normal(0, 1, (300, 10))]
    fresh = [{c: float(v) for c, v in zip(cols, row)} for row in rng.normal(0, 1, (100, 10))]
    copies = [{c: r[c] + float(rng.normal(0, 0.01)) for c in cols} for r in train[:40]]
    out = checks.duplicate_rows(train, fresh + copies, cols)
    assert out["test_rows_near_duplicate"] >= 38
    assert out["test_rows_near_duplicate_fraction"] > 0.25
    assert "F12" in out["summary"]


def test_duplicate_rows_clean_split_reports_none():
    rng = np.random.default_rng(4)
    cols = [f"f{i}" for i in range(10)]
    train = [{c: float(v) for c, v in zip(cols, row)} for row in rng.normal(0, 1, (300, 10))]
    test = [{c: float(v) for c, v in zip(cols, row)} for row in rng.normal(0, 1, (100, 10))]
    out = checks.duplicate_rows(train, test, cols)
    assert out["test_rows_near_duplicate"] == 0
    assert "No feature-space duplicates" in out["summary"]


# --- k-mer baseline ------------------------------------------------------------------------------------------


def test_kmer_baseline_composition_separates_gc_classes():
    rng = random.Random(3)
    levels = {"low": 0.35, "mid": 0.5, "high": 0.65}

    def make(n):
        return [(f"s{i}", lab, rand_seq(rng, 3000, gc)) for i in range(n) for lab, gc in levels.items()]

    out = checks.kmer_baseline(make(10), make(5), k=4, model_score=1.0)
    assert out["task"] == "multiclass" and out["chance"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["gc_only"]["score"] == 1.0 and out["kmer_4"]["score"] > 0.9
    assert "F01" in out["summary"]


def test_kmer_baseline_finds_nothing_without_composition_signal():
    rng = random.Random(4)
    train = [(f"a{i}", str(i % 2), rand_seq(rng, 1500)) for i in range(40)]
    test = [(f"b{i}", str(i % 2), rand_seq(rng, 1500)) for i in range(40)]
    out = checks.kmer_baseline(train, test, k=4)
    assert abs(out["kmer_4"]["score"] - 0.5) < 0.2


# --- through the MCP server ----------------------------------------------------------------------------------


def call(tool: str, args: dict):
    async def go():
        async with Client(server.mcp) as c:
            return await c.call_tool(tool, args)

    return asyncio.run(go())


def test_tools_inline_and_path_policy(tmp_path):
    rows = ep_pairs()
    out = call(
        "check_split_overlap",
        {
            "columns": ["enhancer", "promoter"],
            "train": to_csv(rows[:300]),
            "test": to_csv(rows[300:]),
            "label": "label",
        },
    )
    assert not out.is_error and out.structured_content["n_test"] == 100

    f = tmp_path / "train.csv"
    f.write_text(to_csv(rows))
    server.ALLOW_PATHS = False
    refused = call("check_split_overlap", {"columns": ["enhancer"], "train_path": str(f), "test": to_csv(rows)})
    assert refused.is_error and "only read by a local" in refused.content[0].text
    server.ALLOW_PATHS = True
    try:
        ok = call("check_split_overlap", {"columns": ["enhancer"], "train_path": str(f), "test": to_csv(rows)})
        assert not ok.is_error and ok.structured_content["test_rows_with_any_seen_entity"] == 1.0
    finally:
        server.ALLOW_PATHS = False

    both = call("check_shortcuts", {"label": "label", "features": ["batch"]})
    assert both.is_error and "exactly one of table" in both.content[0].text
    bad = call("check_shortcuts", {"label": "nope", "features": ["batch"], "table": to_csv(batch_table())})
    assert bad.is_error and "no column" in bad.content[0].text


def test_kmer_baseline_tool_csv():
    rng = random.Random(5)
    rows = [
        {"id": i, "label": lab, "sequence": rand_seq(rng, 2000, gc)}
        for i in range(20)
        for lab, gc in (("a", 0.35), ("b", 0.65))
    ]
    out = call("kmer_baseline", {"train": to_csv(rows[:30]), "test": to_csv(rows[30:]), "k": 3})
    assert not out.is_error and out.structured_content["gc_only"]["score"] == 1.0
