"""audit_directory on synthetic analysis directories: it must find the planted flaw, and run nothing where
nothing was planted."""

import random
from pathlib import Path

from test_checks import ep_pairs, mutate, rand_seq, to_csv

from ml_genomics_failures import audit, server


def write_pair(tmp_path: Path, train: list[dict], test: list[dict]):
    (tmp_path / "pairs_train.csv").write_text(to_csv(train))
    (tmp_path / "pairs_test.csv").write_text(to_csv(test))


def analyses(report: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for a in report["analyses"]:
        out.setdefault(a["check"], []).append(a)
    return out


def test_finds_entity_leakage_in_named_pair(tmp_path):
    rows = ep_pairs()
    random.Random(1).shuffle(rows)
    write_pair(tmp_path, rows[:300], rows[300:])
    got = analyses(audit.audit_directory(str(tmp_path), model_score=0.80))
    overlap = got["check_split_overlap"][0]["result"]
    assert overlap["test_rows_with_any_seen_entity"] == 1.0
    assert overlap["entity_memorization_baseline"]["share_of_model_gain"] > 0.6


def test_entity_disjoint_pair_reports_clean(tmp_path):
    rows = ep_pairs()
    held = {r["enhancer"] for r in rows if int(r["enhancer"][1:]) >= 30}
    train = [r for r in rows if r["enhancer"] not in held]
    test = [r for r in rows if r["enhancer"] in held]
    write_pair(tmp_path, train, test)
    got = analyses(audit.audit_directory(str(tmp_path)))
    assert got["check_split_overlap"][0]["result"]["columns"]["enhancer"]["test_rows_with_seen_value"] == 0.0


def test_predictions_file_is_not_mistaken_for_the_split(tmp_path):
    rows = ep_pairs()
    random.Random(1).shuffle(rows)
    write_pair(tmp_path, rows[:300], rows[300:])
    (tmp_path / "predictions_test.csv").write_text(
        to_csv([{"enhancer": r["enhancer"], "promoter": r["promoter"], "score": 0.9} for r in rows[300:]])
    )
    got = analyses(audit.audit_directory(str(tmp_path)))
    assert len(got["check_split_overlap"]) == 1  # only the real test split, not train-vs-predictions


def test_sequence_csv_pair_runs_similarity_and_composition(tmp_path):
    rng = random.Random(2)
    train = [
        {"id": f"tr{i}", "label": lab, "sequence": rand_seq(rng, 400, 0.7 if lab == "a" else 0.3)}
        for i, lab in enumerate(["a", "b"] * 40)
    ]
    test = [
        {"id": f"te{i}", "label": lab, "sequence": rand_seq(rng, 400, 0.7 if lab == "a" else 0.3)}
        for i, lab in enumerate(["a", "b"] * 30)
    ]
    (tmp_path / "train.csv").write_text(to_csv(train))
    (tmp_path / "test.csv").write_text(to_csv(test))
    got = analyses(audit.audit_directory(str(tmp_path), model_score=0.95))
    assert "check_sequence_similarity" in got and "kmer_baseline" in got
    assert got["kmer_baseline"][0]["result"]["gc_only"]["score"] > 0.9


def test_split_column_partitions_one_table(tmp_path):
    rng = random.Random(3)
    rows = [
        {"id": f"s{i}", "split": split, "bound": int(lab), "sequence": rand_seq(rng, 300, 0.65 if lab else 0.35)}
        for i, (split, lab) in enumerate([("train", i % 2) for i in range(60)] + [("test", i % 2) for i in range(40)])
    ]
    (tmp_path / "sequences.csv").write_text(to_csv(rows))
    got = analyses(audit.audit_directory(str(tmp_path)))
    assert "check_split_overlap" in got and "check_sequence_similarity" in got


def test_fasta_pair_and_labels_table(tmp_path):
    rng = random.Random(4)
    train_fa = {f"tr{i}": s for i in range(20) if (s := rand_seq(rng, 300))}
    test_fa = {f"te{i}": mutate(rng, rng.choice(list(train_fa.values())), 0.005) for i in range(15)}
    (tmp_path / "train.fa").write_text("".join(f">{i}\n{s}\n" for i, s in train_fa.items()))
    (tmp_path / "test.fa").write_text("".join(f">{i}\n{s}\n" for i, s in test_fa.items()))
    labels = [{"id": i, "split": "train" if i.startswith("tr") else "test", "activity": "active"} for i in
              list(train_fa) + list(test_fa)]
    (tmp_path / "labels.csv").write_text(to_csv(labels))
    got = analyses(audit.audit_directory(str(tmp_path)))
    bins = got["check_sequence_similarity"][0]["result"]["by_identity"]
    assert sum(b["n_test"] for b in bins[:2]) == 15  # all are near-copies at >=95% estimated identity
    assert "kmer_baseline" in got  # labels table covers the FASTA ids


def test_unpaired_table_with_confounder(tmp_path):
    rows = [
        {"sample": f"s{i}", "status": "case" if i < 40 else "control", "plate": 1 if i < 40 else 2}
        for i in range(80)
    ]
    (tmp_path / "samples.csv").write_text(to_csv(rows))
    got = analyses(audit.audit_directory(str(tmp_path)))
    top = got["check_shortcuts"][0]["result"]["features"][0]
    assert top["feature"] == "plate" and top["score"] == 1.0


def test_directory_without_data_files(tmp_path):
    (tmp_path / "notes.txt").write_text("no data here")
    report = audit.audit_directory(str(tmp_path))
    assert report["analyses"] == [] and "note" in report


def test_server_tool(tmp_path):
    server.ALLOW_PATHS = True
    try:
        rows = ep_pairs()[:80]
        random.Random(1).shuffle(rows)
        write_pair(tmp_path, rows[:50], rows[50:])
        from test_server import call, structured

        out = structured(call("audit_directory", {"path": str(tmp_path)}))
        assert out["analyses"] and out["analyses"][0]["check"] == "check_split_overlap"
    finally:
        server.ALLOW_PATHS = False


def test_server_tool_refuses_when_hosted(tmp_path):
    from test_server import call

    server.ALLOW_PATHS = False
    res = call("audit_directory", {"path": str(tmp_path)})
    assert res.is_error and "hosted" in res.content[0].text
