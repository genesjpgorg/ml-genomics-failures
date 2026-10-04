"""One-call audit of an analysis directory: find the data files, guess the split, run the right checks.

``audit_directory(path)`` scans a directory for CSV/TSV tables and FASTA files, pairs them into train/test
splits (from file names or a ``split`` column), guesses the label, entity and sequence columns, and runs the
checks that apply: entity leakage (L), sequence similarity (L), single-feature shortcuts (S) and the
composition baseline (B). Everything it guessed is reported, so the caller can rerun a single ``check_*``
with corrected arguments; the full numbers stay available there.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from . import checks

MAX_FILE_MB = 64
MAX_FEATURES = 12  # per-table candidate confounders handed to shortcuts()
MAX_ENTITIES = 6  # entity columns handed to split_overlap()
SEQ_SAMPLE = 200

TRAIN_TOKENS = {"train", "training", "tr"}
TEST_TOKENS = {"test", "testing", "te", "val", "valid", "validation", "dev", "holdout", "heldout"}
SPLIT_COL_NAMES = {"split", "set", "partition", "subset", "fold", "group"}
LABEL_NAMES = {
    "label", "labels", "target", "class", "y", "status", "outcome", "phenotype", "activity",
    "response", "category", "truth", "groundtruth", "ground_truth", "is_positive",
}
SEQ_NAMES = {"sequence", "seq", "dna", "rna", "nucleotide", "nucleotides", "peptide", "protein_seq", "orf"}
ID_NAMES = {"id", "name", "sample_id", "seq_id", "sequence_id", "protein_id", "gene_id", "accession", "idx"}
PRED_NAMES = {"score", "prediction", "predicted", "pred", "probability", "prob", "model_score"}
DNA = re.compile(r"^[ACGTUNBDHKMRSVWYacgtunbdhkmrsvwy\-]+$")
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZacdefghiklmnpqrstvwybxz\-]+$")


def _stem_tokens(name: str) -> set[str]:
    stem = Path(name).stem
    if stem.endswith((".csv", ".tsv", ".fa", ".fasta")):
        stem = Path(stem).stem  # a double extension like train.fa.gz loses one level at a time
    return {t for t in re.split(r"[^a-z]+|(?<=[a-z])(?=[A-Z])", stem.lower()) if t}


def _role(path: Path) -> str | None:
    """'train', 'test' or None, from file-name tokens."""
    tokens = _stem_tokens(path.name)
    if tokens & TEST_TOKENS:
        return "test"
    if tokens & TRAIN_TOKENS:
        return "train"
    return None


def _is_seq(values: list[str]) -> bool:
    """Do these string values look like nucleotide (or, failing that, amino-acid) sequences?"""
    vals = [v.strip() for v in values[:SEQ_SAMPLE] if v.strip()]
    if not vals:
        return False
    if sum(set(v.upper()) <= set("ACGTUN") and len(v) >= 20 for v in vals) / len(vals) >= 0.8:
        return True
    return sum(bool(AA.match(v)) and len(v) > 15 for v in vals) / len(vals) >= 0.8


class Table:
    """A parsed CSV/TSV with its columns classified."""

    def __init__(self, path: Path, text: str):
        self.path, self.name = path, path.name
        self.rows = checks.parse_csv(text, self.name)
        self.cols = list(self.rows[0])
        self.label = self._find_label()
        self.seq = self._find_seq()
        self.split = self._find_split()
        self.ids = self._find_ids()

    def values(self, col: str) -> list[str]:
        return [r.get(col, "") for r in self.rows]

    def _find_label(self) -> str | None:
        low = {c.lower(): c for c in self.cols}
        for name in LABEL_NAMES:
            if name in low:
                return low[name]
        for c in self.cols:  # e.g. "interaction_label", "case_label"
            if any(k in c.lower() for k in ("label", "target", "outcome", "phenotype")):
                return c
        # fallback: the only low-cardinality categorical column ("bound"/"unbound", "responder", ...)
        if len(self.cols) > 2:
            few = [
                c for c in self.cols
                if c.lower() not in SPLIT_COL_NAMES and c.lower() not in ID_NAMES
                and 2 <= len({v for v in self.values(c) if v.strip()}) <= 4
                and not self._is_seq_col(c)
            ]
            if len(few) == 1:
                return few[0]
        return None

    def _is_seq_col(self, c: str) -> bool:
        return c.lower() in SEQ_NAMES or _is_seq(self.values(c))

    def _find_seq(self) -> str | None:
        low = {c.lower(): c for c in self.cols}
        for name in SEQ_NAMES:
            if name in low:
                return low[name]
        for c in self.cols:
            if _is_seq(self.values(c)):
                return c
        return None

    def _find_split(self) -> str | None:
        for c in self.cols:
            if c.lower() in SPLIT_COL_NAMES:
                vals = {v.strip().lower() for v in self.values(c) if v.strip()}
                if vals and vals <= (TRAIN_TOKENS | TEST_TOKENS | {"valid"}):
                    return c
        return None

    def _find_ids(self) -> str | None:
        for c in self.cols:
            if c.lower() in ID_NAMES:
                return c
        return self.cols[0] if self.cols else None

    def entity_cols(self) -> list[str]:
        """Categorical columns that could identify a biological entity (gene, protein, batch, plate...)."""
        skip = {self.label, self.seq, self.split}
        out = []
        for c in self.cols:
            if c in skip:
                continue
            vals = self.values(c)
            uniq = {v for v in vals if v.strip()}
            if not uniq:
                continue
            numeric = checks._num(vals) is not None
            if not numeric or len(uniq) <= 10:
                out.append(c)
        return out[:MAX_ENTITIES]

    def feature_cols(self) -> list[str]:
        """Everything that isn't the label/sequence/split/id: candidate confounders for shortcuts()."""
        skip = {self.label, self.seq, self.split, self.ids}
        return [c for c in self.cols if c not in skip][:MAX_FEATURES]


def _is_table(p: Path) -> bool:
    return p.suffix.lower() in (".csv", ".tsv", ".txt") or p.name.endswith((".csv.gz", ".tsv.gz"))


def _is_fasta(p: Path) -> bool:
    return p.suffix.lower() in (".fa", ".fasta", ".fna", ".faa") or p.name.endswith((".fa.gz", ".fasta.gz"))


def _compact(result: dict) -> dict:
    """Keep the summary and small fields; drop per-example lists the single checks already truncate."""
    drop = {"most_similar", "examples"}
    out = {}
    for k, v in result.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _compact(v)
        elif isinstance(v, list) and len(v) > 12:
            out[k] = v[:12] + [f"... ({len(v)} total)"]
        else:
            out[k] = v
    return out


def audit_directory(path: str, model_score: float | None = None) -> dict[str, Any]:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise checks.CheckError(f"no such directory: {path}")
    tables: list[Table] = []
    fastas: dict[str, dict[str, str]] = {}
    files, skipped = [], []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix == ".gz" and not _is_table(p) and not _is_fasta(p):
            continue
        if not (_is_table(p) or _is_fasta(p)):
            continue
        if p.stat().st_size > MAX_FILE_MB * 2**20:
            skipped.append(f"{p.name}: {p.stat().st_size / 2**20:.0f} MB, larger than {MAX_FILE_MB} MB")
            continue
        try:
            text = checks.read_text(str(p))
            if _is_fasta(p):
                fastas[p.name] = checks.parse_fasta(text, p.name)
                files.append({"file": p.name, "kind": "fasta", "records": len(fastas[p.name]), "role": _role(p)})
            else:
                t = Table(p, text)
                tables.append(t)
                files.append(
                    {
                        "file": p.name,
                        "kind": "table",
                        "rows": len(t.rows),
                        "columns": t.cols,
                        "role": _role(p),
                        "guessed": {
                            k: v for k, v in
                            {"label": t.label, "sequence": t.seq, "split": t.split, "id": t.ids}.items() if v
                        },
                    }
                )
        except checks.CheckError as e:
            skipped.append(f"{p.name}: {e}")
    report: dict[str, Any] = {"directory": str(root), "files": files, "analyses": [], "skipped": skipped}
    analyses: list[dict] = report["analyses"]

    def add(check: str, on: str, fn, *args, **kwargs):
        try:
            res = fn(*args, **kwargs)
            analyses.append({"check": check, "on": on, "summary": res.get("summary"), "result": _compact(res)})
        except checks.CheckError as e:
            analyses.append({"check": check, "on": on, "error": str(e)})

    # --- pair tables by file-name role ---------------------------------------------------------------------
    role_of = {t.name: _role(t.path) for t in tables}
    trains = [t for t in tables if role_of[t.name] == "train"]
    tests = [t for t in tables if role_of[t.name] == "test"]
    used: set[str] = set()

    def pair_tables(tr: Table, te: Table, how: str):
        on = f"{tr.name} vs {te.name} ({how})"
        common = [c for c in tr.cols if c in te.cols]
        if tr.label and tr.label not in common:
            return  # a "test" file without the train label is an output (predictions, scores), not the split
        label = tr.label if tr.label in common else None
        entities = [c for c in tr.entity_cols() if c in common]
        seq = tr.seq if tr.seq in common else None
        if not entities and not seq:
            return  # nothing shared to check
        if entities:
            add("check_split_overlap", on, checks.split_overlap, tr.rows, te.rows, entities, label, "auto",
                model_score)
        if seq:
            tr_f = {r.get(tr.ids or "id", str(i)): r[tr.seq] for i, r in enumerate(tr.rows)}
            te_f = {r.get(te.ids or "id", str(i)): r[te.seq] for i, r in enumerate(te.rows)}
            add("check_sequence_similarity", on, checks.sequence_similarity, tr_f, te_f)
            if label:
                add("kmer_baseline", on, checks.kmer_baseline,
                    [(k, r[label], r[tr.seq]) for k, r in zip(tr_f, tr.rows)],
                    [(k, r[label], r[te.seq]) for k, r in zip(te_f, te.rows)], 5, "auto", model_score)
        both = tr.rows + te.rows
        features = [c for c in tr.feature_cols() if c in common]
        if label and features:
            add("check_shortcuts", on, checks.shortcuts, both, label, features, "auto", None, 5, model_score)
        nums = [c for c in features if c not in entities and checks._num(tr.values(c)) is not None]
        if len(nums) >= 3:  # nearest-neighbour spacing is uninformative in 1-2 dimensions
            add("check_duplicate_rows", on, checks.duplicate_rows, tr.rows, te.rows, nums)

    if trains and tests:
        used |= {t.name for t in trains + tests}
        for tr in trains:
            for te in tests:
                pair_tables(tr, te, "file names")

    # --- tables with an explicit split column ---------------------------------------------------------------
    for t in tables:
        if t.name in used or not t.split:
            continue
        tr = [r for r in t.rows if r[t.split].strip().lower() in TRAIN_TOKENS]
        te = [r for r in t.rows if r[t.split].strip().lower() in TEST_TOKENS]
        if tr and te:
            used.add(t.name)
            _pair_rows(t, tr, te, add, model_score)

    # --- unpaired labelled tables: shortcut scan ---------------------------------------------------------
    for t in tables:
        if t.name in used or not t.label:
            continue
        features = t.feature_cols()
        if features:
            used.add(t.name)
            add("check_shortcuts", t.name, checks.shortcuts, t.rows, t.label, features, "auto", None, 5,
                model_score)

    # --- FASTA pairs ------------------------------------------------------------------------------------
    f_roles = {name: _role(Path(name)) for name in fastas}
    f_tr = [n for n, r in f_roles.items() if r == "train"]
    f_te = [n for n, r in f_roles.items() if r == "test"]
    label_map = _label_map(tables)  # id -> (split, label) from tables with id+split+label
    for a in f_tr:
        for b in f_te:
            on = f"{a} vs {b} (file names)"
            scores = _test_scores(tables, fastas[b])
            add("check_sequence_similarity", on, checks.sequence_similarity, fastas[a], fastas[b], 15, None,
                scores or None)
            triples = _fasta_triples(fastas[a], fastas[b], label_map)
            if triples:
                add("kmer_baseline", on, checks.kmer_baseline, triples[0], triples[1], 5, "auto", model_score)

    if not analyses:
        report["note"] = (
            "No train/test split or labelled table was detected. If the split lives elsewhere, run the "
            "check_* tools directly with the right files and columns."
        )
    else:
        report["hint"] = (
            "Guessed labels, entities and splits are listed per file under 'guessed'; rerun a check_* tool "
            "with corrected columns if a guess is wrong. Pass model_score to weigh each result against the model."
        )
    return report


def _pair_rows(t: Table, tr: list[dict], te: list[dict], add, model_score: float | None):
    """split_overlap/similarity/kmer for one table partitioned by its split column."""
    on = f"{t.name} split by {t.split}"
    entities = t.entity_cols()
    if entities:
        add("check_split_overlap", on, checks.split_overlap, tr, te, entities, t.label, "auto", model_score)
    if t.seq:
        key = t.ids or t.cols[0]
        add("check_sequence_similarity", on, checks.sequence_similarity,
            {r.get(key, str(i)): r[t.seq] for i, r in enumerate(tr)},
            {r.get(key, str(i)): r[t.seq] for i, r in enumerate(te)})
        if t.label:
            add("kmer_baseline", on, checks.kmer_baseline,
                [(r.get(key, str(i)), r[t.label], r[t.seq]) for i, r in enumerate(tr)],
                [(r.get(key, str(i)), r[t.label], r[t.seq]) for i, r in enumerate(te)], 5, "auto", model_score)
    nums = [c for c in t.feature_cols() if c not in entities and checks._num(t.values(c)) is not None]
    if len(nums) >= 3:  # nearest-neighbour spacing is uninformative in 1-2 dimensions
        add("check_duplicate_rows", on, checks.duplicate_rows, tr, te, nums)


def _label_map(tables: list[Table]) -> dict[str, tuple[str, str]]:
    """id -> (split, label) gathered from any table that has id, split and label columns."""
    out = {}
    for t in tables:
        if t.split and t.label and t.ids:
            for r in t.rows:
                out[r.get(t.ids, "")] = (r[t.split].strip().lower(), r[t.label])
    return out


def _test_scores(tables: list[Table], test_ids: dict) -> dict[str, float]:
    """Per-test-id scores from a predictions table (id + numeric score, or predicted class vs known label)."""
    for t in tables:
        if not t.ids or t.split or t.label:
            continue
        pred_col = next((c for c in t.cols if c.lower() in PRED_NAMES), None)
        if not pred_col:
            continue
        ids = {r.get(t.ids, "") for r in t.rows}
        if not ids or len(ids & set(test_ids)) < max(1, len(test_ids) // 2):
            continue
        num = checks._num(t.values(pred_col))
        if num is not None:  # a numeric score/probability per test row
            return {r[t.ids]: float(v) for r, v in zip(t.values(t.ids), num) if not math.isnan(v)}
    return {}


def _fasta_triples(
    train_fa: dict[str, str], test_fa: dict[str, str], label_map: dict[str, tuple[str, str]]
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]] | None:
    """(id, label, seq) triples for kmer_baseline when a labels table covers the FASTA ids."""
    if not label_map:
        return None
    tr = [(i, label_map[i][1], s) for i, s in train_fa.items() if i in label_map]
    te = [(i, label_map[i][1], s) for i, s in test_fa.items() if i in label_map]
    return (tr, te) if tr and te else None
