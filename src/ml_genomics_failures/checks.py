"""Executable checks for the pitfalls in the book: measure them on a user's own data.

Pure numpy, no model fitting beyond out-of-fold lookups, so the checks are fast and have nothing to tune:
- ``split_overlap``: entities shared by train and test, and the score of memorizing each entity's label (L).
- ``sequence_similarity``: each test sequence's nearest training sequence by k-mer containment (L).
- ``shortcuts``: how much of the label each candidate feature predicts on its own (S).
- ``kmer_baseline``: GC-only and k-mer composition baselines for sequence -> label (B).

Every check returns a JSON-able dict with a plain-language ``summary`` and the numbers behind it.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np

MAX_EXACT_BP = 20_000_000  # above this, sequence_similarity subsamples k-mers (FracMinHash)
BELOW_CHANCE = (
    "below chance means no association, not a negative one: predicting held-out rows from the other rows of an "
    "uninformative category is biased below chance"
)
SHARE_FLAG = 0.8  # a baseline reaching this fraction of the model's gain over chance "explains most of it"


class CheckError(ValueError):
    """Bad input; the message is meant for the caller."""


# --- inputs --------------------------------------------------------------------------------------------------


def read_text(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise CheckError(f"no such file: {path}")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as h:
        return h.read()


def parse_csv(text: str, name: str) -> list[dict]:
    """CSV or TSV (sniffed from the header line) into row dicts."""
    text = text.strip()
    if not text:
        raise CheckError(f"{name} is empty")
    header = text.splitlines()[0]
    delim = "\t" if header.count("\t") > header.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    if not rows:
        raise CheckError(f"{name} has a header but no rows")
    return rows


def parse_fasta(text: str, name: str) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith(">"):
            cur = line[1:].split()[0] if line[1:].strip() else f"seq{len(seqs) + 1}"
            if cur in seqs:
                raise CheckError(f"{name}: duplicate sequence id {cur!r}")
            seqs[cur] = []
        elif line.strip():
            if cur is None:
                raise CheckError(f"{name}: sequence data before the first '>' header")
            seqs[cur].append(line.strip())
    if not seqs:
        raise CheckError(f"{name}: no FASTA records")
    return {k: "".join(v) for k, v in seqs.items()}


def _strs(rows: list[dict]) -> list[dict]:
    """Row values as strings, as CSV gives them (callers in Python may pass numbers)."""
    return [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]


def require_columns(rows: list[dict], cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise CheckError(f"{name} has no column(s) {missing}; columns are {list(rows[0])}")


# --- statistics ----------------------------------------------------------------------------------------------


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 1-based."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2)[inv]


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = ranks(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def _num(values: list[str]) -> np.ndarray | None:
    """Floats if every non-empty value parses as one, else None."""
    try:
        return np.array([float(v) if v.strip() != "" else np.nan for v in values])
    except ValueError:
        return None


class Label:
    """A label column typed as binary, multiclass or regression, with its metric and chance level."""

    def __init__(self, values: list[str], task: str = "auto"):
        values = [str(v) for v in values]
        num = _num(values)
        uniq = sorted(set(values))
        if task == "auto":
            task = "binary" if len(uniq) == 2 else "regression" if num is not None and len(uniq) > 10 else "multiclass"
        if task not in ("binary", "multiclass", "regression"):
            raise CheckError("task must be auto, binary, multiclass or regression")
        if task == "binary" and len(uniq) != 2:
            raise CheckError(f"binary task needs exactly 2 label values, found {len(uniq)}")
        if task == "regression" and num is None:
            raise CheckError("regression task needs a numeric label")
        self.task, self.values = task, values
        if task == "binary":
            pos = next((u for u in uniq if u.lower() in ("1", "true", "yes", "pos", "positive")), uniq[1])
            self.positive = pos
            self.y = np.array([v == pos for v in values], dtype=float)
        elif task == "regression":
            self.y = num
        else:
            self.y = np.array(values, dtype=object)

    @property
    def metric(self) -> str:
        return {"binary": "auroc", "multiclass": "accuracy", "regression": "spearman"}[self.task]

    def chance(self) -> float:
        if self.task == "binary":
            return 0.5
        if self.task == "regression":
            return 0.0
        return Counter(self.values).most_common(1)[0][1] / len(self.values)  # majority class

    def score(self, idx: np.ndarray, pred: np.ndarray) -> float:
        y = self.y[idx]
        if self.task == "binary":
            return auroc(y, pred)
        if self.task == "regression":
            ok = ~np.isnan(y) & ~np.isnan(pred.astype(float))
            return spearman(y[ok], pred[ok].astype(float))
        return float(np.mean(pred == y))


def _encode(lab: Label, train: np.ndarray, test: np.ndarray, cat: np.ndarray) -> tuple[np.ndarray, float]:
    """Predict test rows from the training rows of the same category (mean label, or majority class); unseen
    categories get the training-wide value. Returns predictions and the fraction of test rows whose category was
    seen in training."""
    groups: dict = defaultdict(list)
    for i in train:
        groups[cat[i]].append(i)
    if lab.task == "multiclass":
        fallback = Counter(lab.y[train]).most_common(1)[0][0]
        table = {k: Counter(lab.y[v]).most_common(1)[0][0] for k, v in groups.items()}
        pred = np.array([table.get(cat[i], fallback) for i in test], dtype=object)
    else:
        y = lab.y
        fallback = float(np.nanmean(y[train]))
        table = {k: float(np.nanmean(y[v])) if not np.all(np.isnan(y[v])) else fallback for k, v in groups.items()}
        pred = np.array([table.get(cat[i], fallback) for i in test], dtype=float)
    seen = float(np.mean([cat[i] in table for i in test])) if len(test) else float("nan")
    return pred, seen


def folds(n: int, k: int, groups: list[str] | None, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if groups is None:
        return np.array_split(rng.permutation(n), k)
    uniq = sorted(set(groups))
    if len(uniq) < k:
        raise CheckError(f"group column has {len(uniq)} groups, fewer than {k} folds")
    fold_of = {g: i % k for i, g in enumerate(rng.permutation(uniq))}
    return [np.array([i for i, g in enumerate(groups) if fold_of[g] == f]) for f in range(k)]


def _share(baseline: float, chance: float, model: float | None) -> float | None:
    """Fraction of the model's gain over chance that the baseline already reaches."""
    if model is None or not math.isfinite(baseline) or model - chance <= 1e-9:
        return None
    return round((baseline - chance) / (model - chance), 3)


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


# --- split overlap -------------------------------------------------------------------------------------------


def split_overlap(
    train: list[dict],
    test: list[dict],
    columns: list[str],
    label: str | None = None,
    task: str = "auto",
    model_score: float | None = None,
) -> dict:
    """Entities shared between train and test, per column, and (with ``label``) the test score of predicting each
    test row from its entities' training labels alone."""
    if not columns:
        raise CheckError("give at least one entity column (e.g. gene, enhancer, promoter, species, individual)")
    train, test = _strs(train), _strs(test)
    require_columns(train, columns, "train")
    require_columns(test, columns, "test")
    per_column = {}
    seen_any = np.zeros(len(test), bool)
    seen_all = np.ones(len(test), bool)
    for col in columns:
        tr = {r[col] for r in train}
        te = [r[col] for r in test]
        hit = np.array([v in tr for v in te])
        seen_any |= hit
        seen_all &= hit
        shared = sorted(tr & set(te))
        per_column[col] = {
            "train_unique": len(tr),
            "test_unique": len(set(te)),
            "shared_unique": len(shared),
            "test_rows_with_seen_value": _r(hit.mean()),
            "examples": shared[:10],
        }
    keys_tr = {tuple(r[c] for c in columns) for r in train}
    dup = float(np.mean([tuple(r[c] for c in columns) in keys_tr for r in test]))
    out = {
        "n_train": len(train),
        "n_test": len(test),
        "columns": per_column,
        "test_rows_with_any_seen_entity": _r(seen_any.mean()),
        "test_rows_with_all_entities_seen": _r(seen_all.mean()),
        "test_rows_duplicated_in_train": _r(dup),
    }
    if label:
        require_columns(train, [label], "train")
        require_columns(test, [label], "test")
        lab = Label([r[label] for r in train] + [r[label] for r in test], task)
        tr_idx, te_idx = np.arange(len(train)), np.arange(len(train), len(train) + len(test))
        rows = train + test
        preds = []
        for col in columns:
            cat = np.array([r[col] for r in rows], dtype=object)
            preds.append(_encode(lab, tr_idx, te_idx, cat)[0])
        if lab.task == "multiclass":  # vote across columns, ties to the first column
            pred = np.array([Counter(p[i] for p in preds).most_common(1)[0][0] for i in range(len(te_idx))], object)
        else:
            pred = np.mean(preds, axis=0)
        base = lab.score(te_idx, pred)
        chance = Label(lab.values[len(train) :], lab.task).chance() if lab.task == "multiclass" else lab.chance()
        out["entity_memorization_baseline"] = {
            "task": lab.task,
            "metric": lab.metric,
            "score": _r(base),
            "chance": _r(chance),
            "model_score": model_score,
            "share_of_model_gain": _share(base, chance, model_score),
            "how": "each test row is predicted from the training labels of its own entities (unseen entity -> "
            "training-wide rate); no features, no learning",
        }
    out["summary"] = _overlap_summary(out, columns)
    return out


def _overlap_summary(o: dict, columns: list[str]) -> str:
    parts = []
    any_seen = o["test_rows_with_any_seen_entity"] or 0
    if any_seen > 0:
        cols = ", ".join(f"{c} {v['test_rows_with_seen_value']:.0%}" for c, v in o["columns"].items())
        parts.append(
            f"{any_seen:.0%} of test rows share at least one entity with training ({cols}). The split isn't "
            f"independent at the level of {', '.join(columns)} (class L; see F03, F04, F09)."
        )
    else:
        parts.append(f"No test row shares a {'/'.join(columns)} value with training: the split is entity-disjoint.")
    if o["test_rows_duplicated_in_train"]:
        parts.append(f"{o['test_rows_duplicated_in_train']:.0%} of test rows repeat a training row exactly.")
    b = o.get("entity_memorization_baseline")
    if b and b["score"] is not None:
        line = f"Memorizing entity label rates scores {b['metric']} {b['score']} (chance {b['chance']})"
        if b["share_of_model_gain"] is not None:
            line += f", {b['share_of_model_gain']:.0%} of the model's gain over chance"
            if b["share_of_model_gain"] >= SHARE_FLAG:
                line += ": most of the reported score is reachable without learning anything about interactions"
        parts.append(line + ".")
    if any_seen > 0:
        parts.append("Re-split so no entity is on both sides (or by chromosome) and re-evaluate.")
    return " ".join(parts)


# --- sequence similarity -------------------------------------------------------------------------------------

_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[ord(chr(_b).lower())] = _i
_LUT[ord("U")] = _LUT[ord("u")] = 3


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def kmer_hashes(seq: str, k: int, scale: int = 1, chunk: int = 1_000_000) -> np.ndarray:
    """Unique hashed canonical k-mers of a DNA/RNA sequence (k-mers with non-ACGT bases skipped); with
    ``scale`` > 1 only hashes below 2^64/scale are kept (FracMinHash)."""
    x = _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    if len(x) < k:
        return np.empty(0, np.uint64)
    shifts = (2 * np.arange(k - 1, -1, -1)).astype(np.uint64)
    out = []
    for start in range(0, len(x) - k + 1, chunk):
        w = np.lib.stride_tricks.sliding_window_view(x[start : start + chunk + k - 1], k)
        w = w[(w != 255).all(1)].astype(np.uint64)
        if not len(w):
            continue
        fwd = (w << shifts).sum(1, dtype=np.uint64)
        rev = ((np.uint64(3) - w[:, ::-1]) << shifts).sum(1, dtype=np.uint64)
        h = _splitmix(np.minimum(fwd, rev))
        if scale > 1:
            h = h[h < np.uint64(2**64 // scale)]
        out.append(h)
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.uint64)


def sequence_similarity(
    train: dict[str, str],
    test: dict[str, str],
    k: int = 15,
    scale: int | None = None,
    test_scores: dict[str, float] | None = None,
    max_occurrences: int = 1000,
) -> dict:
    """Nearest training sequence of every test sequence by k-mer containment (the fraction of the test sequence's
    k-mers found in that training sequence), with a Mash-style identity estimate containment^(1/k)."""
    if not 7 <= k <= 31:
        raise CheckError("k must be between 7 and 31")
    total = sum(map(len, train.values())) + sum(map(len, test.values()))
    scale = scale or max(1, math.ceil(total / MAX_EXACT_BP))
    tr_ids = list(train)
    hashes, owner = [], []
    for i, t in enumerate(tr_ids):
        h = kmer_hashes(train[t], k, scale)
        hashes.append(h)
        owner.append(np.full(len(h), i, np.int64))
    allh = np.concatenate(hashes) if hashes else np.empty(0, np.uint64)
    allo = np.concatenate(owner) if owner else np.empty(0, np.int64)
    order = np.argsort(allh, kind="stable")
    allh, allo = allh[order], allo[order]
    uniq, cnt = np.unique(allh, return_counts=True)
    common = uniq[cnt > max_occurrences]  # repeats shared by very many training sequences: skip, they'd dominate
    per_test = []
    for t, s in test.items():
        h = kmer_hashes(s, k, scale)
        if len(common):
            h = h[~np.isin(h, common)]
        if not len(h):
            per_test.append({"id": t, "length": len(s), "nearest_train": None, "containment": None, "identity": None})
            continue
        lo = np.searchsorted(allh, h, "left")
        hi = np.searchsorted(allh, h, "right")
        n = hi - lo
        pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
        counts = np.bincount(allo[pos], minlength=len(tr_ids))
        best = int(counts.argmax()) if counts.any() else None
        c = counts[best] / len(h) if best is not None else 0.0
        per_test.append(
            {
                "id": t,
                "length": len(s),
                "nearest_train": tr_ids[best] if best is not None and c > 0 else None,
                "containment": round(float(c), 4),
                "identity": round(float(c ** (1 / k)), 4) if c > 0 else 0.0,
            }
        )
    bins = [(0.99, 1.01, ">=0.99"), (0.95, 0.99, "0.95-0.99"), (0.90, 0.95, "0.90-0.95"), (0.80, 0.90, "0.80-0.90")]
    bins.append((-1.0, 0.80, "<0.80 or unrelated"))
    table = []
    for lo_, hi_, name in bins:
        members = [p for p in per_test if p["identity"] is not None and lo_ <= p["identity"] < hi_]
        row = {"identity": name, "n_test": len(members)}
        if test_scores is not None:
            vals = [float(test_scores[p["id"]]) for p in members if p["id"] in test_scores]
            row["mean_test_score"] = _r(float(np.mean(vals))) if vals else None
            row["n_scored"] = len(vals)
        table.append(row)
    scored = [p for p in per_test if p["identity"] is not None]
    out = {
        "k": k,
        "scale": scale,
        "n_train": len(train),
        "n_test": len(test),
        "skipped_repeat_kmers": len(common),
        "by_identity": table,
        "most_similar": sorted(scored, key=lambda p: -p["containment"])[:20],
        "note": "identity is a k-mer estimate: reliable above ~0.85, a lower bound for diverged sequences; use "
        "MMseqs2 or BLAST for alignment-level identity and for proteins",
    }
    if test_scores is not None:
        pairs = [(p["identity"], float(test_scores[p["id"]])) for p in scored if p["id"] in test_scores]
        if pairs:
            ident, sc = map(np.array, zip(*pairs))
            out["score_vs_identity_spearman"] = _r(spearman(ident, sc))
    out["summary"] = _similarity_summary(out)
    return out


def _similarity_summary(o: dict) -> str:
    n = o["n_test"]
    near = sum(r["n_test"] for r in o["by_identity"][:2])
    parts = [f"{near}/{n} test sequences have a training sequence at >=95% estimated identity."]
    if near / max(n, 1) > 0.1:
        parts.append(
            "Test accuracy partly measures recall of near-copies (class L; see F12). Cluster by identity "
            "(MMseqs2/CD-HIT) and split by cluster."
        )
    rho = o.get("score_vs_identity_spearman")
    if rho is not None:
        parts.append(f"Spearman between test score and identity to training: {rho}.")
        if rho > 0.2:
            parts.append("Performance falls with distance from training: report it by identity bin.")
    return " ".join(parts)


# --- shortcuts -----------------------------------------------------------------------------------------------


def duplicate_rows(
    train: list[dict],
    test: list[dict],
    features: list[str],
    ratio: float = 0.1,
    max_train: int = 4000,
) -> dict:
    """Feature-space near-duplicates across the split (class L; case F12's mechanism for non-sequence rows):
    reprocessed samples, re-annotated rows or perturbed copies can sit on both sides with different IDs and pass
    every entity check. For each test row, distance to the nearest training row in z-scored feature space,
    relative to the typical train-to-train spacing; ``ratio`` of that spacing flags near-copies."""
    if not features:
        raise CheckError("give numeric feature columns (exclude IDs, entities and the label)")
    train, test = _strs(train), _strs(test)
    require_columns(train, features, "train")
    require_columns(test, features, "test")
    x_tr = np.stack([_num([r[f] for r in train]) for f in features], axis=1)
    x_te = np.stack([_num([r[f] for r in test]) for f in features], axis=1)
    if np.isnan(x_tr).any() or np.isnan(x_te).any():
        raise CheckError("features must be numeric without missing values")
    keep = x_tr.std(axis=0) > 0
    if keep.sum() == 0:
        raise CheckError("all features are constant in training")
    x_tr, x_te = x_tr[:, keep], x_te[:, keep]
    dropped = [f for f, k in zip(features, keep) if not k]
    mu, sd = x_tr.mean(axis=0), np.maximum(x_tr.std(axis=0), 1e-12)
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd
    if len(x_tr) > max_train:
        sub = np.random.default_rng(0).choice(len(x_tr), max_train, replace=False)
        x_tr_sub = x_tr[np.sort(sub)]
        note = f"nearest distances computed against a {max_train}-row subsample of training"
    else:
        x_tr_sub, note = x_tr, None
    nearest = np.empty(len(x_te), int)
    dist = np.empty(len(x_te))
    for s in range(0, len(x_te), 500):
        d = np.sqrt(((x_te[s : s + 500, None, :] - x_tr_sub[None, :, :]) ** 2).sum(axis=2))
        nearest[s : s + 500] = d.argmin(axis=1)
        dist[s : s + 500] = d[np.arange(len(d)), nearest[s : s + 500]]
    ref = x_tr if len(x_tr) <= 1000 else x_tr[np.random.default_rng(1).choice(len(x_tr), 1000, False)]
    nn = np.empty(len(ref))
    for s in range(0, len(ref), 500):
        blk = ref[s : s + 500]
        dd = np.sqrt(((blk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
        dd[np.arange(len(blk)), s + np.arange(len(blk))] = np.inf  # exclude self
        nn[s : s + 500] = dd.min(axis=1)
    spacing = float(np.median(nn)) if len(nn) else float("nan")
    flagged = dist < ratio * spacing if np.isfinite(spacing) else dist == 0
    order = np.argsort(dist)[:10]
    low_dim = int(keep.sum()) < 3
    out = {
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "n_features": int(keep.sum()),
        "constant_features_dropped": dropped,
        "median_train_spacing": _r(spacing),
        "near_duplicate_threshold": _r(ratio * spacing),
        "test_rows_near_duplicate": int(flagged.sum()),
        "test_rows_near_duplicate_fraction": _r(float(flagged.mean())),
        "nearest_distance_quantiles": {
            f"q{q}": _r(float(np.quantile(dist, q / 100))) for q in (0, 10, 25, 50, 75)
        },
        "closest_examples": [
            {"test_row": int(i), "train_row": int(nearest[i]), "distance": _r(float(dist[i]))}
            for i in order
        ],
    }
    if note:
        out["note"] = note
    if low_dim:
        out["note"] = (out.get("note") or "") + (
            " Fewer than 3 numeric features: in low dimensions every point is close to a neighbour, "
            "so flagged counts overstate duplication — treat them skeptically."
        ).strip()
    n = out["test_rows_near_duplicate"]
    if n:
        out["summary"] = (
            f"{n} of {len(x_te)} test rows ({out['test_rows_near_duplicate_fraction']:.0%}) sit closer to a "
            f"training row than {ratio:.0%} of typical training spacing — likely reprocessed or perturbed "
            "copies with new IDs. An entity-disjoint split doesn't catch these (class L; F12's mechanism "
            "applied to feature rows). Remove duplicates at the sample level, then re-split."
        )
    else:
        out["summary"] = (
            f"No test row is nearer than {ratio:.0%} of typical training spacing to a training row. "
            "No feature-space duplicates found."
        )
    return out


def shortcuts(
    rows: list[dict],
    label: str,
    features: list[str],
    task: str = "auto",
    group_column: str | None = None,
    n_folds: int = 5,
    model_score: float | None = None,
) -> dict:
    """Cross-validated score of predicting the label from each feature alone. Categorical features predict from
    the training rows of the same category; numeric ones are cut into 10 quantile bins (fitted on training
    folds). With ``group_column`` the folds keep groups together, as an entity-level split would."""
    if not features:
        raise CheckError("give at least one candidate feature column (batch, GC, distance, gene, ancestry, ...)")
    rows = _strs(rows)
    require_columns(rows, [label, *features] + ([group_column] if group_column else []), "table")
    lab = Label([r[label] for r in rows], task)
    groups = [r[group_column] for r in rows] if group_column else None
    fs = folds(len(rows), n_folds, groups)
    chance = lab.chance()
    results = []
    for f in features:
        vals = [r[f] for r in rows]
        num = _num(vals)
        kind = "numeric" if num is not None and len(set(vals)) > 10 else "categorical"
        preds = np.empty(len(rows), dtype=object if lab.task == "multiclass" else float)
        seen = []
        for i, te in enumerate(fs):
            tr = np.concatenate([x for j, x in enumerate(fs) if j != i])
            if kind == "numeric":
                edges = np.unique(np.nanquantile(num[tr], np.linspace(0, 1, 11)[1:-1]))
                cat = np.where(np.isnan(num), -1, np.searchsorted(edges, num, "right")).astype(object)
            else:
                cat = np.array(vals, dtype=object)
            p, s = _encode(lab, tr, te, cat)
            preds[te] = p
            seen.append(s)
        score = _fold_mean(lab, fs, preds, chance)
        results.append(
            {
                "feature": f,
                "kind": kind,
                "score": _r(score),
                "share_of_model_gain": _share(score, chance, model_score),
                "test_rows_with_seen_category": _r(float(np.mean(seen))) if kind == "categorical" else None,
                **({"note": BELOW_CHANCE} if math.isfinite(score) and score < chance - 0.05 else {}),
            }
        )
    results.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1e9))
    out = {
        "task": lab.task,
        "metric": lab.metric,
        "chance": _r(chance),
        "model_score": model_score,
        "folds": f"{n_folds}-fold, grouped by {group_column}" if group_column else f"{n_folds}-fold, random rows",
        "features": results,
    }
    out["summary"] = _shortcut_summary(out)
    return out


def _fold_mean(lab: Label, fs: list[np.ndarray], preds: np.ndarray, chance: float) -> float:
    """Metric per fold, averaged with fold-size weights. Pooling out-of-fold predictions instead would bias the
    score below chance: each fold's fallback (the training mean) anti-correlates with its held-out labels.
    A fold with constant predictions carries no information and scores chance."""
    scores, weights = [], []
    for te in fs:
        p = preds[te]
        constant = len(set(p.tolist())) <= 1
        sc = chance if constant and lab.task != "multiclass" else lab.score(te, p)
        if math.isfinite(sc):
            scores.append(sc)
            weights.append(len(te))
    return float(np.average(scores, weights=weights)) if scores else float("nan")


def _shortcut_summary(o: dict) -> str:
    top = o["features"][0]
    if top["score"] is None or top["score"] <= o["chance"] + 0.05:
        return (
            f"No single feature predicts the label beyond chance ({o['metric']} chance {o['chance']}; best "
            f"{top['feature']} {top['score']}). No shortcut among these features."
        )
    parts = [f"Best single feature: {top['feature']} ({o['metric']} {top['score']}, chance {o['chance']})."]
    flagged = [r for r in o["features"] if (r["share_of_model_gain"] or 0) >= SHARE_FLAG]
    if flagged:
        names = ", ".join(f"{r['feature']} ({r['share_of_model_gain']:.0%})" for r in flagged)
        parts.append(
            f"These reach most of the model's gain over chance on their own: {names}. The headline may be "
            "about them, not the biology (class S; see F10, F11, F01). Re-evaluate with them matched or held "
            "constant."
        )
    elif o["model_score"] is None:
        parts.append("Pass model_score to see what fraction of the model's result each feature explains.")
    return " ".join(parts)


# --- k-mer composition baseline -----------------------------------------------------------------------------


@cache
def _canonical_index(k: int) -> np.ndarray:
    """Index of each of the 4^k k-mer codes' canonical (strand-independent) form."""
    codes = np.arange(4**k)
    digits = (codes[:, None] >> (2 * np.arange(k)[::-1])) & 3
    rc = ((3 - digits[:, ::-1]) << (2 * np.arange(k)[::-1])).sum(1)
    return np.unique(np.minimum(codes, rc), return_inverse=True)[1]


def kmer_profile(seq: str, k: int) -> np.ndarray:
    """Canonical k-mer frequencies (4^k/2-ish dims) of a sequence; k-mers with non-ACGT bases skipped."""
    x = _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    canon = _canonical_index(k)
    counts = np.zeros(canon.max() + 1)
    if len(x) >= k:
        w = np.lib.stride_tricks.sliding_window_view(x, k)
        w = w[(w != 255).all(1)].astype(np.int64)
        counts += np.bincount(canon[(w << (2 * np.arange(k)[::-1])).sum(1)], minlength=len(counts))
    return counts / max(counts.sum(), 1)


def gc_content(seq: str) -> float:
    x = _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    x = x[x != 255]
    return float(np.isin(x, (1, 2)).mean()) if len(x) else float("nan")


def kmer_baseline(
    train: list[tuple[str, str, str]],
    test: list[tuple[str, str, str]],
    k: int = 5,
    task: str = "auto",
    model_score: float | None = None,
    n_neighbors: int = 5,
) -> dict:
    """GC-only and k-mer composition baselines; ``train``/``test`` are (id, label, sequence) triples. Predicts
    each test label from the nearest training sequences (cosine on z-scored k-mer frequencies) and from the
    nearest GC content."""
    if not 1 <= k <= 8:
        raise CheckError("k must be between 1 and 8")
    lab = Label([t[1] for t in train] + [t[1] for t in test], task)
    n_tr = len(train)
    te_idx = np.arange(n_tr, n_tr + len(test))
    prof = np.stack([kmer_profile(t[2], k) for t in train + test])
    mu, sd = prof[:n_tr].mean(0), prof[:n_tr].std(0) + 1e-12
    z = (prof - mu) / sd
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    gc = np.array([gc_content(t[2]) for t in train + test])
    nn = min(n_neighbors, n_tr)
    y_tr = lab.y[:n_tr]

    def predict(dist: np.ndarray) -> np.ndarray:  # (n_test, n_train) distances -> predictions
        idx = np.argsort(dist, axis=1)[:, :nn]
        if lab.task == "multiclass":
            return np.array([Counter(y_tr[i]).most_common(1)[0][0] for i in idx], dtype=object)
        return np.nanmean(y_tr[idx].astype(float), axis=1)

    k_pred = predict(1 - z[n_tr:] @ z[:n_tr].T)
    g_pred = predict(np.abs(gc[n_tr:, None] - gc[None, :n_tr]))
    chance = Label(lab.values[n_tr:], lab.task).chance() if lab.task == "multiclass" else lab.chance()
    res = {
        "task": lab.task,
        "metric": lab.metric,
        "chance": _r(chance),
        "model_score": model_score,
        "gc_only": {"score": _r(lab.score(te_idx, g_pred))},
        f"kmer_{k}": {"score": _r(lab.score(te_idx, k_pred))},
        "how": f"{nn}-nearest training sequences by GC content, and by cosine distance of z-scored canonical "
        f"{k}-mer frequencies; majority label (classification) or mean (regression)",
    }
    for key in ("gc_only", f"kmer_{k}"):
        res[key]["share_of_model_gain"] = _share(res[key]["score"] or float("nan"), chance, model_score)
    best = max(("gc_only", f"kmer_{k}"), key=lambda s: res[s]["score"] if res[s]["score"] is not None else -1e9)
    parts = [
        f"Composition alone reaches {lab.metric} {res[best]['score']} ({best.replace('_', ' ')}; chance {chance:.3g})."
    ]
    share = res[best]["share_of_model_gain"]
    if share is not None:
        parts.append(f"That is {share:.0%} of the model's gain over chance.")
        if share >= SHARE_FLAG:
            parts.append(
                "The model adds little beyond sequence composition (class B; see F01, F02, F08). A control that "
                "keeps composition (e.g. token shuffling) is not a null."
            )
    res["summary"] = " ".join(parts)
    return res
