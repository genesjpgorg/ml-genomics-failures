"""Generate the pilot eval tasks: small analyses with a planted flaw (or a planted absence of one).

usage: evals/.venv/bin/python evals/make_tasks.py

Each task is evals/tasks/<id>/ with
  workspace/   what the agent sees: README.md (the write-up and its claim) plus data and results files
  rubric.json  what only the judge sees: whether the claim is flawed, the mechanism, what to quantify, the fix

Domains are disguised (protein interactions, splice variants, proteomics, phage contigs...) so an agent can't
match a task to a famous paper, and so the book's cases don't contain the answer verbatim. T08 is a flaw the book
has no case for (feature selection before cross-validation). Every planted property is verified with the
package's own checks before the task is written, so the ground truth is measured, not assumed.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from ml_genomics_failures import checks

TASKS = ROOT / "tasks"


# --- helpers -------------------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_fasta(path: Path, seqs: dict[str, str]) -> None:
    path.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))


def task(tid: str, readme: str, rubric: dict) -> Path:
    d = TASKS / tid
    if d.exists():
        shutil.rmtree(d)
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "README.md").write_text(readme.strip() + "\n")
    (d / "rubric.json").write_text(json.dumps({"id": tid, **rubric}, indent=2) + "\n")
    return d / "workspace"


def rand_seq(rng: random.Random, n: int, gc: float = 0.5) -> str:
    return "".join(rng.choices("GCAT", weights=[gc / 2, gc / 2, (1 - gc) / 2, (1 - gc) / 2], k=n))


def mutate(rng: random.Random, s: str, rate: float) -> str:
    return "".join(rng.choice("ACGT".replace(c, "")) if rng.random() < rate else c for c in s)


def r3(x: float) -> float:
    return round(float(x), 3)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"planted property failed: {msg}")


# --- T01: protein-protein interactions, random pair split (leakage, cf. F03) --------------------------------


def t01() -> None:
    rng = np.random.default_rng(1)
    prop = {f"P{i:03d}": rng.beta(0.6, 1.2) for i in range(60)}
    names = sorted(prop)
    pairs = set()
    while len(pairs) < 700:
        a, b = sorted(rng.choice(names, 2, replace=False))
        pairs.add((a, b))
    rows = [
        {"protein_a": a, "protein_b": b, "label": int(rng.random() < min(1, 1.8 * prop[a] * prop[b]))}
        for a, b in sorted(pairs)
    ]
    idx = rng.permutation(len(rows))
    train = [rows[i] for i in idx[:560]]
    test = [rows[i] for i in idx[560:]]
    mem = checks.split_overlap(train, test, ["protein_a", "protein_b"], label="label")
    # the "model": memorized per-protein interaction rates plus noise (what a pair model on a leaky split learns)
    rate: dict[str, list[int]] = {}
    for r in train:
        for p in (r["protein_a"], r["protein_b"]):
            rate.setdefault(p, []).append(r["label"])
    score = [
        np.mean([np.mean(rate[r["protein_a"]]), np.mean(rate[r["protein_b"]])]) + rng.normal(0, 0.03) for r in test
    ]
    auc = checks.auroc(np.array([r["label"] for r in test]), np.array(score))
    check(mem["test_rows_with_any_seen_entity"] > 0.95, "T01 overlap")
    check(abs(mem["entity_memorization_baseline"]["score"] - auc) < 0.05, "T01 memorization ~ model")
    ws = task(
        "T01",
        f"""
# PairNet: predicting protein–protein interactions from sequence

We trained PairNet, a siamese transformer over ESM-style protein sequence embeddings, to predict whether two
yeast proteins physically interact (label 1 = interaction in our curated screen, 0 = tested, no interaction).

- Data: 700 tested protein pairs over 60 proteins (`pairs_train.csv`, `pairs_test.csv`).
- Split: pairs were randomly assigned to train (80%) and test (20%).
- Result: on the held-out test pairs PairNet reaches **AUROC {auc:.2f}** (`predictions_test.csv`).

**Claim:** PairNet has learned sequence determinants of physical interaction and generalizes to new protein pairs,
so we will use it to predict interaction partners for uncharacterized proteins.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "L (leakage through shared entities)",
            "mechanism": "Pairs were split at random, so every test pair's proteins also appear in training. Labels "
            "depend only on per-protein interaction propensity, so memorizing each protein's training interaction "
            "rate reproduces the reported AUROC; nothing pair-specific is shown, and nothing about new proteins.",
            "quantify": [
                f"about {mem['test_rows_with_any_seen_entity']:.0%} of test pairs share a protein with training",
                (
                    f"a per-protein label-rate baseline with no sequence information reaches AUROC "
                    f"~{mem['entity_memorization_baseline']['score']:.2f}, about the model's {auc:.2f}"
                ),
            ],
            "fix": "Protein-disjoint split (no test protein in training), or report performance separately for "
            "pairs with 0, 1 or 2 proteins seen in training; compare with a per-protein degree baseline.",
            "book_case": "F03 / F04 (disguised)",
        },
    )
    write_csv(ws / "pairs_train.csv", train)
    write_csv(ws / "pairs_test.csv", test)
    write_csv(
        ws / "predictions_test.csv",
        [{"protein_a": r["protein_a"], "protein_b": r["protein_b"], "score": f"{s:.4f}"} for r, s in zip(test, score)],
    )


# --- C01: protein-protein interactions, protein-disjoint split (clean) --------------------------------------


def c01() -> None:
    rng = np.random.default_rng(11)
    names = [f"P{i:03d}" for i in range(80)]
    tr_p, te_p = names[:60], names[60:]

    def pairs(pool, n):
        out = set()
        while len(out) < n:
            a, b = sorted(rng.choice(pool, 2, replace=False))
            out.add((a, b))
        rows = []
        for a, b in sorted(out):
            coexp = rng.normal()
            label = int(rng.random() < 1 / (1 + np.exp(-(2.2 * coexp - 0.8))))
            rows.append({"protein_a": a, "protein_b": b, "coexpression_z": f"{coexp:.3f}", "label": label})
        return rows

    train, test = pairs(tr_p, 600), pairs(te_p, 150)
    score = [1 / (1 + np.exp(-(2.0 * float(r["coexpression_z"]) - 0.7 + rng.normal(0, 0.6)))) for r in test]
    auc = checks.auroc(np.array([r["label"] for r in test]), np.array(score))
    ov = checks.split_overlap(train, test, ["protein_a", "protein_b"], label="label")
    check(ov["test_rows_with_any_seen_entity"] == 0, "C01 disjoint")
    sh = checks.shortcuts(test, "label", ["coexpression_z"])
    ws = task(
        "C01",
        f"""
# Predicting protein–protein interactions from co-expression

We predict whether two yeast proteins physically interact from a pair-level feature: the z-scored co-expression of
their genes across 300 conditions (`coexpression_z`), using logistic regression.

- Data: `pairs_train.csv` (600 pairs among 60 proteins), `pairs_test.csv` (150 pairs among 20 other proteins).
- Split: protein-disjoint — no protein in the test set appears in any training pair.
- Result: **AUROC {auc:.2f}** on the test pairs (`predictions_test.csv`).

**Claim:** co-expression predicts physical interaction for pairs of proteins never seen in training
(AUROC {auc:.2f}), so it is a useful prior for ranking candidate interaction partners of uncharacterized proteins.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": "Clean control. The split is protein-disjoint (0% overlap), the signal is pair-level "
            f"(co-expression alone gives AUROC ~{sh['features'][0]['score']:.2f}), and the claim is modest.",
            "false_alarms": [
                "claiming proteins leak between train and test",
                "calling the result invalid because a single feature does the work (that is the claim)",
            ],
            "acceptable_caveats": ["co-expression is not causal", "single dataset / organism", "sample size"],
        },
    )
    write_csv(ws / "pairs_train.csv", train)
    write_csv(ws / "pairs_test.csv", test)
    write_csv(
        ws / "predictions_test.csv",
        [{"protein_a": r["protein_a"], "protein_b": r["protein_b"], "score": f"{s:.4f}"} for r, s in zip(test, score)],
    )


# --- T02: splice-variant classifier, gene-level label homogeneity (circularity, cf. F09) -------------------


def t02() -> None:
    rng = np.random.default_rng(2)
    genes = [f"GENE{i:02d}" for i in range(80)]
    gene_rate = {g: (0.92 if rng.random() < 0.5 else 0.06) for g in genes}
    rows = []
    for g in genes:
        for j in range(int(rng.integers(8, 20))):
            y = int(rng.random() < gene_rate[g])
            cons = rng.normal(0.25 * y, 1.0)  # a weak real variant-level signal
            rows.append({"variant_id": f"{g}_v{j}", "gene": g, "conservation": f"{cons:.3f}", "label": y})
    idx = rng.permutation(len(rows))
    n_test = len(rows) // 5
    for k, i in enumerate(idx):
        rows[i]["split"] = "test" if k < n_test else "train"
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    rate: dict[str, list[int]] = {}
    for r in train:
        rate.setdefault(r["gene"], []).append(r["label"])
    for r in rows:
        r["model_score"] = (
            f"{np.mean(rate.get(r['gene'], [0.5])) + 0.05 * float(r['conservation']) + rng.normal(0, 0.03):.4f}"
        )
    auc = checks.auroc(np.array([r["label"] for r in test]), np.array([float(r["model_score"]) for r in test]))
    mem = checks.split_overlap(train, test, ["gene"], label="label")
    cons_auc = checks.auroc(np.array([r["label"] for r in test]), np.array([float(r["conservation"]) for r in test]))
    check(mem["entity_memorization_baseline"]["score"] > auc - 0.04, "T02 gene memorization ~ model")
    ws = task(
        "T02",
        f"""
# SpliceScore: classifying splice-disrupting variants

SpliceScore is a gradient-boosted model over 140 variant-level features (conservation, predicted splice-site
strength, distance to exon boundary, ...) that classifies rare variants as splice-disrupting (1) or not (0), with
labels from a minigene assay compendium.

- Data: `variants.csv`: variant id, gene, one representative feature (`conservation`), assay label, split, and the
  model's test-time score. The full feature matrix is too large to share here.
- Split: variants randomly assigned to train (80%) and test (20%).
- Result: **AUROC {auc:.2f}** on test variants.

**Claim:** SpliceScore learns variant-level determinants of splice disruption and should be used to prioritize
variants of uncertain significance in diagnostic pipelines.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "C/L (gene-level circularity)",
            "mechanism": "Labels are almost uniform within each gene (genes are mostly all-disrupting or all-benign) "
            "and the split is by variant, so test variants come from training genes. Predicting a variant from its "
            "gene's training label rate reproduces the AUROC; that says nothing about which variant in a gene "
            "disrupts splicing.",
            "quantify": [
                "labels are near-homogeneous within genes",
                (
                    f"a gene label-rate baseline reaches AUROC ~{mem['entity_memorization_baseline']['score']:.2f} "
                    f"vs the model's {auc:.2f}"
                ),
                f"the variant-level feature alone gives only ~{cons_auc:.2f}",
            ],
            "fix": "Gene-level split, or evaluate within genes that have both labels; report a gene-prior baseline.",
            "book_case": "F09 (disguised)",
        },
    )
    write_csv(ws / "variants.csv", rows)


# --- T03 / C02: proteomics case-control with plates (batch confounding, cf. F10) ----------------------------


def proteomics(seed: int, confounded: bool) -> tuple[list[dict], list[dict], dict]:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n, p = 120, 200
    status = np.array([1] * 60 + [0] * 60)
    if confounded:  # cases mostly run on plates 1-2, controls on 3-4
        plate = np.where(rng.random(n) < 0.85, np.where(status == 1, 1, 3), np.where(status == 1, 3, 1))
        plate = plate + rng.integers(0, 2, n)
    else:  # balanced: each plate gets 15 cases and 15 controls
        plate = np.concatenate([np.repeat(np.arange(1, 5), 15), np.repeat(np.arange(1, 5), 15)])
    plate_effect = rng.normal(0, 1.0, (5, p))
    x = rng.normal(0, 1, (n, p)) + plate_effect[plate] * 0.9
    if not confounded:
        x[status == 1, :3] += 1.0  # a real 3-protein signal
    model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=12), LogisticRegression(max_iter=1000))
    prob = cross_val_predict(
        model, x, status, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba"
    )[:, 1]
    auc = checks.auroc(status, prob)
    dates = {1: "2024-03-04", 2: "2024-03-05", 3: "2024-04-22", 4: "2024-04-23"}
    samples = [
        {
            "sample_id": f"S{i:03d}",
            "status": "case" if status[i] else "control",
            "plate": int(plate[i]),
            "run_date": dates[int(plate[i])],
        }
        for i in range(n)
    ]
    matrix = [{"sample_id": f"S{i:03d}", **{f"prot{j:03d}": f"{x[i, j]:.3f}" for j in range(p)}} for i in range(n)]
    sel = SelectKBest(f_classif, k=12).fit(x, status).get_support(indices=True)
    plate_only = checks.shortcuts(samples, "status", ["plate", "run_date"])
    return samples, matrix, {"auc": auc, "signature": [f"prot{j:03d}" for j in sel], "plate_only": plate_only}


def t03() -> None:
    samples, matrix, r = proteomics(3, confounded=True)
    plate_auc = r["plate_only"]["features"][0]["score"]
    check(plate_auc > 0.8 and r["auc"] > 0.8, "T03 plate confounded and model high")
    ws = task(
        "T03",
        f"""
# A 12-protein plasma signature for early pancreatic cancer

We profiled 200 plasma proteins by targeted mass spectrometry in 60 early-stage pancreatic cancer cases and 60
matched controls, and trained a logistic-regression classifier on the 12 most discriminative proteins.

- Data: `samples.csv` (sample, case/control status, plate, run date), `protein_matrix.csv` (log-intensities).
- Evaluation: 5-fold stratified cross-validation over the 120 samples.
- Result: **cross-validated AUROC {r["auc"]:.2f}** (`results.json`).

**Claim:** the 12-protein signature detects early pancreatic cancer from plasma and is ready for validation as a
screening test.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "S (batch confounding)",
            "mechanism": "Cases were mostly run on plates 1-2 (March) and controls on plates 3-4 (April). Plates "
            "shift every protein, and there is no real disease signal in this data, so the classifier learns the "
            "plate. (Feature selection on all samples before CV also inflates the AUROC.)",
            "quantify": [
                f"plate or run date alone predicts case status with AUROC ~{plate_auc:.2f}",
                "~85% of cases on plates 1-2 vs ~15% of controls",
            ],
            "fix": "Randomize cases and controls across plates, analyse within plate or correct for plate inside CV "
            "folds, and validate on an independently processed cohort.",
            "book_case": "F10 (disguised)",
        },
    )
    write_csv(ws / "samples.csv", samples)
    write_csv(ws / "protein_matrix.csv", matrix)
    (ws / "results.json").write_text(
        json.dumps({"cv_auroc": r3(r["auc"]), "signature": r["signature"], "cv": "5-fold stratified"}, indent=2)
    )


def c02() -> None:
    samples, matrix, r = proteomics(12, confounded=False)
    plate_auc = r["plate_only"]["features"][0]["score"]
    check(plate_auc < 0.6 and r["auc"] > 0.7, "C02 balanced and real signal")
    ws = task(
        "C02",
        f"""
# Plasma proteins associated with sepsis-induced kidney injury

We measured 200 plasma proteins by targeted mass spectrometry in 60 sepsis patients who developed acute kidney
injury (cases) and 60 who did not (controls). Samples were randomized across 4 plates, 15 cases and 15 controls per
plate, before processing.

- Data: `samples.csv` (status, plate, run date), `protein_matrix.csv` (log-intensities).
- Evaluation: 5-fold stratified cross-validation; standardization, selection of 12 proteins and logistic regression
  are fitted inside each training fold.
- Result: **cross-validated AUROC {r["auc"]:.2f}** (`results.json`).

**Claim:** plasma proteins measured at sepsis onset distinguish patients who go on to develop kidney injury
(cross-validated AUROC {r["auc"]:.2f}); the signal is worth testing in an external cohort.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": f"Clean control. Plates are balanced (plate alone AUROC ~{plate_auc:.2f}), selection is "
            "inside CV folds, three proteins carry a real shift, and the claim asks for external validation.",
            "false_alarms": ["claiming plate/batch confounding", "claiming feature-selection leakage"],
            "acceptable_caveats": [
                "needs external validation",
                "modest sample size",
                "plate effects exist but are balanced",
            ],
        },
    )
    write_csv(ws / "samples.csv", samples)
    write_csv(ws / "protein_matrix.csv", matrix)
    (ws / "results.json").write_text(
        json.dumps({"cv_auroc": r3(r["auc"]), "cv": "5-fold stratified, selection inside folds"}, indent=2)
    )
    (ws / "analysis.py").write_text(
        """import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

s = pd.read_csv("samples.csv"); x = pd.read_csv("protein_matrix.csv").set_index("sample_id").loc[s.sample_id].values
y = (s.status == "case").astype(int).values
model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=12), LogisticRegression(max_iter=1000))
p = cross_val_predict(model, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("cv AUROC", round(roc_auc_score(y, p), 3))
"""
    )


# --- T04: promoter activity with paralogous sequences across the split (cf. F12) ----------------------------


def t04() -> None:
    rng = random.Random(4)
    train = {f"prom{i:03d}": rand_seq(rng, 300) for i in range(300)}
    label = {k: rng.choice(["active", "inactive"]) for k in train}  # no learnable sequence signal
    test, pred = {}, {}
    for i in range(70):  # paralog copies of training promoters keep their label
        src = f"prom{rng.randrange(300):03d}"
        t = f"test{i:03d}"
        test[t] = mutate(rng, train[src], rng.uniform(0.01, 0.04))
        label[t] = label[src]
        pred[t] = label[src]
    for i in range(70, 100):
        t = f"test{i:03d}"
        test[t] = rand_seq(rng, 300)
        label[t] = rng.choice(["active", "inactive"])
        pred[t] = rng.choice(["active", "inactive"])
    acc = np.mean([pred[t] == label[t] for t in test])
    novel_acc = np.mean([pred[t] == label[t] for t in list(test)[70:]])
    sim = checks.sequence_similarity(train, test)
    near = sum(b["n_test"] for b in sim["by_identity"][:3])  # k-mer estimates on 300 bp spread 96-99% down to ~0.92
    check(near >= 68, "T04 near-copies detected")
    ws = task(
        "T04",
        f"""
# PromoNet: predicting promoter activity from sequence

PromoNet is a convolutional network that predicts whether a 300 bp promoter is active (drives reporter expression
above background) in our massively parallel reporter assay in HepG2 cells.

- Data: `train.fa` (300 promoters), `test.fa` (100 promoters), `labels.csv` (activity per promoter),
  `predictions_test.csv` (PromoNet's predicted class for each test promoter).
- Split: promoters were randomly sampled into train and test sets.
- Result: **test accuracy {acc:.2f}** (chance 0.50).

**Claim:** PromoNet has learned the regulatory grammar of HepG2 promoters and can predict the activity of new,
unseen promoter sequences.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "L (homology leakage)",
            "mechanism": "70 of the 100 test promoters are near-copies (96-99% identity) of training promoters "
            "with the same label; on the 30 genuinely new sequences accuracy is at chance. Labels have no "
            "learnable sequence signal at all.",
            "quantify": [
                f"~{near}/100 test sequences have a training sequence at >=90% identity (true identity 96-99%)",
                f"accuracy on near-copies ~1.0 vs ~{novel_acc:.2f} on dissimilar test promoters",
            ],
            "fix": "Cluster sequences by identity (e.g. MMseqs2/CD-HIT, <80%) and split by cluster; report accuracy "
            "by identity to the nearest training sequence.",
            "book_case": "F12 (disguised)",
        },
    )
    write_fasta(ws / "train.fa", train)
    write_fasta(ws / "test.fa", test)
    write_csv(
        ws / "labels.csv",
        [{"id": k, "split": "train" if k in train else "test", "activity": v} for k, v in label.items()],
    )
    write_csv(ws / "predictions_test.csv", [{"id": t, "predicted": pred[t]} for t in test])


# --- T05: phage vs host contigs separable by GC alone (composition, cf. F01/F08) ----------------------------


def t05() -> None:
    rng = random.Random(5)

    def make(n, prefix):
        rows = []
        for i in range(n):
            lab = "phage" if i % 2 else "bacterial"
            gc = rng.gauss(0.41 if lab == "phage" else 0.55, 0.025)
            rows.append({"id": f"{prefix}{i:03d}", "label": lab, "sequence": rand_seq(rng, 1500, gc)})
        return rows

    train, test = make(200, "tr"), make(100, "te")
    base = checks.kmer_baseline(
        [(r["id"], r["label"], r["sequence"]) for r in train], [(r["id"], r["label"], r["sequence"]) for r in test], k=4
    )
    model_acc = r3(min(0.99, base["gc_only"]["score"] + 0.01))
    check(base["gc_only"]["score"] > 0.9, "T05 GC separates")
    ws = task(
        "T05",
        f"""
# PhageFormer: identifying phage contigs in gut metagenomes

PhageFormer is a 6-layer transformer over 6-mer tokens that classifies 1.5 kb metagenomic contigs as phage or
bacterial. Attention maps highlight regions resembling phage structural and replication genes.

- Data: `train.csv` and `test.csv` (contig id, label, sequence).
- Result: **test accuracy {model_acc:.2f}** (`results.json`).

**Claim:** PhageFormer learns phage-specific gene content and sequence features, and can discover novel phages in
metagenomes where homology-based tools fail.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "B (missing composition baseline)",
            "mechanism": "Phage and bacterial contigs differ in GC content (~41% vs ~55%); GC content alone or k-mer "
            "composition classifies them almost perfectly, so the accuracy says nothing about gene content. The "
            "sequences are random apart from GC: there are no genes to learn.",
            "quantify": [
                f"GC content alone reaches accuracy ~{base['gc_only']['score']:.2f}",
                f"the model's {model_acc} is no better than the composition baseline",
            ],
            "fix": "Report GC/k-mer baselines; evaluate on GC-matched phage and host contigs; test on phages from "
            "hosts with similar GC.",
            "book_case": "F01 / F08 (disguised)",
        },
    )
    write_csv(ws / "train.csv", train)
    write_csv(ws / "test.csv", test)
    (ws / "results.json").write_text(json.dumps({"test_accuracy": model_acc, "n_test": 100}, indent=2))


# --- T06: shuffled control mistaken for evidence of grammar (cf. F02) ---------------------------------------


def t06() -> None:
    rng = random.Random(6)

    def make(n, prefix):
        rows = []
        for i in range(n):
            lab = int(i % 2)
            gc = rng.gauss(0.58 if lab else 0.45, 0.03)
            rows.append({"id": f"{prefix}{i:04d}", "bound": lab, "sequence": rand_seq(rng, 200, gc)})
        return rows

    train, test = make(600, "tr"), make(200, "te")
    base = checks.kmer_baseline(
        [(r["id"], str(r["bound"]), r["sequence"]) for r in train],
        [(r["id"], str(r["bound"]), r["sequence"]) for r in test],
        k=4,
    )
    gc_acc = base["gc_only"]["score"]
    check(gc_acc > 0.8, "T06 GC predicts binding")
    ws = task(
        "T06",
        """
# BindGrammar: a CNN learns the cis-regulatory grammar of FOXA2 binding

We trained a CNN to classify 200 bp sequences as FOXA2-bound (ChIP-seq peak summits) or unbound (random accessible
regions) in liver.

- Data: `sequences.csv` (id, split, bound, sequence).
- Result: **test AUROC 0.89**.
- Control: we retrained the same CNN on dinucleotide-shuffled versions of every sequence (labels unchanged).
  The shuffled-sequence model reaches **AUROC 0.86**. Both numbers are from one training run (seed 0);
  see `results.json`.

**Claim:** because the CNN outperforms the shuffled-sequence control, it has learned the arrangement of FOXA2
motifs and co-factor motifs (cis-regulatory grammar), not just sequence composition.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "K (control that isn't a null)",
            "mechanism": "Dinucleotide shuffling keeps composition, and composition (GC) separates bound from "
            "unbound almost as well as the CNN. A 0.03 gap from a single seed is not evidence of grammar. The "
            "sequences contain no motifs at all: bound sequences are just GC-richer.",
            "quantify": [
                f"GC content alone gives accuracy ~{gc_acc:.2f} on the test set",
                "the shuffled control (0.86) keeps nearly all of the performance; the 0.03 gap has no error bar",
            ],
            "fix": "Use GC-matched negatives, report GC/k-mer baselines and multiple seeds, add a true null "
            "(permuted labels), and test grammar directly (motif insertion/rearrangement experiments).",
            "book_case": "F02 (disguised)",
        },
    )
    rows = [{**r, "split": "train"} for r in train] + [{**r, "split": "test"} for r in test]
    write_csv(
        ws / "sequences.csv",
        [{"id": r["id"], "split": r["split"], "bound": r["bound"], "sequence": r["sequence"]} for r in rows],
    )
    (ws / "results.json").write_text(
        json.dumps({"cnn_test_auroc": 0.89, "shuffled_control_test_auroc": 0.86, "seeds": [0]}, indent=2)
    )


# --- T07 / C04: sequence-to-expression across genes vs across individuals (cf. F05) -------------------------


def expression_tables(seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    obs = rng.normal(5, 2, 2000)
    pred = obs * 0.8 + rng.normal(0, 1.2, 2000)
    chroms = rng.choice(["chr8", "chr9", "chr10"], 2000)
    genes = [
        {
            "gene": f"G{i:04d}",
            "test_chromosome": chroms[i],
            "predicted_log_tpm": f"{pred[i]:.3f}",
            "observed_log_tpm": f"{obs[i]:.3f}",
        }
        for i in range(2000)
    ]
    eff_obs = rng.normal(0, 0.4, 300)
    eff_pred = rng.normal(0, 0.05, 300) + 0.02 * eff_obs
    variants = [
        {
            "variant": f"rs{100000 + i}",
            "gene": f"G{rng.integers(0, 2000):04d}",
            "predicted_effect": f"{eff_pred[i]:.4f}",
            "observed_eqtl_beta": f"{eff_obs[i]:.4f}",
        }
        for i in range(300)
    ]
    stats = {
        "gene_r": float(np.corrcoef(pred, obs)[0, 1]),
        "var_r": float(np.corrcoef(eff_pred, eff_obs)[0, 1]),
        "sign": float(np.mean(np.sign(eff_pred) == np.sign(eff_obs))),
    }
    return genes, variants, stats


def t07() -> None:
    genes, variants, s = expression_tables(7)
    check(s["gene_r"] > 0.7 and abs(s["var_r"]) < 0.2, "T07 axes differ")
    ws = task(
        "T07",
        f"""
# ExprNet for personalized regulatory variant interpretation

ExprNet predicts gene expression (log TPM, lymphoblastoid cells) from 100 kb of reference DNA sequence around each
gene's TSS.

- Evaluation on held-out chromosomes 8–10: predicted vs observed expression across 2,000 genes gives
  **Pearson r = {s["gene_r"]:.2f}** (`gene_level_eval.csv`).
- We also scored 300 common variants by in-silico mutagenesis and include the comparison with eQTL effect sizes
  from a 400-individual cohort (`variant_eval.csv`).

**Claim:** ExprNet accurately predicts expression from sequence (r = {s["gene_r"]:.2f}), so we will use it to
predict the expression impact of each patient's personal regulatory variants in our rare-disease cohort.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "A (wrong axis of generalization)",
            "mechanism": "r = 0.8 is across genes on the reference genome; the intended use is across individuals "
            "(effects of personal variants). The included variant table shows the model does not predict eQTL "
            "effects.",
            "quantify": [
                f"variant-level correlation with eQTL betas r ~{s['var_r']:.2f}",
                f"sign agreement ~{s['sign']:.0%} (chance 50%)",
            ],
            "fix": "Validate on the axis of use: per-gene predictions across individuals (paired genomes and "
            "transcriptomes), including the sign of effects; don't use across-gene r to justify variant calls.",
            "book_case": "F05 (disguised)",
        },
    )
    write_csv(ws / "gene_level_eval.csv", genes)
    write_csv(ws / "variant_eval.csv", variants)


def c04() -> None:
    genes, _, s = expression_tables(14)
    ws = task(
        "C04",
        f"""
# ExprNet: predicting baseline expression levels from reference sequence

ExprNet predicts gene expression (log TPM, lymphoblastoid cells) from 100 kb of reference DNA sequence around each
gene's TSS. Training used genes on all chromosomes except 8–10.

- Evaluation on held-out chromosomes 8–10: predicted vs observed expression across 2,000 genes gives
  **Pearson r = {s["gene_r"]:.2f}** (`gene_level_eval.csv`).

**Claim:** reference sequence near the TSS explains much of the variation in baseline expression **between genes**
in this cell type (r = {s["gene_r"]:.2f} on held-out chromosomes). We make no claim about predicting differences
between individuals or the effects of genetic variants, which this evaluation does not test.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": "Clean control. The claim is about the axis that was tested (across genes, held-out "
            "chromosomes) and explicitly excludes variant effects.",
            "false_alarms": [
                "calling it unsupported because it doesn't predict individual variant effects (not claimed)"
            ],
            "acceptable_caveats": [
                "paralogs across chromosomes could leak",
                "one cell type",
                "no comparison with simpler baselines (e.g. CpG/GC content)",
            ],
        },
    )
    write_csv(ws / "gene_level_eval.csv", genes)


# --- C03: motif order classifier with a proper split and controls (clean) -----------------------------------


def c03() -> None:
    rng = random.Random(13)
    a, b = "TGACTCAG", "CACGTGAC"

    def make(n, prefix):
        rows = []
        for i in range(n):
            lab = i % 2
            s = list(rand_seq(rng, 200))
            p1 = rng.randrange(10, 80)
            p2 = rng.randrange(110, 180)
            first, second = (a, b) if lab else (b, a)
            s[p1 : p1 + 8] = first
            s[p2 : p2 + 8] = second
            rows.append({"id": f"{prefix}{i:04d}", "order_ab": lab, "sequence": "".join(s)})
        return rows

    train, test = make(800, "tr"), make(200, "te")
    base = checks.kmer_baseline(
        [(r["id"], str(r["order_ab"]), r["sequence"]) for r in train],
        [(r["id"], str(r["order_ab"]), r["sequence"]) for r in test],
        k=4,
    )
    sim = checks.sequence_similarity({r["id"]: r["sequence"] for r in train}, {r["id"]: r["sequence"] for r in test})
    check(
        abs(base["kmer_4"]["score"] - 0.5) < 0.12 and abs(base["gc_only"]["score"] - 0.5) < 0.12,
        "C03 composition uninformative",
    )
    check(sim["by_identity"][-1]["n_test"] == 200, "C03 no similar sequences")
    pred = [r["order_ab"] if rng.random() < 0.9 else 1 - r["order_ab"] for r in test]
    acc = np.mean([p == r["order_ab"] for p, r in zip(pred, test)])
    ws = task(
        "C03",
        f"""
# A CNN detects the order of AP-1 and E-box motifs in synthetic enhancers

In a synthetic enhancer library, every 200 bp sequence carries one AP-1 site (TGACTCAG) and one E-box (CACGTGAC);
only the order differs (AP-1 upstream = class 1, E-box upstream = class 0). Background sequence is random, so both
classes have the same composition. We trained a CNN to predict the order class.

- Data: `sequences.csv` (id, split, order_ab, sequence); `predictions_test.csv`.
- Split: independently synthesized sequences; the most similar train/test pair shares < 40% identity outside the motifs.
- Result: **test accuracy {acc:.2f}**; mean over 3 seeds 0.90 ± 0.01.
- Controls: dinucleotide-shuffled sequences (motifs destroyed) give 0.51; permuted labels give 0.50.

**Claim:** the CNN learns motif order: it distinguishes sequences that differ only in the arrangement of the same
two motifs.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": f"Clean control. Composition is identical between classes (GC-only ~{base['gc_only']['score']:.2f}, "
            f"4-mer ~{base['kmer_4']['score']:.2f}), no near-duplicate sequences, three seeds, a true null and a "
            "control that destroys the motifs.",
            "false_alarms": [
                "claiming a composition shortcut",
                "claiming sequence leakage",
                "claiming the controls are invalid",
            ],
            "acceptable_caveats": ["synthetic library may not transfer to genomic enhancers", "one motif pair"],
        },
    )
    rows = [{**r, "split": "train"} for r in train] + [{**r, "split": "test"} for r in test]
    write_csv(
        ws / "sequences.csv",
        [{"id": r["id"], "split": r["split"], "order_ab": r["order_ab"], "sequence": r["sequence"]} for r in rows],
    )
    write_csv(ws / "predictions_test.csv", [{"id": r["id"], "predicted": p} for r, p in zip(test, pred)])


# --- T08: feature selection before cross-validation (not in the book) --------------------------------------


def t08() -> None:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(8)
    n, p = 60, 5000
    x = rng.normal(0, 1, (n, p))
    y = np.array([1] * 30 + [0] * 30)
    rng.shuffle(y)  # labels unrelated to the data
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    sel = SelectKBest(f_classif, k=20).fit(x, y).get_support(indices=True)
    leaky = cross_val_predict(LogisticRegression(max_iter=1000), x[:, sel], y, cv=cv, method="predict_proba")[:, 1]
    proper = cross_val_predict(
        make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
        x,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    leaky_auc, proper_auc = checks.auroc(y, leaky), checks.auroc(y, proper)
    check(leaky_auc > 0.85 and proper_auc < 0.7, "T08 leaky selection inflates")
    ws = task(
        "T08",
        f"""
# A 20-gene expression signature predicts immunotherapy response in melanoma

From bulk RNA-seq of 60 pre-treatment melanoma biopsies (5,000 most variable genes), we derived a 20-gene
signature that predicts response to anti-PD-1 therapy.

- Data: `expression.csv` (patient × gene, z-scored), `response.csv` (responder 1/0).
- Analysis: `analysis.py` (selects the 20 genes most associated with response, then evaluates a logistic regression
  by 5-fold stratified cross-validation).
- Result: **cross-validated AUROC {leaky_auc:.2f}** (`results.json`).

**Claim:** the 20-gene signature predicts anti-PD-1 response (CV AUROC {leaky_auc:.2f}) and should be tested
prospectively to select patients for therapy.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "leaky preprocessing (not in the book's cases)",
            "mechanism": "The 20 genes are selected using all 60 samples, including each fold's test samples, "
            "before cross-validation. With 5,000 features and 60 samples this alone produces a high AUROC; the "
            "labels here are random.",
            "quantify": [f"with selection inside the CV folds the AUROC drops to ~{proper_auc:.2f}"],
            "fix": "Put feature selection inside the cross-validation (nested CV / pipeline), and validate on an "
            "independent cohort.",
            "book_case": None,
        },
    )
    genes = [f"gene{j:04d}" for j in range(p)]
    write_csv(
        ws / "expression.csv",
        [{"patient": f"pt{i:02d}", **{g: f"{x[i, j]:.3f}" for j, g in enumerate(genes)}} for i in range(n)],
    )
    write_csv(ws / "response.csv", [{"patient": f"pt{i:02d}", "responder": int(y[i])} for i in range(n)])
    (ws / "results.json").write_text(
        json.dumps({"cv_auroc": r3(leaky_auc), "signature": [genes[j] for j in sel]}, indent=2)
    )
    (ws / "analysis.py").write_text(
        """import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

x = pd.read_csv("expression.csv").set_index("patient")
y = pd.read_csv("response.csv").set_index("patient").loc[x.index, "responder"].values

# pick the 20 genes most associated with response
selector = SelectKBest(f_classif, k=20).fit(x.values, y)
signature = x.columns[selector.get_support()]

# evaluate the signature by 5-fold cross-validation
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(LogisticRegression(max_iter=1000), x[signature].values, y, cv=cv, method="predict_proba")[:, 1]
print("cv AUROC", round(roc_auc_score(y, p), 3))
"""
    )


GENERATORS = [t01, t02, t03, t04, t05, t06, t07, t08, c01, c02, c03, c04]

if __name__ == "__main__":
    for g in GENERATORS:
        g()
        print("ok", g.__name__.upper())
    print(f"{len(GENERATORS)} tasks in {TASKS.relative_to(ROOT.parent)}")
