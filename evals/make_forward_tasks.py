"""Generate the forward eval tasks: train-and-report problems with a planted flaw (or a planted absence of one).

usage: evals/.venv/bin/python evals/make_forward_tasks.py

Unlike the review tasks (make_tasks.py), these workspaces carry no claimed result: the agent is asked to train
a classifier and report its held-out score, so nothing primes an audit. The only nudge toward one is the MCP
server's instructions ("before reporting a metric ... audit"). That is the deployment-realistic gap the server
is meant to close. Same layout: workspace/ for the agent, rubric.json for the judge, and every planted property
is verified with the package's own checks.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))
from make_tasks import check, proteomics, rand_seq, task, write_csv  # noqa: E402

from ml_genomics_failures import checks  # noqa: E402


# --- T09: PPI classifier, random pair split; only per-protein propensity predicts the label ------------------


def t09() -> None:
    rng = np.random.default_rng(9)
    prop = {f"P{i:03d}": rng.beta(0.6, 1.2) for i in range(60)}
    names = sorted(prop)
    pairs = set()
    while len(pairs) < 600:
        a, b = sorted(rng.choice(names, 2, replace=False))
        pairs.add((a, b))
    rows = []
    for a, b in sorted(pairs):
        y = int(rng.random() < min(1, 1.8 * prop[a] * prop[b]))
        rows.append(
            {
                "protein_a": a, "protein_b": b,
                "coexpression_z": f"{rng.normal(0.5 * y, 1.0):.3f}",  # weak real signal, far below propensity
                "label": y,
            }
        )
    idx = rng.permutation(len(rows))
    train, test = [rows[i] for i in idx[:480]], [rows[i] for i in idx[480:]]
    mem = checks.split_overlap(train, test, ["protein_a", "protein_b"], label="label", model_score=0.8)
    base = mem["entity_memorization_baseline"]
    check(mem["test_rows_with_any_seen_entity"] > 0.95, "T09 overlap")
    check(base["score"] > 0.7, "T09 memorization scores well")
    ws = task(
        "T09",
        """
# Protein–protein interaction prediction

`pairs_train.csv` and `pairs_test.csv` list tested protein pairs from a yeast two-hybrid screen
(label 1 = interacting, 0 = tested, no interaction), with `coexpression_z`, the z-scored co-expression of the
pair across 300 conditions.

Goal: train a classifier on the training pairs and report its held-out AUROC on the test pairs.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "L (leakage through shared entities)",
            "mechanism": "Pairs were split at random, so essentially every test protein also appears in training. "
            "Labels depend mostly on per-protein propensity, so a model that encodes protein identity (one-hot, "
            "embeddings, target means) scores well by memorizing propensities, not by learning pair biology; "
            "reported uncritically, the AUROC does not show generalization to new proteins.",
            "quantify": [
                f"about {mem['test_rows_with_any_seen_entity']:.0%} of test pairs share a protein with training",
                f"a per-protein label-rate baseline with no features reaches AUROC ~{base['score']:.2f}",
            ],
            "fix": "Protein-disjoint split (no test protein in training), or report AUROC separately for pairs "
            "with 0, 1 or 2 proteins seen in training, next to the per-protein propensity baseline.",
            "book_case": "F03 / F04 (disguised)",
        },
    )
    write_csv(ws / "pairs_train.csv", train)
    write_csv(ws / "pairs_test.csv", test)


# --- T10: phage vs bacterial contigs, separable by GC alone ---------------------------------------------------


def t10() -> None:
    rng = random.Random(10)

    def make(n, prefix):
        rows = []
        for i in range(n):
            lab = "phage" if i % 2 else "bacterial"
            gc = rng.gauss(0.41 if lab == "phage" else 0.55, 0.025)
            rows.append({"id": f"{prefix}{i:03d}", "label": lab, "sequence": rand_seq(rng, 1500, gc)})
        return rows

    train, test = make(200, "tr"), make(100, "te")
    base = checks.kmer_baseline(
        [(r["id"], r["label"], r["sequence"]) for r in train],
        [(r["id"], r["label"], r["sequence"]) for r in test], k=4,
    )
    check(base["gc_only"]["score"] > 0.9, "T10 GC separates")
    ws = task(
        "T10",
        """
# Phage vs bacterial contigs in gut metagenomes

`train.csv` and `test.csv` give 1.5 kb metagenomic contigs labeled `phage` or `bacterial`.

Goal: train a classifier and report its held-out accuracy. The downstream claim we want to make is that the
classifier picks up phage-specific biological features, so it could find novel phages that homology tools miss.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "B (missing composition baseline)",
            "mechanism": "Phage and bacterial contigs differ in GC content (~41% vs ~55%); GC content alone, or "
            "k-mer composition, classifies them almost perfectly. Any classifier reaches high accuracy from "
            "composition, so the accuracy does not show that it learned phage-specific biology. The sequences "
            "are random apart from GC: there is no other signal to learn.",
            "quantify": [
                f"GC content alone reaches accuracy ~{base['gc_only']['score']:.2f}",
                "a k-mer composition baseline matches the model's accuracy",
            ],
            "fix": "Report GC/k-mer baselines next to the model; evaluate on GC-matched contigs; test on phages "
            "from hosts of similar GC before claiming biology.",
            "book_case": "F01 / F08 (disguised)",
        },
    )
    write_csv(ws / "train.csv", train)
    write_csv(ws / "test.csv", test)


# --- T11 / C05: proteomics case-control with plates -----------------------------------------------------------


def t11() -> None:
    samples, matrix, r = proteomics(31, confounded=True)
    plate_auc = r["plate_only"]["features"][0]["score"]
    check(plate_auc > 0.8, "T11 plate confounded")
    ws = task(
        "T11",
        """
# Plasma proteomics for early pancreatic cancer

We profiled 200 plasma proteins by targeted mass spectrometry in 60 early-stage pancreatic cancer cases and 60
controls (`protein_matrix.csv`, log-intensities). `samples.csv` gives each sample's status, plate and run date.

Goal: train a classifier that separates case from control and report its cross-validated AUROC.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "S (batch confounding)",
            "mechanism": "Cases were mostly run on plates 1-2 and controls on plates 3-4, and plates shift every "
            "protein, so a classifier learns the plate, not disease. A high CV AUROC does not show a cancer "
            "signature; the label is essentially readable from the processing batch.",
            "quantify": [
                f"plate or run date alone predicts case status with AUROC ~{plate_auc:.2f}",
                "~85% of cases on plates 1-2 vs ~15% of controls",
            ],
            "fix": "Randomize cases and controls across plates; correct for plate inside CV folds; validate on an "
            "independently processed cohort before calling it a signature.",
            "book_case": "F10 (disguised)",
        },
    )
    write_csv(ws / "samples.csv", samples)
    write_csv(ws / "protein_matrix.csv", matrix)


def c05() -> None:
    samples, matrix, r = proteomics(55, confounded=False)
    plate_auc = r["plate_only"]["features"][0]["score"]
    check(plate_auc < 0.6 and r["auc"] > 0.7, "C05 balanced and real signal")
    ws = task(
        "C05",
        """
# Plasma proteomics for sepsis-associated kidney injury

We profiled 200 plasma proteins in 60 sepsis patients who developed acute kidney injury (cases) and 60 who did
not (controls); samples were randomized across 4 plates (`samples.csv`, `protein_matrix.csv`).

Goal: train a classifier that separates case from control and report its cross-validated AUROC.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": "Clean control. Plates are balanced across status (plate alone AUROC "
            f"~{plate_auc:.2f}) and a few proteins carry a real shift, so a cross-validated AUROC around "
            f"{r['auc']:.2f} is earned by signal, not batch.",
            "false_alarms": ["claiming plate/batch confounding", "claiming feature-selection leakage"],
            "acceptable_caveats": ["needs external validation", "modest sample size", "selection inside CV"],
        },
    )
    write_csv(ws / "samples.csv", samples)
    write_csv(ws / "protein_matrix.csv", matrix)


# --- T12 / C06: effect prediction, a hidden clustering column ------------------------------------------------


def families(seed: int, family_disjoint: bool):
    """Gene effect-prediction rows: labels follow per-family propensity; a weak real per-variant signal exists.
    With family_disjoint=False the split is gene-disjoint but every family appears on both sides (homology
    leakage); with True it is family-disjoint (clean)."""
    rng = np.random.default_rng(seed)
    fams = {f"pf{i:03d}": (0.9 if rng.random() < 0.5 else 0.1) for i in range(40)}
    genes = {f"g{i:03d}": fam for i, fam in enumerate(f for f in sorted(fams) for _ in range(4))}
    rows = []
    for g, fam in genes.items():
        for j in range(int(rng.integers(4, 9))):
            cons = rng.normal(0, 1)
            y = int(rng.random() < np.clip(fams[fam] + 0.15 * np.tanh(cons), 0, 1))
            rows.append({"variant": f"{g}_v{j}", "gene": g, "pfam": fam,
                         "conservation": f"{cons:.3f}", "label": y})
    if family_disjoint:
        held = set(sorted(fams)[-10:])
        train = [r for r in rows if r["pfam"] not in held]
        test = [r for r in rows if r["pfam"] in held]
    else:
        order = rng.permutation(sorted(genes))
        tr_genes = set(order[:120])
        train = [r for r in rows if r["gene"] in tr_genes]
        test = [r for r in rows if r["gene"] not in tr_genes]
    return train, test


def t12() -> None:
    train, test = families(12, family_disjoint=False)
    mem = checks.split_overlap(train, test, ["gene", "pfam"], label="label")
    base = checks.split_overlap(train, test, ["pfam"], label="label")["entity_memorization_baseline"]
    check(mem["columns"]["gene"]["test_rows_with_seen_value"] == 0.0, "T12 gene-disjoint")
    check(mem["columns"]["pfam"]["test_rows_with_seen_value"] == 1.0, "T12 family-shared")
    check(base["score"] > 0.7, "T12 family memorization scores")
    ws = task(
        "T12",
        """
# Predicting regulatory-variant effects across genes

`train.csv` and `test.csv` give common regulatory variants (variant id, gene, Pfam domain family,
`conservation` score) and whether the variant alters expression (`label`).

The split is gene-disjoint: no test variant's gene appears in training.

Goal: train a classifier and report its held-out AUROC.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "L (leakage through a clustering level)",
            "mechanism": "The split is gene-disjoint but not family-disjoint: every test variant's Pfam family "
            "appears in training, and labels follow per-family propensity, so a model can score well by "
            "memorizing family label rates. Gene-level disjointness is only the appearance of a clean split; "
            "the generalization claimed (to unrelated genes) is untested.",
            "quantify": [
                "100% of test variants share their pfam family with training",
                f"a family label-rate baseline reaches AUROC ~{base['score']:.2f}",
            ],
            "fix": "Split at the family level (no test variant's family in training), or report AUROC within "
            "families unseen in training.",
            "book_case": "F12 / F09 (disguised)",
        },
    )
    write_csv(ws / "train.csv", train)
    write_csv(ws / "test.csv", test)


def c06() -> None:
    train, test = families(13, family_disjoint=True)
    mem = checks.split_overlap(train, test, ["gene", "pfam"], label="label")
    check(mem["test_rows_with_any_seen_entity"] == 0.0, "C06 family-disjoint")
    ws = task(
        "C06",
        """
# Predicting regulatory-variant effects in unseen gene families

`train.csv` and `test.csv` give common regulatory variants (variant id, gene, Pfam domain family,
`conservation` score) and whether the variant alters expression (`label`).

The split is family-disjoint: no test variant's Pfam family appears in training.

Goal: train a classifier and report its held-out AUROC.
""",
        {
            "flawed": False,
            "expected_verdict": "SUPPORTED",
            "mechanism": "Clean control. The split is disjoint at both gene and family level, so a moderate "
            "AUROC (driven by conservation, the one real signal) is earned; a family label-rate baseline is "
            "useless on unseen families.",
            "false_alarms": ["claiming family leakage (the split is family-disjoint)",
                             "calling the result invalid because conservation does most of the work"],
            "acceptable_caveats": ["modest effect size", "one label source", "wider families may still overlap"],
        },
    )
    write_csv(ws / "train.csv", train)
    write_csv(ws / "test.csv", test)


# --- T13: two independent flaws at once (leaky entities AND a confounded batch) -------------------------------


def t13() -> None:
    """PPI-style pairs with a random split (entity leakage) and a processing batch confounded with the label.
    A reviewer who finds one flaw may stop; the rubric wants both."""
    rng = np.random.default_rng(13)
    prop = {f"P{i:03d}": rng.beta(0.6, 1.2) for i in range(60)}
    names = sorted(prop)
    pairs = set()
    while len(pairs) < 600:
        a, b = sorted(rng.choice(names, 2, replace=False))
        pairs.add((a, b))
    rows = []
    for a, b in sorted(pairs):
        y = int(rng.random() < min(1, 1.8 * prop[a] * prop[b]))
        # batch correlates with label: positives disproportionately processed in batch 1-2
        batch = int(rng.random() < (0.8 if y else 0.2)) + 1
        rows.append({"protein_a": a, "protein_b": b, "batch": batch,
                     "coexpression_z": f"{rng.normal(0.4 * y, 1.0):.3f}", "label": y})
    idx = rng.permutation(len(rows))
    train, test = [rows[i] for i in idx[:480]], [rows[i] for i in idx[480:]]
    mem = checks.split_overlap(train, test, ["protein_a", "protein_b"], label="label")
    sh = checks.shortcuts(train + test, "label", ["batch"], group_column=None)
    check(mem["test_rows_with_any_seen_entity"] > 0.95, "T13 overlap")
    check(sh["features"][0]["score"] > 0.7, "T13 batch confounded")
    ws = task(
        "T13",
        """
# Protein–protein interaction prediction

`pairs_train.csv` and `pairs_test.csv` list tested protein pairs (label 1 = interacting), the `batch` in which
the pair was assayed, and `coexpression_z` for the pair.

Goal: train a classifier on the training pairs and report its held-out AUROC on the test pairs.
""",
        {
            "flawed": True,
            "expected_verdict": "NOT SUPPORTED",
            "class": "L + S (two flaws)",
            "mechanism": "Two independent flaws: (1) pairs were split at random, so every test protein also "
            "appears in training and labels follow per-protein propensity — memorizing entity label rates "
            "reproduces most of the score; (2) the assay batch is confounded with the label (positives were "
            "mostly run in batch 1), so batch predicts the label without any biology. A valid review names "
            "BOTH: fixing only the split leaves the confounded batch; fixing only the batch leaves the "
            "memorization.",
            "quantify": [
                f"~{mem['test_rows_with_any_seen_entity']:.0%} of test pairs share a protein with training",
                f"batch alone predicts the label with AUROC ~{sh['features'][0]['score']:.2f}",
            ],
            "fix": "Entity-disjoint split AND batch-matched (or batch-corrected) evaluation; report both "
            "baselines.",
            "book_case": "F03 + F10 (disguised)",
        },
    )
    write_csv(ws / "pairs_train.csv", train)
    write_csv(ws / "pairs_test.csv", test)


GENERATORS = [t09, t10, t11, t12, t13, c05, c06]

if __name__ == "__main__":
    for g in GENERATORS:
        g()
        print("ok", g.__name__.upper())
