# Predicting regulatory-variant effects across genes

`train.csv` and `test.csv` give common regulatory variants (variant id, gene, Pfam domain family,
`conservation` score) and whether the variant alters expression (`label`).

The split is gene-disjoint: no test variant's gene appears in training.

Goal: train a classifier and report its held-out AUROC.
