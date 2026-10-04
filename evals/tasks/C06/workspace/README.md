# Predicting regulatory-variant effects in unseen gene families

`train.csv` and `test.csv` give common regulatory variants (variant id, gene, Pfam domain family,
`conservation` score) and whether the variant alters expression (`label`).

The split is family-disjoint: no test variant's Pfam family appears in training.

Goal: train a classifier and report its held-out AUROC.
