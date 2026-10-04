# Predicting protein–protein interactions from co-expression

We predict whether two yeast proteins physically interact from a pair-level feature: the z-scored co-expression of
their genes across 300 conditions (`coexpression_z`), using logistic regression.

- Data: `pairs_train.csv` (600 pairs among 60 proteins), `pairs_test.csv` (150 pairs among 20 other proteins).
- Split: protein-disjoint — no protein in the test set appears in any training pair.
- Result: **AUROC 0.81** on the test pairs (`predictions_test.csv`).

**Claim:** co-expression predicts physical interaction for pairs of proteins never seen in training
(AUROC 0.81), so it is a useful prior for ranking candidate interaction partners of uncharacterized proteins.
