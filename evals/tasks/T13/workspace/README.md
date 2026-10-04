# Protein–protein interaction prediction

`pairs_train.csv` and `pairs_test.csv` list tested protein pairs (label 1 = interacting), the `batch` in which
the pair was assayed, and `coexpression_z` for the pair.

Goal: train a classifier on the training pairs and report its held-out AUROC on the test pairs.
