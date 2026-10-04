# Protein–protein interaction prediction

`pairs_train.csv` and `pairs_test.csv` list tested protein pairs from a yeast two-hybrid screen
(label 1 = interacting, 0 = tested, no interaction), with `coexpression_z`, the z-scored co-expression of the
pair across 300 conditions.

Goal: train a classifier on the training pairs and report its held-out AUROC on the test pairs.
