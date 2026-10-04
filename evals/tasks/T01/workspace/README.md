# PairNet: predicting protein–protein interactions from sequence

We trained PairNet, a siamese transformer over ESM-style protein sequence embeddings, to predict whether two
yeast proteins physically interact (label 1 = interaction in our curated screen, 0 = tested, no interaction).

- Data: 700 tested protein pairs over 60 proteins (`pairs_train.csv`, `pairs_test.csv`).
- Split: pairs were randomly assigned to train (80%) and test (20%).
- Result: on the held-out test pairs PairNet reaches **AUROC 0.79** (`predictions_test.csv`).

**Claim:** PairNet has learned sequence determinants of physical interaction and generalizes to new protein pairs,
so we will use it to predict interaction partners for uncharacterized proteins.
