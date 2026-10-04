# SpliceScore: classifying splice-disrupting variants

SpliceScore is a gradient-boosted model over 140 variant-level features (conservation, predicted splice-site
strength, distance to exon boundary, ...) that classifies rare variants as splice-disrupting (1) or not (0), with
labels from a minigene assay compendium.

- Data: `variants.csv`: variant id, gene, one representative feature (`conservation`), assay label, split, and the
  model's test-time score. The full feature matrix is too large to share here.
- Split: variants randomly assigned to train (80%) and test (20%).
- Result: **AUROC 0.94** on test variants.

**Claim:** SpliceScore learns variant-level determinants of splice disruption and should be used to prioritize
variants of uncertain significance in diagnostic pipelines.
