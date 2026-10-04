# PromoNet: predicting promoter activity from sequence

PromoNet is a convolutional network that predicts whether a 300 bp promoter is active (drives reporter expression
above background) in our massively parallel reporter assay in HepG2 cells.

- Data: `train.fa` (300 promoters), `test.fa` (100 promoters), `labels.csv` (activity per promoter),
  `predictions_test.csv` (PromoNet's predicted class for each test promoter).
- Split: promoters were randomly sampled into train and test sets.
- Result: **test accuracy 0.85** (chance 0.50).

**Claim:** PromoNet has learned the regulatory grammar of HepG2 promoters and can predict the activity of new,
unseen promoter sequences.
