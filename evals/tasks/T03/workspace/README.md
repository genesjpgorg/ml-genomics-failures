# A 12-protein plasma signature for early pancreatic cancer

We profiled 200 plasma proteins by targeted mass spectrometry in 60 early-stage pancreatic cancer cases and 60
matched controls, and trained a logistic-regression classifier on the 12 most discriminative proteins.

- Data: `samples.csv` (sample, case/control status, plate, run date), `protein_matrix.csv` (log-intensities).
- Evaluation: 5-fold stratified cross-validation over the 120 samples.
- Result: **cross-validated AUROC 0.90** (`results.json`).

**Claim:** the 12-protein signature detects early pancreatic cancer from plasma and is ready for validation as a
screening test.
