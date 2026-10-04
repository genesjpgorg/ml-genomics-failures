# Plasma proteins associated with sepsis-induced kidney injury

We measured 200 plasma proteins by targeted mass spectrometry in 60 sepsis patients who developed acute kidney
injury (cases) and 60 who did not (controls). Samples were randomized across 4 plates, 15 cases and 15 controls per
plate, before processing.

- Data: `samples.csv` (status, plate, run date), `protein_matrix.csv` (log-intensities).
- Evaluation: 5-fold stratified cross-validation; standardization, selection of 12 proteins and logistic regression
  are fitted inside each training fold.
- Result: **cross-validated AUROC 0.71** (`results.json`).

**Claim:** plasma proteins measured at sepsis onset distinguish patients who go on to develop kidney injury
(cross-validated AUROC 0.71); the signal is worth testing in an external cohort.
