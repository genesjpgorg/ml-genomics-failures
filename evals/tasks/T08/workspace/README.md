# A 20-gene expression signature predicts immunotherapy response in melanoma

From bulk RNA-seq of 60 pre-treatment melanoma biopsies (5,000 most variable genes), we derived a 20-gene
signature that predicts response to anti-PD-1 therapy.

- Data: `expression.csv` (patient × gene, z-scored), `response.csv` (responder 1/0).
- Analysis: `analysis.py` (selects the 20 genes most associated with response, then evaluates a logistic regression
  by 5-fold stratified cross-validation).
- Result: **cross-validated AUROC 0.99** (`results.json`).

**Claim:** the 20-gene signature predicts anti-PD-1 response (CV AUROC 0.99) and should be tested
prospectively to select patients for therapy.
