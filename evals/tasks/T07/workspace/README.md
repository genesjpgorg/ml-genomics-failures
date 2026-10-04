# ExprNet for personalized regulatory variant interpretation

ExprNet predicts gene expression (log TPM, lymphoblastoid cells) from 100 kb of reference DNA sequence around each
gene's TSS.

- Evaluation on held-out chromosomes 8–10: predicted vs observed expression across 2,000 genes gives
  **Pearson r = 0.80** (`gene_level_eval.csv`).
- We also scored 300 common variants by in-silico mutagenesis and include the comparison with eQTL effect sizes
  from a 400-individual cohort (`variant_eval.csv`).

**Claim:** ExprNet accurately predicts expression from sequence (r = 0.80), so we will use it to
predict the expression impact of each patient's personal regulatory variants in our rare-disease cohort.
