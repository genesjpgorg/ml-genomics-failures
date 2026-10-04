# ExprNet: predicting baseline expression levels from reference sequence

ExprNet predicts gene expression (log TPM, lymphoblastoid cells) from 100 kb of reference DNA sequence around each
gene's TSS. Training used genes on all chromosomes except 8–10.

- Evaluation on held-out chromosomes 8–10: predicted vs observed expression across 2,000 genes gives
  **Pearson r = 0.79** (`gene_level_eval.csv`).

**Claim:** reference sequence near the TSS explains much of the variation in baseline expression **between genes**
in this cell type (r = 0.79 on held-out chromosomes). We make no claim about predicting differences
between individuals or the effects of genetic variants, which this evaluation does not test.
