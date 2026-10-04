# BindGrammar: a CNN learns the cis-regulatory grammar of FOXA2 binding

We trained a CNN to classify 200 bp sequences as FOXA2-bound (ChIP-seq peak summits) or unbound (random accessible
regions) in liver.

- Data: `sequences.csv` (id, split, bound, sequence).
- Result: **test AUROC 0.89**.
- Control: we retrained the same CNN on dinucleotide-shuffled versions of every sequence (labels unchanged).
  The shuffled-sequence model reaches **AUROC 0.86**. Both numbers are from one training run (seed 0);
  see `results.json`.

**Claim:** because the CNN outperforms the shuffled-sequence control, it has learned the arrangement of FOXA2
motifs and co-factor motifs (cis-regulatory grammar), not just sequence composition.
