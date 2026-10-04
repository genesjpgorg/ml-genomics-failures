# A CNN detects the order of AP-1 and E-box motifs in synthetic enhancers

In a synthetic enhancer library, every 200 bp sequence carries one AP-1 site (TGACTCAG) and one E-box (CACGTGAC),
4–10 bp apart; only the order differs (AP-1 upstream = class 1, E-box upstream = class 0). The motif pair is placed
uniformly along the sequence, so each motif's absolute position has nearly the same distribution in both classes,
and background sequence is random without other motif copies, so both classes have the same composition. We
trained a CNN to predict the order class.

- Data: `sequences.csv` (id, split, order_ab, sequence); `predictions_test.csv`.
- Split: independently synthesized sequences; the most similar train/test pair shares < 40% identity outside the motifs.
- Result: **test accuracy 0.89**; mean over 3 seeds 0.90 ± 0.01.
- Controls: dinucleotide-shuffled sequences (motifs destroyed) give 0.51; permuted labels give 0.50.

**Claim:** the CNN learns motif order: it distinguishes sequences that differ only in the arrangement of the same
two motifs.
