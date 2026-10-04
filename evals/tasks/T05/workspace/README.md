# PhageFormer: identifying phage contigs in gut metagenomes

PhageFormer is a 6-layer transformer over 6-mer tokens that classifies 1.5 kb metagenomic contigs as phage or
bacterial. Attention maps highlight regions resembling phage structural and replication genes.

- Data: `train.csv` and `test.csv` (contig id, label, sequence).
- Result: **test accuracy 0.99** (`results.json`).

**Claim:** PhageFormer learns phage-specific gene content and sequence features, and can discover novel phages in
metagenomes where homology-based tools fail.
