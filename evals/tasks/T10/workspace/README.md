# Phage vs bacterial contigs in gut metagenomes

`train.csv` and `test.csv` give 1.5 kb metagenomic contigs labeled `phage` or `bacterial`.

Goal: train a classifier and report its held-out accuracy. The downstream claim we want to make is that the
classifier picks up phage-specific biological features, so it could find novel phages that homology tools miss.
