# Data

The notebook fetches and derives all working data from public sources. Nothing
in this folder is committed to the repo aside from this README and the small
sample-input file under `sample/`.

## SNP panel (large)

Parker 2017 NHGRI canine SNP panel:

> https://research.nhgri.nih.gov/dog_genome/downloads/datasets/SNP/2017-parker-data/

Download `All_Pure_150k.bed`, `.bim`, `.fam` and place them here in `data/`.

## Parker supplementary (small, derived)

Two small files are derived from Parker et al. (2017) supplementary materials:

- `parker_clades.csv` — breed code → clade mapping (175 rows)
- `parker_tree/parker_2017_nj.nex` — the published neighbor-joining tree

Source paper:

> Parker HG, Dreger DL, Rimbault M, Davis BW, Mullen AB, Carpintero-Ramirez G,
> Ostrander EA. (2017). Genomic Analyses Reveal the Influence of Geographic
> Origin, Migration, and Hybridization on Modern Dog Breed Development.
> *Cell Reports* 19(4):697–708. doi:10.1016/j.celrep.2017.03.079

The notebook calls `helpers.data_assets.ensure_parker_assets(...)` during
setup, which downloads these files from the project's release mirror if they
aren't already present locally. If both the local copy and the mirror are
unavailable, the helper raises with a pointer back to this README.

## Sample

`sample/keep_50_dogs.txt` — 50-dog subset spanning all 23 Parker clades, used
by the test-run path documented in the top-level `README.md`.
