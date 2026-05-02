# PhyloAE: Finding Admixed Dog Breeds from Embedding Geometry

CBMF W4761 (Computational Genomics) Spring 2026 final project. PhyloAE is a
multi-scale MDS autoencoder for dog-breed population genomics. The 4-page
Bioinformatics Applications Note describing the method and findings is
`report.pdf`.

## Layout

- `phyloae.ipynb` — the canonical end-to-end notebook: data download, baseline
  embedding sweeps, ADMIXTURE comparison, PhyloAE training, Parker NJ-tree
  validation, figure rendering.
- `helpers/` — Python modules called from the notebook: `lib.py` (shared
  utilities), `figures/` (display-item generators), `sweep/` (the
  hyperparameter-sweep driver), and `data_assets.py` (Parker-asset fetcher).
- `data/` — sample input plus a README pointing at the SNP panel and Parker
  supplementary sources. The notebook fetches the large/derived files at
  setup time.
- `results/` — committed reference outputs: rendered figures
  (`results/figures/`) and pinned metric files (`results/metrics/`).
- `report.pdf` — the Applications Note (deliverable).

## System requirements

Python 3.11+ on macOS (ARM64/Intel) or Linux. ~16 GB RAM for the full sweep.
PLINK 1.9 and ADMIXTURE 1.3 are installed by `helpers/bootstrap_popgen_tools.sh`.
Python deps in `requirements.txt`.

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash helpers/bootstrap_popgen_tools.sh
export PATH="$(pwd)/.tools/bin:$PATH"
```

## Test run on sample (~2 min)

`data/sample/keep_50_dogs.txt` is a 50-dog list spanning all 23 Parker clades.
Subsample the SNP panel and run the notebook against it:

```sh
plink --bfile data/All_Pure_150k --keep data/sample/keep_50_dogs.txt \
      --make-bed --out data/sample/sample
jupyter lab phyloae.ipynb
```

Set `RAW_PREFIX = DATA_DIR / 'sample' / 'sample'` in the configuration cell
and run all cells. Expected outputs land in `results/metrics/` (e.g.,
`parker_concordance.json`, `mantel_nj.json`); the bundled
`results/metrics/*.json/.tsv` files are reference outputs from the full
pipeline.

## Full run on the Parker panel

Download `All_Pure_150k.{bed,bim,fam}` per `data/README.md`, place them in
`data/`, then open `phyloae.ipynb` and run the cells in order. The first
setup cell fetches the small Parker supplementary assets. Subsequent cells
load cached embeddings and metrics from `results/` where available.

## License

MIT — see `LICENSE`.
