#!/usr/bin/env python3
"""Run PhyloAE distance-metric sweep locally with MPS GPU. Safe to interrupt — caches each run.

Extends the existing PhyloAE sweep with target_metric ∈ {euclidean, manhattan, cosine}.
Existing 240 Euclidean runs cache-hit; 480 new (Manhattan + Cosine) train.
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _proj)
os.chdir(_proj)
print(f"Working dir: {os.getcwd()}", flush=True)

import torch
torch.set_num_threads(12)  # all M4 Pro cores

from pathlib import Path
from helpers import lib  # noqa: E402

meta, X = lib.load_plink_raw(Path("results/qc/all_qc_ldpruned_additive.raw"))
meta, X_scaled, breed_order, color_mpl, color_plotly, \
    clade_order, clade_color_mpl, clade_color_plotly = \
    lib.prepare_features(meta, X, clade_csv=Path("data/parker_clades.csv"))

EMBED_DIR = Path("results/embeddings")
METRIC_DIR = Path("results/metrics")

embs, summary = lib.run_phyloae_sweep(
    X_scaled, meta,
    hidden_dims_list=[(512, 64), (1024, 128)],
    lrs=[1e-3, 3e-4],
    lambda_breeds=[0.1, 1.0, 10.0],
    lambda_clades=[0.1, 1.0],
    n_components=[2, 3],
    seeds=[42, 123, 256, 789, 1337],
    embed_dir=EMBED_DIR,
    metric_dir=METRIC_DIR,
    target_metrics=["euclidean", "manhattan", "cosine"],
    max_epochs=500,
)
print(f"\nDone! {len(summary)} runs total.")
