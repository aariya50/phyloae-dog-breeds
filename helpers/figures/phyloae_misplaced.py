#!/usr/bin/env python3
"""Plot PhyloAE 2D embedding with consensus-misplaced breeds highlighted."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import lib  # noqa: E402

MEDITERRANEAN_LABELS = ["PHAR", "IBIZ", "CIRN_Italy", "AZWK_Mali", "KOMO", "GPYR", "MAAB_Italy"]

# Set by Phase A of the cosine-3D headline swap (2026-04-27).
# Mean across 5 seeds: Parker=65.14%, Mantel=0.6056. This seed's Parker is
# closest to the mean -- see plan file. The 2D companion of the 3D headline run
# (same hidden_dims/lr/lambdas/seed) is used for the misplaced-highlight scatter.
PHYLOAE_OVERRIDE_RUN_ID = "phyloae_cosine_h512x64_lr0p0003_lb0p1_lc0p1_seed789_2d"


def _select_best_phyloae_csv(metrics_dir: Path) -> Path:
    summary_path = metrics_dir / "phyloae_sweep_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        # Phase A cosine-3D headline override: force PhyloAE to a specific run_id (2D companion).
        if PHYLOAE_OVERRIDE_RUN_ID:
            override = summary[summary["run_id"].astype(str) == PHYLOAE_OVERRIDE_RUN_ID]
            if not override.empty:
                csv_path = Path(str(override.iloc[0].get("embedding_csv", ""))).expanduser()
                if csv_path.exists():
                    return csv_path
        best = lib.select_best_run(summary, n_components=2)
        csv_path = Path(str(best.get("embedding_csv", ""))).expanduser()
        if csv_path.exists():
            return csv_path

    metrics_k = metrics_dir / "embedding_metrics_by_k.tsv"
    if metrics_k.exists():
        df = pd.read_csv(metrics_k, sep="\t")
        sub = df[df["method"].astype(str) == "PhyloAE"].copy()
        sub["composite"] = sub["trustworthiness"] + sub["silhouette_breed"]
        best_row = sub.sort_values("composite", ascending=False).iloc[0]
        csv_path = Path(str(best_row["embedding_csv"])).expanduser()
        if csv_path.exists():
            return csv_path

    raise FileNotFoundError("Unable to resolve best PhyloAE 2D embedding CSV.")


def run(project_root: Path) -> None:
    metrics_dir = project_root / "results" / "metrics"
    out_path = project_root / "results" / "figures" / "presentation" / "phyloae_misplaced.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "parker_concordance.json") as f:
        concordance = json.load(f)
    misplaced_breeds = {entry["breed"] for entry in concordance["cross_method_misplaced"]}

    emb_csv = _select_best_phyloae_csv(metrics_dir)
    emb = pd.read_csv(emb_csv)

    breed_col = "breed" if "breed" in emb.columns else "FID"
    emb["_breed"] = emb[breed_col].astype(str)
    emb["_highlight"] = emb["_breed"].isin(misplaced_breeds)

    bg = emb[~emb["_highlight"]]
    fg = emb[emb["_highlight"]]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    ax.scatter(bg["PhyloAE1"], bg["PhyloAE2"],
               c="#4B5563", alpha=0.3, s=15, zorder=1)
    ax.scatter(fg["PhyloAE1"], fg["PhyloAE2"],
               c="#F6AD55", alpha=0.95, s=40, edgecolors="black",
               linewidths=0.5, zorder=2)

    for breed in MEDITERRANEAN_LABELS:
        subset = fg[fg["_breed"] == breed]
        if subset.empty:
            continue
        cx, cy = subset["PhyloAE1"].mean(), subset["PhyloAE2"].mean()
        ax.annotate(breed, (cx, cy), fontsize=8, color="#F6AD55",
                    textcoords="offset points", xytext=(6, 4),
                    zorder=3)

    ax.set_title(f"PhyloAE Embedding — {len(misplaced_breeds)} Consensus-Misplaced Breeds Highlighted",
                 color="white", fontsize=14, pad=10)
    ax.set_xlabel("PhyloAE 1", color="white", fontsize=10)
    ax.set_ylabel("PhyloAE 2", color="white", fontsize=10)
    ax.tick_params(colors="#6B7280", labelsize=8)
    ax.grid(True, color="white", alpha=0.06, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#364152")

    fig.savefig(out_path, bbox_inches="tight", dpi=200, facecolor="#0E1117")
    plt.close(fig)
    print(f"Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    run(PROJECT_ROOT)
