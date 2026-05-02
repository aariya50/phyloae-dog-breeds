#!/usr/bin/env python3
"""Continuous per-breed admixture score from best cached embeddings (2D/3D).

Outputs:
- results/metrics/admixture_score.json
- results/figures/presentation/admixture_score.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import lib  # noqa: E402

METHODS = ["UMAP", "PaCMAP", "t-SNE", "TriMAP", "PHATE", "PhyloAE"]
SUMMARY_FILES = {
    "UMAP": "umap_sweep_summary.csv",
    "PaCMAP": "pacmap_sweep_summary.csv",
    "t-SNE": "tsne_sweep_summary.csv",
    "TriMAP": "trimap_sweep_summary.csv",
    "PHATE": "phate_sweep_summary.csv",
    "PhyloAE": "phyloae_sweep_summary.csv",
}

# Set by Phase A of the cosine-3D headline swap (2026-04-27).
# Mean across 5 seeds: Parker=65.14%, Mantel=0.6056. This seed's Parker is
# closest to the mean -- see plan file.
PHYLOAE_OVERRIDE_RUN_ID = "phyloae_cosine_h512x64_lr0p0003_lb0p1_lc0p1_seed789_3d"


def _find_coord_columns(df: pd.DataFrame, method: str) -> list[str]:
    prefix_map = {
        "UMAP": "UMAP",
        "PaCMAP": "PaCMAP",
        "t-SNE": "TSNE",
        "TriMAP": "TriMAP",
        "PHATE": "PHATE",
        "PhyloAE": "PhyloAE",
    }
    prefix = prefix_map[method]
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    tagged: list[tuple[int, str]] = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            tagged.append((int(match.group(1)), col))
    if len(tagged) >= 2:
        tagged.sort(key=lambda x: x[0])
        return [col for _, col in tagged]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols
    raise ValueError(f"Could not infer coordinate columns for {method}.")


def _select_best_embedding_csv(metrics_dir: Path, method: str) -> tuple[Path, int]:
    summary_path = metrics_dir / SUMMARY_FILES[method]
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        needed = {"n_components", "trustworthiness", "silhouette_breed", "embedding_csv"}
        missing = needed - set(summary.columns)
        if missing:
            raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")
        summary = summary.copy()
        summary["composite"] = summary["trustworthiness"] + summary["silhouette_breed"]
        summary = summary.dropna(subset=["composite"])
        if summary.empty:
            raise ValueError(f"No valid composite rows in {summary_path}")
        # Phase A cosine-3D headline override: force PhyloAE to a specific run_id.
        if method == "PhyloAE" and PHYLOAE_OVERRIDE_RUN_ID:
            override = summary[summary["run_id"].astype(str) == PHYLOAE_OVERRIDE_RUN_ID]
            if not override.empty:
                best_row = override.iloc[0]
                csv_path = Path(str(best_row.get("embedding_csv", ""))).expanduser()
                if csv_path.exists():
                    return csv_path, int(best_row["n_components"])
        best_row = summary.sort_values("composite", ascending=False).iloc[0]
        csv_path = Path(str(best_row.get("embedding_csv", ""))).expanduser()
        if csv_path.exists():
            return csv_path, int(best_row["n_components"])

    metrics_k = metrics_dir / "embedding_metrics_by_k.tsv"
    if metrics_k.exists():
        df = pd.read_csv(metrics_k, sep="\t")
        needed = {"method", "trustworthiness", "silhouette_breed", "embedding_csv"}
        if needed.issubset(set(df.columns)):
            sub = df[df["method"].astype(str) == method].copy()
            sub["composite"] = sub["trustworthiness"] + sub["silhouette_breed"]
            sub = sub.dropna(subset=["composite"])
            if sub.empty:
                raise ValueError(f"No valid rows for method={method} in {metrics_k}")
            best_row = sub.sort_values("composite", ascending=False).iloc[0]
            csv_path = Path(str(best_row["embedding_csv"])).expanduser()
            if csv_path.exists():
                emb = pd.read_csv(csv_path)
                dims = len(_find_coord_columns(emb, method))
                return csv_path, int(best_row.get("n_components", dims))

    raise FileNotFoundError(
        f"Unable to resolve best embedding CSV for {method}. "
        "Expected sweep summary with embedding_csv paths."
    )


def _build_per_dog_df(embedding_csv: Path, method: str, clade_map: dict[str, str]) -> pd.DataFrame:
    emb = pd.read_csv(embedding_csv)
    coord_cols = _find_coord_columns(emb, method)

    if "breed" in emb.columns:
        breed = emb["breed"].astype(str)
    elif "FID" in emb.columns:
        breed = emb["FID"].astype(str)
    else:
        raise ValueError(f"No breed/FID column found in {embedding_csv}")

    out = pd.DataFrame({"breed": breed, "clade": breed.map(clade_map)})
    for idx, col in enumerate(coord_cols, start=1):
        out[f"coord_{idx}"] = emb[col].astype(float)
    missing = out["clade"].isna().sum()
    if missing:
        raise ValueError(
            f"{method}: {missing} samples could not be mapped to Parker clades; "
            "check breed_code alignment."
        )
    return out


def _compute_admixture_scores(per_dog: pd.DataFrame) -> pd.DataFrame:
    coord_cols = [c for c in per_dog.columns if c.startswith("coord_")]
    clade_centroids = per_dog.groupby("clade", as_index=False)[coord_cols].mean()
    breed_centroids = per_dog.groupby(["breed", "clade"], as_index=False)[coord_cols].mean()

    clade_names = clade_centroids["clade"].tolist()
    clade_xy = clade_centroids[coord_cols].to_numpy(dtype=float)
    clade_names_arr = np.array(clade_names, dtype=object)
    own_idx = breed_centroids["clade"].map({c: i for i, c in enumerate(clade_names)}).to_numpy(dtype=int)
    D = cdist(breed_centroids[coord_cols].to_numpy(dtype=float), clade_xy)
    own_dist = np.take_along_axis(D, own_idx[:, None], axis=1).ravel()
    other = D.copy()
    other[np.arange(len(other)), own_idx] = np.inf
    nearest_other_idx = other.argmin(axis=1)
    nearest_other_dist = np.take_along_axis(other, nearest_other_idx[:, None], axis=1).ravel()

    out = breed_centroids[["breed", "clade"]].copy()
    out["score"] = own_dist / np.maximum(nearest_other_dist, 1e-12)
    out["own_clade_distance"] = own_dist
    out["nearest_other_clade"] = clade_names_arr[nearest_other_idx]
    out["nearest_other_clade_distance"] = nearest_other_dist
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def _plot_consensus(consensus_top15: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=220, facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    breeds = consensus_top15["breed"].astype(str).tolist()
    scores = consensus_top15["mean_score"].astype(float).to_numpy()

    bars = ax.barh(breeds, scores, color="#F6AD55", edgecolor="#F6AD55", alpha=0.95)
    ax.invert_yaxis()

    xmax = float(np.max(scores)) if len(scores) else 1.0
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel("Mean admixture score", color="white", fontsize=12)
    ax.set_title("Consensus Per-Breed Admixture Score (6-method mean)", color="white", fontsize=18, pad=12)
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(axis="x", color="white", alpha=0.12, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#364152")

    for bar, score in zip(bars, scores):
        ax.text(
            float(bar.get_width()) + 0.01 * xmax,
            float(bar.get_y()) + float(bar.get_height()) / 2.0,
            f"{score:.3f}",
            va="center",
            ha="left",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    fig.savefig(out_path, bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)


def run(project_root: Path) -> None:
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    metrics_dir = results_dir / "metrics"

    json_out = metrics_dir / "admixture_score.json"
    fig_out = results_dir / "figures" / "presentation" / "admixture_score.png"

    clade_df = pd.read_csv(data_dir / "parker_clades.csv")
    clade_map = dict(zip(clade_df["breed_code"].astype(str), clade_df["clade"].astype(str)))

    per_method_scores: dict[str, pd.DataFrame] = {}
    per_method_top20: dict[str, list[dict]] = {}
    dimensions_used: dict[str, int] = {}

    for method in METHODS:
        emb_csv, n_components = _select_best_embedding_csv(metrics_dir, method)
        per_dog = _build_per_dog_df(emb_csv, method, clade_map)
        breed_scores = _compute_admixture_scores(per_dog)
        per_method_scores[method] = breed_scores
        dimensions_used[method] = n_components

        top20 = []
        for _, row in breed_scores.head(20).iterrows():
            top20.append(
                {
                    "breed": str(row["breed"]),
                    "clade": str(row["clade"]),
                    "score": float(row["score"]),
                }
            )
        per_method_top20[method] = top20

    shared_breeds = sorted(set.intersection(*(set(df["breed"].astype(str)) for df in per_method_scores.values())))

    consensus_rows: list[dict] = []
    for breed in shared_breeds:
        method_scores = {
            method: float(
                per_method_scores[method].loc[
                    per_method_scores[method]["breed"].astype(str) == breed, "score"
                ].iloc[0]
            )
            for method in METHODS
        }
        clade = str(
            per_method_scores["PhyloAE"].loc[
                per_method_scores["PhyloAE"]["breed"].astype(str) == breed, "clade"
            ].iloc[0]
        )
        mean_score = float(np.mean(list(method_scores.values())))
        consensus_rows.append(
            {
                "breed": breed,
                "parker_clade": clade,
                "mean_score": mean_score,
                "per_method_scores": method_scores,
            }
        )

    consensus_rows.sort(key=lambda r: r["mean_score"], reverse=True)
    consensus_top20 = consensus_rows[:20]

    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_method_top20": per_method_top20,
        "consensus_top20": consensus_top20,
        "dimensions_used": dimensions_used,
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    consensus_df = pd.DataFrame(consensus_rows)
    _plot_consensus(consensus_df.head(15), fig_out)

    print("Per-method top score leaders:")
    for method in METHODS:
        row = per_method_scores[method].iloc[0]
        print(f"  - {method}: {row['breed']} ({row['score']:.4f})")

    top3 = [row["breed"] for row in consensus_rows[:3]]
    print("Consensus top-3 breeds:", ", ".join(top3))
    print(f"Wrote JSON: {json_out}")
    print(f"Wrote figure: {fig_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-breed admixture score from cached best embeddings.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to Project root (default: inferred from script location).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.project_root.resolve())
