#!/usr/bin/env python3
"""Parker concordance analysis over best cached embeddings (2D/3D).

Outputs:
- results/metrics/parker_concordance.json
- results/metrics/internal_metrics_best.json
- results/figures/presentation/parker_concordance.png
"""
from __future__ import annotations

import argparse
import difflib
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
A_PRIORI = [
    "Eurasier",
    "CatahoulaLeopardDog",
    "Chinook",
    "RedboneCoonhound",
    "Catahoula",
    "Redbone",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


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


def _coerce_optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _select_best_embedding_csv(metrics_dir: Path, method: str) -> tuple[Path, dict]:
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
                    return csv_path, {
                        "trustworthiness": float(best_row["trustworthiness"]),
                        "silhouette_breed": float(best_row["silhouette_breed"]),
                        "seed_stability": _coerce_optional_float(best_row.get("seed_stability_vs_base")),
                        "n_components": int(best_row["n_components"]),
                        "composite": float(best_row["composite"]),
                    }
        best_row = summary.sort_values("composite", ascending=False).iloc[0]
        csv_path = Path(str(best_row.get("embedding_csv", ""))).expanduser()
        if csv_path.exists():
            return csv_path, {
                "trustworthiness": float(best_row["trustworthiness"]),
                "silhouette_breed": float(best_row["silhouette_breed"]),
                "seed_stability": _coerce_optional_float(best_row.get("seed_stability_vs_base")),
                "n_components": int(best_row["n_components"]),
                "composite": float(best_row["composite"]),
            }

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
                return csv_path, {
                    "trustworthiness": float(best_row["trustworthiness"]),
                    "silhouette_breed": float(best_row["silhouette_breed"]),
                    "seed_stability": _coerce_optional_float(best_row.get("seed_stability_vs_base")),
                    "n_components": int(best_row.get("n_components", dims)),
                    "composite": float(best_row["composite"]),
                }

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

    sample_id_col = "IID" if "IID" in emb.columns else "sample_id"
    if sample_id_col not in emb.columns:
        sample_id = np.arange(len(emb)).astype(str)
    else:
        sample_id = emb[sample_id_col].astype(str)

    out = pd.DataFrame({"sample_id": sample_id, "breed": breed, "clade": breed.map(clade_map)})
    for idx, col in enumerate(coord_cols, start=1):
        out[f"coord_{idx}"] = emb[col].astype(float)
    missing = out["clade"].isna().sum()
    if missing:
        raise ValueError(
            f"{method}: {missing} samples could not be mapped to Parker clades; "
            "check breed_code alignment."
        )
    return out


def _compute_concordance(per_dog: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    coord_cols = [c for c in per_dog.columns if c.startswith("coord_")]
    clade_centroids = per_dog.groupby("clade", as_index=False)[coord_cols].mean()
    breed_centroids = per_dog.groupby(["breed", "clade"], as_index=False)[coord_cols].mean()

    clade_names = clade_centroids["clade"].tolist()
    clade_xy = clade_centroids[coord_cols].to_numpy(dtype=float)
    D = cdist(breed_centroids[coord_cols].to_numpy(dtype=float), clade_xy)
    nearest_clades = [clade_names[i] for i in D.argmin(axis=1)]

    breed_centroids["nearest_clade"] = nearest_clades
    breed_centroids["correct"] = breed_centroids["nearest_clade"] == breed_centroids["clade"]
    pct_correct = float(breed_centroids["correct"].mean())
    return breed_centroids, pct_correct


def _fuzzy_a_priori_recovered(cross_method_misplaced: list[dict]) -> list[str]:
    breed_codes = sorted({str(row["breed"]) for row in cross_method_misplaced})
    code_norm = {code: _normalize(code) for code in breed_codes}
    recovered: set[str] = set()

    for target in A_PRIORI:
        t_norm = _normalize(target)
        direct = [code for code, cn in code_norm.items() if t_norm in cn or cn in t_norm]
        if direct:
            recovered.update(direct)
            continue

        scored = []
        for code, cn in code_norm.items():
            ratio = difflib.SequenceMatcher(None, t_norm, cn).ratio()
            scored.append((ratio, code))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.72:
            recovered.add(scored[0][1])

    return sorted(recovered)


def _plot(
    per_method_pct_correct: dict[str, float],
    cross_method_misplaced: list[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(per_method_pct_correct.items(), key=lambda kv: kv[1], reverse=True)
    methods = [k for k, _ in sorted_items]
    pct_vals = [100.0 * v for _, v in sorted_items]

    fig = plt.figure(figsize=(10, 4), dpi=220, facecolor="#0E1117")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#0E1117")
    bars = ax.barh(methods, pct_vals, color="#F6AD55", edgecolor="#F6AD55", alpha=0.93)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% Breeds Correctly Placed to Parker Clade", color="white", fontsize=12)
    ax.set_title("Parker Concordance by Embedding Method", color="white", fontsize=18, pad=10)
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(axis="x", color="white", alpha=0.12, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#364152")

    for bar, val in zip(bars, pct_vals):
        ax.text(
            min(val + 1.0, 98.0),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
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
    fig_out = results_dir / "figures" / "presentation" / "parker_concordance.png"
    json_out = metrics_dir / "parker_concordance.json"
    internal_json_out = metrics_dir / "internal_metrics_best.json"

    clade_df = pd.read_csv(data_dir / "parker_clades.csv")
    clade_map = dict(zip(clade_df["breed_code"].astype(str), clade_df["clade"].astype(str)))

    raw_matrix = results_dir / "qc" / "all_qc_ldpruned_additive.raw"
    meta_raw, X = lib.load_plink_raw(raw_matrix)
    meta, *_ = lib.prepare_features(meta_raw, X, clade_csv=data_dir / "parker_clades.csv")

    observed_breeds = set(meta["breed"].astype(str).unique())
    missing_breeds = sorted(observed_breeds - set(clade_map.keys()))
    if missing_breeds:
        raise ValueError(f"Breeds present in metadata but absent in parker_clades.csv: {missing_breeds[:10]}")

    per_method_pct_correct: dict[str, float] = {}
    per_method_tables: dict[str, pd.DataFrame] = {}
    dimensions_used: dict[str, int] = {}
    internal_metrics: dict[str, dict[str, float | int | None]] = {}

    for method in METHODS:
        emb_csv, best_meta = _select_best_embedding_csv(metrics_dir, method)
        per_dog = _build_per_dog_df(emb_csv, method, clade_map)
        breed_eval, pct_correct = _compute_concordance(per_dog)
        per_method_tables[method] = breed_eval
        per_method_pct_correct[method] = pct_correct
        dimensions_used[method] = int(best_meta["n_components"])
        internal_metrics[method] = best_meta

    misplaced_by_method = {
        method: set(df.loc[~df["correct"], "breed"].astype(str).tolist())
        for method, df in per_method_tables.items()
    }
    cross_misplaced_breeds = sorted(set.intersection(*(misplaced_by_method[m] for m in METHODS)))

    cross_method_misplaced: list[dict] = []
    for breed in cross_misplaced_breeds:
        method_nearest = {
            method: str(
                per_method_tables[method].loc[
                    per_method_tables[method]["breed"].astype(str) == breed, "nearest_clade"
                ].iloc[0]
            )
            for method in METHODS
        }
        parker_clade = str(
            per_method_tables["PhyloAE"].loc[
                per_method_tables["PhyloAE"]["breed"].astype(str) == breed, "clade"
            ].iloc[0]
        )
        cross_method_misplaced.append(
            {
                "breed": breed,
                "parker_clade": parker_clade,
                "method_nearest": method_nearest,
            }
        )

    a_priori_recovered = _fuzzy_a_priori_recovered(cross_method_misplaced)

    payload = {
        "per_method_pct_correct": per_method_pct_correct,
        "cross_method_misplaced": cross_method_misplaced,
        "a_priori_recovered": a_priori_recovered,
        "dimensions_used": dimensions_used,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    internal_json_out.write_text(json.dumps(internal_metrics, indent=2), encoding="utf-8")

    _plot(per_method_pct_correct, cross_method_misplaced, fig_out)

    sorted_pcts = sorted(per_method_pct_correct.items(), key=lambda kv: kv[1], reverse=True)
    print("Per-method pct correct (descending):")
    for method, pct in sorted_pcts:
        print(f"  - {method}: {pct * 100:.2f}%")

    print(f"Cross-method misplaced shortlist size: {len(cross_method_misplaced)}")

    recovered_set = set(a_priori_recovered)
    not_recovered = sorted(set(A_PRIORI) - {x for x in A_PRIORI if any(_normalize(x) in _normalize(r) or _normalize(r) in _normalize(x) for r in recovered_set)})
    print("A-priori recovered (fuzzy):", ", ".join(a_priori_recovered) if a_priori_recovered else "None")
    print("A-priori not recovered:", ", ".join(not_recovered) if not_recovered else "None")

    phyloae_ok = per_method_pct_correct["PhyloAE"] >= 0.90
    unsup_best = max(per_method_pct_correct[m] for m in METHODS if m != "PhyloAE")
    unsup_ok = 0.60 <= unsup_best <= 0.85
    shortlist_ok = 3 <= len(cross_method_misplaced) <= 15

    print("Gate check:")
    print(f"  - PhyloAE pct_correct >= 0.90: {phyloae_ok} ({per_method_pct_correct['PhyloAE'] * 100:.2f}%)")
    print(f"  - Best unsupervised in [0.60, 0.85]: {unsup_ok} ({unsup_best * 100:.2f}%)")
    print(f"  - Shortlist size in [3, 15]: {shortlist_ok} ({len(cross_method_misplaced)})")

    print(f"Wrote JSON: {json_out}")
    print(f"Wrote JSON: {internal_json_out}")
    print(f"Wrote figure: {fig_out}")

    if not (unsup_ok and shortlist_ok):
        print("Warning: gate checks failed; investigate best-run selection or data alignment before proceeding.")
    else:
        print("PARKER_CONCORDANCE_READY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Parker clade concordance on best cached embeddings.")
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
