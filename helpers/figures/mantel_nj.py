#!/usr/bin/env python3
"""External validation via Mantel-style correlation against NJ-tree distances.

Outputs:
- results/metrics/mantel_nj.json
- results/figures/presentation/mantel_nj.png
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

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
DEFAULT_PARKER_TREE = Path(__file__).resolve().parents[2] / "data" / "parker_tree" / "parker_2017_nj.nex"

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


def _group_centroids(X: np.ndarray, labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    centroids = pd.DataFrame(np.asarray(X)).groupby(np.asarray(labels).astype(str), sort=True).mean()
    return centroids.index.tolist(), centroids.to_numpy()


def _breed_centroids_from_embedding(embedding_csv: Path, method: str) -> pd.DataFrame:
    emb = pd.read_csv(embedding_csv)
    coord_cols = _find_coord_columns(emb, method)

    if "breed" in emb.columns:
        breed = emb["breed"].astype(str)
    elif "FID" in emb.columns:
        breed = emb["FID"].astype(str)
    else:
        raise ValueError(f"No breed/FID column found in {embedding_csv}")

    df = pd.DataFrame({"breed": breed})
    coord_out_cols: list[str] = []
    for idx, col in enumerate(coord_cols, start=1):
        out_col = f"coord_{idx}"
        df[out_col] = emb[col].astype(float)
        coord_out_cols.append(out_col)
    return df.groupby("breed", as_index=False)[coord_out_cols].mean()


def _canonical_tip_name(name: object) -> str:
    return str(name).strip().strip("'").strip('"').replace(" ", "_")


def _load_iid_to_breed_map(project_root: Path) -> dict[str, str]:
    fam_path = project_root / "data" / "All_Pure_150k.fam"
    if fam_path.exists():
        fam = pd.read_csv(
            fam_path,
            sep=r"\s+",
            header=None,
            names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
            usecols=["FID", "IID"],
        )
        return dict(zip(fam["IID"].astype(str), fam["FID"].astype(str)))

    raw_matrix = project_root / "results" / "qc" / "all_qc_ldpruned_additive.raw"
    if raw_matrix.exists():
        meta_raw = pd.read_csv(raw_matrix, sep=r"\s+", usecols=["FID", "IID"])
        return dict(zip(meta_raw["IID"].astype(str), meta_raw["FID"].astype(str)))

    return {}


def _extract_newick_from_nexus(nexus_path: Path) -> str:
    text = nexus_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"tree\s+[^=]+?=\s*(?:\[[^\]]*\]\s*)?(.+?);",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not extract Newick tree from NEXUS file: {nexus_path}")
    newick = match.group(1).replace("\n", "").replace("\r", "").replace("\x0c", "").strip()
    return f"{newick};"


def _read_parker_tree(parker_tree_path: Path):
    from skbio import TreeNode

    parse_errors: list[str] = []
    for fmt in ("nexus", "newick"):
        try:
            return TreeNode.read(str(parker_tree_path), format=fmt)
        except Exception as exc:
            parse_errors.append(f"{fmt}: {type(exc).__name__}: {exc}")

    newick = _extract_newick_from_nexus(parker_tree_path)
    try:
        return TreeNode.read([newick])
    except Exception as exc:
        parse_errors.append(f"newick-extracted: {type(exc).__name__}: {exc}")

    raise ValueError(
        f"Unable to parse Parker tree at {parker_tree_path}. Attempts: {' | '.join(parse_errors)}"
    )


def _build_reference_matrix_nj(project_root: Path) -> tuple[str, list[str], np.ndarray, dict[str, int]]:
    raw_matrix = project_root / "results" / "qc" / "all_qc_ldpruned_additive.raw"
    clade_csv = project_root / "data" / "parker_clades.csv"

    meta_raw, X = lib.load_plink_raw(raw_matrix)
    meta, X_scaled, *_ = lib.prepare_features(meta_raw, X, clade_csv=clade_csv)

    breeds, centroids = _group_centroids(X_scaled, meta["breed"].astype(str).to_numpy())
    raw_dist = squareform(pdist(centroids, metric="euclidean"))

    try:
        from skbio import DistanceMatrix
        from skbio.tree import nj

        tree = nj(DistanceMatrix(raw_dist, ids=breeds))
        patristic_dm = tree.tip_tip_distances()
        dm_ids = list(patristic_dm.ids)
        dm_arr = np.asarray(patristic_dm.data, dtype=float)

        id_to_idx = {name: i for i, name in enumerate(dm_ids)}
        order_idx = [id_to_idx[b] for b in breeds]
        patristic = dm_arr[np.ix_(order_idx, order_idx)]
        return "nj_tree_patristic", breeds, patristic, {"reference_breed_count": len(breeds)}
    except Exception as exc:
        print(f"Warning: could not build/load NJ patristic distances ({exc}); using raw SNP centroids.")
        return "raw_snp_centroid", breeds, raw_dist, {"reference_breed_count": len(breeds)}


def _build_reference_matrix_parker(
    project_root: Path,
    parker_tree_path: Path,
) -> tuple[str, list[str], np.ndarray, dict[str, int]]:
    if not parker_tree_path.exists():
        raise FileNotFoundError(f"Parker tree file not found: {parker_tree_path}")

    tree = _read_parker_tree(parker_tree_path)
    tips = [_canonical_tip_name(tip.name) for tip in tree.tips()]
    iid_to_breed = _load_iid_to_breed_map(project_root)

    suffix_to_iids: dict[str, list[str]] = {}
    for iid in iid_to_breed:
        if "_" in iid:
            suffix = iid.split("_", 1)[1]
            suffix_to_iids.setdefault(suffix, []).append(iid)

    exact_hits = 0
    suffix_hits = 0
    prefix_hits = 0
    tip_to_breed: dict[str, str] = {}
    for tip in tips:
        if tip in iid_to_breed:
            tip_to_breed[tip] = iid_to_breed[tip]
            exact_hits += 1
            continue
        if "_" in tip:
            suffix = tip.split("_", 1)[1]
            candidates = suffix_to_iids.get(suffix, [])
            if len(candidates) == 1:
                tip_to_breed[tip] = iid_to_breed[candidates[0]]
                suffix_hits += 1
                continue
            tip_to_breed[tip] = tip.split("_", 1)[0]
        else:
            tip_to_breed[tip] = tip
        prefix_hits += 1

    patristic_dm = tree.tip_tip_distances()
    dm_ids = [_canonical_tip_name(name) for name in patristic_dm.ids]
    dm_arr = np.asarray(patristic_dm.data, dtype=float)
    if len(dm_ids) != len(set(dm_ids)):
        raise ValueError("Parker tree tip IDs are not unique after canonicalization.")

    id_to_idx = {name: i for i, name in enumerate(dm_ids)}
    breed_to_indices: dict[str, list[int]] = {}
    for tip in tips:
        idx = id_to_idx.get(tip)
        if idx is None:
            raise ValueError(f"Parker tip '{tip}' not found in patristic matrix IDs.")
        breed = tip_to_breed[tip]
        breed_to_indices.setdefault(breed, []).append(idx)

    breeds = sorted(breed_to_indices.keys())
    n = len(breeds)
    breed_dist = np.zeros((n, n), dtype=float)
    for i, breed_a in enumerate(breeds):
        idx_a = np.asarray(breed_to_indices[breed_a], dtype=int)
        for j in range(i + 1, n):
            breed_b = breeds[j]
            idx_b = np.asarray(breed_to_indices[breed_b], dtype=int)
            mean_dist = float(dm_arr[np.ix_(idx_a, idx_b)].mean())
            breed_dist[i, j] = mean_dist
            breed_dist[j, i] = mean_dist

    meta = {
        "tree_tip_count": len(tips),
        "reference_breed_count": len(breeds),
        "mapped_by_iid_count": int(exact_hits),
        "mapped_by_suffix_count": int(suffix_hits),
        "mapped_by_prefix_count": int(prefix_hits),
    }
    return "parker_2017_patristic_breed_mean", breeds, breed_dist, meta


def _build_reference_matrix(
    project_root: Path,
    reference: str,
    parker_tree_path: Path,
) -> tuple[str, list[str], np.ndarray, dict[str, int]]:
    if reference == "parker":
        return _build_reference_matrix_parker(project_root, parker_tree_path)
    if reference == "nj":
        return _build_reference_matrix_nj(project_root)
    raise ValueError(f"Unsupported reference: {reference}")


def _corr_upper_tri(
    a: np.ndarray, b: np.ndarray, method: str = "pearson"
) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError(f"Matrix shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] < 3:
        raise ValueError("Need at least 3 breeds to compute correlation.")

    tri = np.triu_indices(a.shape[0], k=1)
    av = a[tri]
    bv = b[tri]
    if method == "spearman":
        r, p = spearmanr(av, bv)
    else:
        r, p = pearsonr(av, bv)
    return float(r), float(p)


COMBOS = ["euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman"]
COMBO_LABELS = ["Euclid-Pearson", "Euclid-Spearman", "Manhattan-Pearson", "Manhattan-Spearman"]
COMBO_COLORS = ["#F6AD55", "#63B3ED", "#68D391", "#FC8181"]


def _plot_results_grouped(
    per_method_mantel: dict[str, dict[str, float]], out_path: Path, reference_name: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    methods_sorted = sorted(
        per_method_mantel.keys(),
        key=lambda m: per_method_mantel[m].get("euclidean_pearson", 0),
        reverse=True,
    )

    n_methods = len(methods_sorted)
    n_combos = len(COMBOS)
    bar_height = 0.18
    y_positions = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=220, facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    for i, (combo, color, label) in enumerate(zip(COMBOS, COMBO_COLORS, COMBO_LABELS)):
        offsets = y_positions + (i - n_combos / 2 + 0.5) * bar_height
        values = [per_method_mantel[m].get(combo, 0.0) for m in methods_sorted]
        bars = ax.barh(offsets, values, height=bar_height, color=color, alpha=0.90, label=label)
        for bar, val in zip(bars, values):
            x = float(bar.get_width())
            ax.text(
                x + 0.008,
                float(bar.get_y()) + float(bar.get_height()) / 2.0,
                f"{val:.3f}",
                va="center",
                ha="left",
                color="white",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods_sorted)
    ax.invert_yaxis()

    all_vals = [v for m in per_method_mantel.values() for v in m.values()]
    xmin = min(min(all_vals) - 0.05, -0.05)
    xmax = max(max(all_vals) + 0.10, 0.10)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Mantel r on pairwise breed distances", color="white", fontsize=12)
    ax.set_title(
        f"Mantel Sensitivity: Distance × Correlation vs {reference_name}",
        color="white",
        fontsize=16,
        pad=12,
    )
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(axis="x", color="white", alpha=0.12, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#364152")
    ax.legend(loc="lower right", fontsize=9, facecolor="#1A202C", edgecolor="#364152", labelcolor="white")

    fig.savefig(out_path, bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)


def run(project_root: Path, reference: str, parker_tree_path: Path) -> None:
    results_dir = project_root / "results"
    metrics_dir = results_dir / "metrics"

    json_out = metrics_dir / "mantel_nj.json"
    fig_out = results_dir / "figures" / "presentation" / "mantel_nj.png"

    reference_name, ref_breeds, ref_mat, reference_meta = _build_reference_matrix(
        project_root, reference=reference, parker_tree_path=parker_tree_path
    )
    ref_index = {b: i for i, b in enumerate(ref_breeds)}

    per_method_r: dict[str, float] = {}
    per_method_mantel: dict[str, dict[str, float]] = {}
    p_values_legacy: dict[str, float] = {}
    p_values: dict[str, dict[str, float]] = {}
    dimensions_used: dict[str, int] = {}
    shared_breed_counts: dict[str, int] = {}

    dist_metrics = [("euclidean", "euclidean"), ("manhattan", "cityblock")]
    corr_methods = ["pearson", "spearman"]

    for method in METHODS:
        emb_csv, n_components = _select_best_embedding_csv(metrics_dir, method)
        emb_cent = _breed_centroids_from_embedding(emb_csv, method)
        dimensions_used[method] = n_components

        emb_breeds = emb_cent["breed"].astype(str).tolist()
        shared = sorted(set(emb_breeds) & set(ref_breeds))
        shared_breed_counts[method] = len(shared)
        if len(shared) < 3:
            raise ValueError(f"{method}: only {len(shared)} shared breeds with reference; need >= 3.")

        emb_sub = emb_cent.set_index("breed").loc[shared]
        coord_cols = [c for c in emb_sub.columns if c.startswith("coord_")]
        emb_xy = emb_sub[coord_cols].to_numpy(dtype=float)

        ref_idx = [ref_index[b] for b in shared]
        ref_sub = ref_mat[np.ix_(ref_idx, ref_idx)]

        method_r: dict[str, float] = {}
        method_p: dict[str, float] = {}

        for dist_label, dist_scipy in dist_metrics:
            emb_mat = squareform(pdist(emb_xy, metric=dist_scipy))
            for corr in corr_methods:
                key = f"{dist_label}_{corr}"
                r, p = _corr_upper_tri(emb_mat, ref_sub, method=corr)
                method_r[key] = r
                method_p[key] = p

        per_method_mantel[method] = method_r
        p_values[method] = method_p
        per_method_r[method] = method_r["euclidean_pearson"]
        p_values_legacy[method] = method_p["euclidean_pearson"]

    payload = {
        "reference": reference_name,
        "reference_meta": reference_meta,
        "per_method_mantel_r": per_method_r,
        "per_method_mantel": per_method_mantel,
        "p_values": p_values,
        "dimensions_used": dimensions_used,
        "shared_breed_counts": shared_breed_counts,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _plot_results_grouped(per_method_mantel, fig_out, reference_name=reference_name)

    # Summary table
    header = f"{'Method':<10} {'Euclid-Pearson':>15} {'Euclid-Spearman':>16} {'Manhattan-Pearson':>18} {'Manhattan-Spearman':>19}"
    print(f"\nReference matrix: {reference_name}")
    print(header)
    for method in METHODS:
        mr = per_method_mantel[method]
        print(
            f"{method:<10} {mr['euclidean_pearson']:>15.3f} {mr['euclidean_spearman']:>16.3f} "
            f"{mr['manhattan_pearson']:>18.3f} {mr['manhattan_spearman']:>19.3f}"
        )

    print(f"\nWrote JSON: {json_out}")
    print(f"Wrote figure: {fig_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute external validation correlations between embedding distances and NJ-tree distances."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to Project root (default: inferred from script location).",
    )
    parser.add_argument(
        "--reference",
        choices=["parker", "nj"],
        default="parker",
        help="Reference matrix source: Parker 2017 NJ tree ('parker') or local NJ from SNP centroids ('nj').",
    )
    parser.add_argument(
        "--parker-tree",
        type=Path,
        default=DEFAULT_PARKER_TREE,
        help="Path to Parker 2017 NJ tree file (NEXUS/Newick). Used when --reference parker.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.project_root.resolve(),
        reference=str(args.reference),
        parker_tree_path=args.parker_tree.expanduser(),
    )
