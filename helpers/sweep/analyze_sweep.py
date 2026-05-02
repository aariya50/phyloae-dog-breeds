#!/usr/bin/env python3
"""Analyze the PhyloAE distance-metric sweep (euclidean / manhattan / cosine).

Per target_metric:
  - mean + best of trustworthiness, silhouette_breed, seed_stability_vs_base, composite
  - best config row (by composite, n_components=2) -> Mantel-r vs Parker tree, Parker concordance

Writes results/metrics/best_metric.txt with the winner and headline numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.figures import mantel_nj as mnj  # type: ignore  # noqa: E402
from helpers.figures import parker_concordance as pc  # type: ignore  # noqa: E402

SUMMARY_CSV = PROJECT_ROOT / "results" / "metrics" / "phyloae_sweep_summary.csv"
PARKER_TREE = PROJECT_ROOT / "data" / "parker_tree" / "parker_2017_nj.nex"
CLADE_CSV = PROJECT_ROOT / "data" / "parker_clades.csv"
OUT_TXT = PROJECT_ROOT / "results" / "metrics" / "best_metric.txt"


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")
    print(f"Columns: {list(df.columns)}")

    if "target_metric" not in df.columns:
        print("ERROR: target_metric column missing -- sweep may have written legacy schema.")
        sys.exit(1)

    df["composite"] = df["trustworthiness"] + df["silhouette_breed"]
    metrics = ["euclidean", "manhattan", "cosine"]

    # 1+2. Mean + best per metric
    print("\n=== Per-metric summary statistics ===")
    agg_cols = ["trustworthiness", "silhouette_breed", "seed_stability_vs_base", "composite"]
    rows = []
    for tm in metrics:
        sub = df[df["target_metric"] == tm]
        rec = {"target_metric": tm, "n": len(sub)}
        for c in agg_cols:
            rec[f"{c}_mean"] = float(sub[c].mean())
            rec[f"{c}_max"] = float(sub[c].max())
        rows.append(rec)
    stat_df = pd.DataFrame(rows)
    print(stat_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Best config per metric (by composite, restrict to n_components=2 to match downstream figs)
    print("\n=== Best 2D config per metric (by composite = trust + silhouette) ===")
    df2d = df[df["n_components"] == 2].copy()
    best_rows: dict[str, pd.Series] = {}
    for tm in metrics:
        sub = df2d[df2d["target_metric"] == tm].sort_values("composite", ascending=False)
        if sub.empty:
            print(f"  {tm}: no 2D rows!")
            continue
        best_rows[tm] = sub.iloc[0]
        r = best_rows[tm]
        print(
            f"  {tm:9s}  trust={r['trustworthiness']:.4f}  sil={r['silhouette_breed']:.4f}  "
            f"stab={r['seed_stability_vs_base']:.4f}  composite={r['composite']:.4f}  "
            f"hd={r['hidden_dims']} lr={r['lr']} lb={r['lambda_breed']} lc={r['lambda_clade']} seed={int(r['seed'])}"
        )

    # 4. Mantel-r vs Parker tree + Parker concordance per metric's best config
    print("\n=== Loading Parker reference (NJ tree patristic) ===")
    ref_name, ref_breeds, ref_mat, _ref_meta = mnj._build_reference_matrix_parker(
        PROJECT_ROOT, PARKER_TREE
    )
    ref_index = {b: i for i, b in enumerate(ref_breeds)}

    clade_df = pd.read_csv(CLADE_CSV)
    clade_map = dict(zip(clade_df["breed_code"].astype(str), clade_df["clade"].astype(str)))

    per_metric_results: dict[str, dict] = {}

    for tm in metrics:
        if tm not in best_rows:
            continue
        emb_csv = Path(str(best_rows[tm]["embedding_csv"]))
        if not emb_csv.exists():
            print(f"  {tm}: missing embedding CSV {emb_csv}")
            continue

        # Mantel-r
        emb_cent = mnj._breed_centroids_from_embedding(emb_csv, "PhyloAE")
        emb_breeds = emb_cent["breed"].astype(str).tolist()
        shared = sorted(set(emb_breeds) & set(ref_breeds))
        emb_sub = emb_cent.set_index("breed").loc[shared]
        coord_cols = [c for c in emb_sub.columns if c.startswith("coord_")]
        emb_xy = emb_sub[coord_cols].to_numpy(dtype=float)
        ref_idx = [ref_index[b] for b in shared]
        ref_sub = ref_mat[np.ix_(ref_idx, ref_idx)]

        emb_dmat = squareform(pdist(emb_xy, metric="euclidean"))
        mantel_r, mantel_p = mnj._corr_upper_tri(emb_dmat, ref_sub, method="pearson")
        mantel_rs, _ = mnj._corr_upper_tri(emb_dmat, ref_sub, method="spearman")

        # Parker concordance
        per_dog = pc._build_per_dog_df(emb_csv, "PhyloAE", clade_map)
        _, pct_correct = pc._compute_concordance(per_dog)

        per_metric_results[tm] = {
            "trustworthiness": float(best_rows[tm]["trustworthiness"]),
            "silhouette_breed": float(best_rows[tm]["silhouette_breed"]),
            "seed_stability": float(best_rows[tm]["seed_stability_vs_base"]),
            "composite": float(best_rows[tm]["composite"]),
            "mantel_pearson": float(mantel_r),
            "mantel_spearman": float(mantel_rs),
            "parker_concordance": float(pct_correct),
            "embedding_csv": str(emb_csv),
            "run_id": str(best_rows[tm]["run_id"]),
        }

    # 5. Side-by-side table
    print("\n=== Side-by-side per-metric (best 2D config, downstream-relevant numbers) ===")
    hdr = f"{'metric':<10} {'trust':>7} {'sil':>7} {'stab':>7} {'composite':>10} {'mantel_r':>9} {'mantel_rho':>10} {'parker%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for tm in metrics:
        if tm not in per_metric_results:
            continue
        r = per_metric_results[tm]
        print(
            f"{tm:<10} {r['trustworthiness']:>7.4f} {r['silhouette_breed']:>7.4f} {r['seed_stability']:>7.4f} "
            f"{r['composite']:>10.4f} {r['mantel_pearson']:>9.4f} {r['mantel_spearman']:>10.4f} "
            f"{100 * r['parker_concordance']:>7.2f}%"
        )

    # Decide winner: prefer mantel_pearson (primary headline). Tiebreak: parker_concordance.
    winner = max(
        per_metric_results,
        key=lambda m: (per_metric_results[m]["mantel_pearson"], per_metric_results[m]["parker_concordance"]),
    )
    w = per_metric_results[winner]
    print(f"\n=== WINNER: {winner} ===")
    print(
        f"  mantel_r={w['mantel_pearson']:.4f}  parker_concordance={100*w['parker_concordance']:.2f}%  "
        f"trust={w['trustworthiness']:.4f}  composite={w['composite']:.4f}"
    )

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"winner_metric={winner}",
        f"run_id={w['run_id']}",
        f"embedding_csv={w['embedding_csv']}",
        f"mantel_pearson={w['mantel_pearson']:.6f}",
        f"mantel_spearman={w['mantel_spearman']:.6f}",
        f"parker_concordance={w['parker_concordance']:.6f}",
        f"trustworthiness={w['trustworthiness']:.6f}",
        f"silhouette_breed={w['silhouette_breed']:.6f}",
        f"composite={w['composite']:.6f}",
        "",
        "# Per-metric (best 2D config)",
    ]
    for tm in metrics:
        if tm not in per_metric_results:
            continue
        r = per_metric_results[tm]
        lines.append(
            f"{tm}: mantel_r={r['mantel_pearson']:.4f} parker={100*r['parker_concordance']:.2f}% "
            f"trust={r['trustworthiness']:.4f} composite={r['composite']:.4f} run_id={r['run_id']}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote: {OUT_TXT}")


if __name__ == "__main__":
    main()
