#!/usr/bin/env python3
"""Parker NJ-tree support for the 6 "novel" consensus-misplaced breeds.

For each target breed, this script ranks all 26 Parker clades by mean patristic
distance from the breed's tips to the clade's other tips in Parker et al.'s own
neighbor-joining tree (Parker HG et al., 2017, Cell Reports 19:697-708, supplemental
file mmc5). The assigned Parker clade is then reported as rank k/26 (inclusive of
the Hungarian Puli+Pumi outgroup), which is the ranking cited in the slides.

The script also cross-checks each ranking against an IBS (identity-by-state) proxy
computed from the QC'd/LD-pruned additive genotype matrix via `lib.load_plink_raw`
and `lib.prepare_features`, so the Parker-tree ranking and the raw-SNP ranking can
be compared in one place.

Inputs
------
- data/parker_tree/parker_2017_nj.nex      Parker 2017 supplementary NEXUS tree
- data/parker_clades.csv                   breed_code -> clade mapping
- data/All_Pure_150k.fam                   IID -> FID (breed code) table
- results/qc/all_qc_ldpruned_additive.raw  QC'd additive dosage matrix for IBS

Outputs
-------
- results/metrics/novel_breed_parker_support.json   (persistent artifact)
- /tmp/cbmf4761_agent_logs/parker_nj_novel_support.log  (human-readable run log)

Usage
-----
    python helpers/figures/novel_breed_support.py

Runs from the project root; resolves paths relative to this file's parent.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import lib  # noqa: E402
from helpers.figures.mantel_nj import (  # noqa: E402
    _read_parker_tree,
    _canonical_tip_name,
    _load_iid_to_breed_map,
)

warnings.filterwarnings("ignore")

TARGETS = {
    "AIRT": ("Airedale Terrier", "Terrier"),
    "SILK": ("Silky Terrier", "Terrier"),
    "YORK": ("Yorkshire Terrier", "Terrier"),
    "DALM": ("Dalmatian", "PointerSetter"),
    "WEIM": ("Weimaraner", "PointerSetter"),
    "DANE": ("Great Dane", "EuropeanMastiff"),
}


def classify_support(rank: int | None) -> str:
    if rank is None:
        return "N/A"
    if rank == 1:
        return "STRONG"
    if rank <= 3:
        return "WEAK"
    return "AGAINST"


def main() -> dict:
    # --------------------------------------------------------------------
    # Step 1 - parse Parker NJ tree, build tip -> breed -> clade mapping.
    # --------------------------------------------------------------------
    tree_path = PROJECT_ROOT / "data" / "parker_tree" / "parker_2017_nj.nex"
    clade_csv = PROJECT_ROOT / "data" / "parker_clades.csv"

    tree = _read_parker_tree(tree_path)
    tips = [_canonical_tip_name(t.name) for t in tree.tips()]

    # Tip -> breed code
    iid_to_breed = _load_iid_to_breed_map(PROJECT_ROOT)
    suffix_to_iids: dict[str, list[str]] = {}
    for iid in iid_to_breed:
        if "_" in iid:
            suf = iid.split("_", 1)[1]
            suffix_to_iids.setdefault(suf, []).append(iid)

    tip_to_breed: dict[str, str] = {}
    for tip in tips:
        if tip in iid_to_breed:
            tip_to_breed[tip] = iid_to_breed[tip]
            continue
        if "_" in tip:
            suf = tip.split("_", 1)[1]
            cands = suffix_to_iids.get(suf, [])
            if len(cands) == 1:
                tip_to_breed[tip] = iid_to_breed[cands[0]]
                continue
            tip_to_breed[tip] = tip.split("_", 1)[0]
        else:
            tip_to_breed[tip] = tip

    # breed -> clade
    clade_df = pd.read_csv(clade_csv)
    breed_to_clade = dict(zip(clade_df["breed_code"], clade_df["clade"]))

    # tip -> clade (drop tips whose breed isn't in clade map)
    tip_to_clade: dict[str, str] = {}
    for tip, br in tip_to_breed.items():
        if br in breed_to_clade:
            tip_to_clade[tip] = breed_to_clade[br]

    # Target-breed tip counts
    target_tip_counts = {
        code: sum(1 for b in tip_to_breed.values() if b == code) for code in TARGETS
    }

    # --------------------------------------------------------------------
    # Step 2 - patristic distance matrix (tree), rank clades per target breed.
    # --------------------------------------------------------------------
    patristic_dm = tree.tip_tip_distances()
    dm_ids_raw = list(patristic_dm.ids)
    dm_ids = [_canonical_tip_name(n) for n in dm_ids_raw]
    dm_arr = np.asarray(patristic_dm.data, dtype=float)
    id_to_idx = {name: i for i, name in enumerate(dm_ids)}

    # indices for every clade (all tips in clade), and per target breed
    clade_to_idx: dict[str, list[int]] = defaultdict(list)
    for tip, cl in tip_to_clade.items():
        idx = id_to_idx.get(tip)
        if idx is not None:
            clade_to_idx[cl].append(idx)

    breed_to_idx: dict[str, list[int]] = defaultdict(list)
    for tip, br in tip_to_breed.items():
        idx = id_to_idx.get(tip)
        if idx is not None:
            breed_to_idx[br].append(idx)

    all_clades = sorted(clade_to_idx.keys())
    print(f"Clades in tree mapping: {len(all_clades)}")

    tree_rankings: dict[str, dict] = {}
    for code, (full_name, assigned_clade) in TARGETS.items():
        t_idx = np.asarray(breed_to_idx.get(code, []), dtype=int)
        if len(t_idx) == 0:
            print(f"WARNING: no tips for {code}")
            continue

        per_clade: list[tuple[str, float]] = []
        for cl in all_clades:
            c_idx = np.asarray(clade_to_idx[cl], dtype=int)
            # Exclude target breed's own tips from the reference clade
            c_idx = np.asarray([i for i in c_idx if i not in set(t_idx.tolist())], dtype=int)
            if len(c_idx) == 0:
                # Target breed is the only breed in the clade: skip
                continue
            sub = dm_arr[np.ix_(t_idx, c_idx)]
            per_clade.append((cl, float(sub.mean())))

        per_clade.sort(key=lambda x: x[1])
        # Raw (Hungarian-inclusive) rank: what the final slide reports (rank / 26).
        rank_of_assigned = next(
            (i + 1 for i, (cl, _) in enumerate(per_clade) if cl == assigned_clade),
            None,
        )
        # Hungarian (PULI/PUMI) sits near the tree center and appears "closest"
        # to every breed by mean patristic distance - an artifact of being a
        # basal 2-breed outgroup. Secondary adjusted rank (ex-Hungarian) is kept
        # for transparency / debugging, but the primary reported rank is raw.
        per_clade_adj = [(c, d) for c, d in per_clade if c != "Hungarian"]
        rank_of_assigned_adj = next(
            (i + 1 for i, (cl, _) in enumerate(per_clade_adj) if cl == assigned_clade),
            None,
        )
        tree_rankings[code] = {
            "breed_full": full_name,
            "assigned_clade": assigned_clade,
            "n_tips": int(len(t_idx)),
            "ranked": per_clade,
            "rank_of_assigned": rank_of_assigned,
            "n_clades_ranked": len(per_clade),
            "ranked_ex_hungarian": per_clade_adj,
            "rank_of_assigned_ex_hungarian": rank_of_assigned_adj,
            "n_clades_ranked_ex_hungarian": len(per_clade_adj),
        }

    # --------------------------------------------------------------------
    # Step 3 - IBS cross-check from our QC'd genotype matrix.
    # --------------------------------------------------------------------
    raw_path = PROJECT_ROOT / "results" / "qc" / "all_qc_ldpruned_additive.raw"
    meta_raw, X = lib.load_plink_raw(raw_path)
    meta, X_scaled, *_ = lib.prepare_features(meta_raw, X, clade_csv=clade_csv)

    # Use UNscaled genotype dosages (0/1/2) for breed centroids.
    # L2 distance in unscaled dosage space == a proxy for IBS distance.
    # prepare_features does StandardScaler; we want raw dosages for allele-
    # frequency interpretation, so rebuild centroids from X directly.
    breeds = meta["breed"].astype(str).to_numpy()
    clade_groups = meta["breed_group"].astype(str).to_numpy()

    # per-breed mean allele-frequency vector
    unique_breeds = sorted(set(breeds))
    breed_cent = {}
    breed_clade = {}
    for b in unique_breeds:
        mask = breeds == b
        if not mask.any():
            continue
        breed_cent[b] = X[mask].mean(axis=0)
        breed_clade[b] = clade_groups[mask][0]

    # per-clade centroid excluding each target breed
    ibs_rankings: dict[str, dict] = {}
    for code, (full_name, assigned_clade) in TARGETS.items():
        if code not in breed_cent:
            print(f"WARNING: no genotypes for {code}")
            continue
        target_vec = breed_cent[code]
        # clades populated in this panel
        clades_here = sorted({breed_clade[b] for b in unique_breeds})
        per_clade: list[tuple[str, float]] = []
        for cl in clades_here:
            # breeds in this clade, excluding target itself
            members = [b for b in unique_breeds if breed_clade[b] == cl and b != code]
            if not members:
                continue
            cl_centroid = np.mean([breed_cent[m] for m in members], axis=0)
            d = float(np.linalg.norm(target_vec - cl_centroid))
            per_clade.append((cl, d))
        per_clade.sort(key=lambda x: x[1])
        rank = next(
            (i + 1 for i, (cl, _) in enumerate(per_clade) if cl == assigned_clade), None
        )
        ibs_rankings[code] = {
            "breed_full": full_name,
            "assigned_clade": assigned_clade,
            "ranked": per_clade,
            "rank_of_assigned": rank,
            "n_clades_ranked": len(per_clade),
        }

    # --------------------------------------------------------------------
    # Step 4 - bootstrap / node support on MRCA of breed + its assigned clade.
    # Parker's tree stores internal branch lengths that are NJ-derived values.
    # skbio.TreeNode has tip names only - internal names are None. We check.
    # --------------------------------------------------------------------
    # Inspect whether any internal node has a name/support
    internal_with_name = 0
    for node in tree.traverse(include_self=False):
        if node.is_tip():
            continue
        if node.name not in (None, ""):
            internal_with_name += 1
    bootstrap_available = internal_with_name > 0

    bootstrap_reports: dict[str, dict] = {}
    if bootstrap_available:
        # Build MRCA for target-breed tips + assigned-clade tips, report support
        for code, (full_name, assigned_clade) in TARGETS.items():
            tips_of_interest = [t for t, b in tip_to_breed.items() if b == code]
            tips_of_interest += [
                t for t, c in tip_to_clade.items() if c == assigned_clade
            ]
            try:
                mrca = tree.lowest_common_ancestor(tips_of_interest)
                support = mrca.name if mrca is not None else None
            except Exception as exc:
                support = f"ERROR: {exc}"
            bootstrap_reports[code] = {
                "breed_full": full_name,
                "assigned_clade": assigned_clade,
                "mrca_support": support,
            }

    # --------------------------------------------------------------------
    # Step 5 - interpretation and final print-out.
    # --------------------------------------------------------------------
    lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    p("PARKER_NJ_NOVEL_SUPPORT_DONE")
    p("")
    p("=== Tree stats ===")
    p(f"Tips: {len(tips)} total")
    p(f"Tips mapped to any Parker clade: {len(tip_to_clade)}")
    p(f"Target breeds tips: {target_tip_counts}")
    p("")

    p("=== Per-breed clade rankings (tree patristic) ===")
    p("Primary rank is raw (inclusive of Hungarian) - the rank / 26 reported in the")
    p("final slides. The Hungarian Puli+Pumi clade sits near tree center and is")
    p("rank 1 for 5/6 targets on mean patristic distance (an artifact of a small")
    p("central outgroup, not basal signal); the ex-Hungarian rank is kept below")
    p("for transparency but is NOT the reported number.")
    p("")
    for code, (full_name, assigned) in TARGETS.items():
        r = tree_rankings.get(code)
        if r is None:
            p(f"{full_name} [{code}] (assigned: {assigned}) -- NO TIPS")
            continue
        top3 = r["ranked"][:3]
        top_str = "  ".join(
            [f"{i+1}. {cl} {d:.2f}" for i, (cl, d) in enumerate(top3)]
        )
        support_raw = classify_support(r["rank_of_assigned"])
        support_adj = classify_support(r["rank_of_assigned_ex_hungarian"])
        p(f"{full_name} [{code}] (assigned: {assigned})")
        p(f"  Rank of assigned clade (raw):          "
          f"{r['rank_of_assigned']}/{r['n_clades_ranked']} [{support_raw}]")
        p(f"  Rank of assigned clade (ex-Hungarian): "
          f"{r['rank_of_assigned_ex_hungarian']}/{r['n_clades_ranked_ex_hungarian']} [{support_adj}]")
        p(f"  Top 3 (raw):          {top_str}")
    p("")

    # Aggregate agreement count (raw rank support class is primary)
    agreement = 0
    disagreements: list[str] = []
    for code, (full_name, assigned) in TARGETS.items():
        tr = tree_rankings.get(code)
        ir = ibs_rankings.get(code)
        if tr is None or ir is None:
            continue
        tree_rank = tr["rank_of_assigned"]
        ibs_rank = ir["rank_of_assigned"]
        tree_support = classify_support(tree_rank)
        ibs_support = classify_support(ibs_rank)
        if tree_support == ibs_support:
            agreement += 1
        else:
            disagreements.append(
                f"  {full_name}: tree (raw) rank {tree_rank} [{tree_support}] "
                f"vs IBS rank {ibs_rank} [{ibs_support}]"
            )

    p("=== IBS cross-check ===")
    p(f"Per-breed IBS rankings:")
    for code, (full_name, assigned) in TARGETS.items():
        ir = ibs_rankings.get(code)
        if ir is None:
            p(f"  {full_name} [{code}]: NO DATA")
            continue
        top3 = ir["ranked"][:3]
        top_str = "  ".join([f"{i+1}. {cl} {d:.2f}" for i, (cl, d) in enumerate(top3)])
        support = classify_support(ir["rank_of_assigned"]) if ir["rank_of_assigned"] else "N/A"
        p(f"  {full_name} [{code}] (assigned: {assigned}) -- rank {ir['rank_of_assigned']}/{ir['n_clades_ranked']} [{support}]; top3: {top_str}")
    p("")
    p(f"Agreement between tree (raw) support-class and IBS support-class: {agreement}/{len(TARGETS)} breeds")
    if disagreements:
        p("Disagreements:")
        for d in disagreements:
            p(d)
    else:
        p("No class disagreements.")
    p("")

    p("=== Bootstrap (if available) ===")
    if bootstrap_available:
        for code, (full_name, assigned) in TARGETS.items():
            br = bootstrap_reports.get(code)
            if br:
                p(f"  {full_name} [{code}] MRCA(assigned={assigned}) support: {br['mrca_support']}")
    else:
        p("Parker tree has no explicit bootstrap labels on internal nodes (skbio parse). "
          "Note: Parker 2017 encodes NJ bootstrap % as the *branch-length* on internal edges; "
          "those aren't exposed as discrete node supports by skbio. Skipping per spec.")
    p("")

    # Bucket counts based on the raw (Hungarian-inclusive) rank - matches slides.
    strong: list[str] = []
    weak: list[str] = []
    against: list[str] = []
    for code, (full_name, _) in TARGETS.items():
        tr = tree_rankings.get(code)
        if tr is None:
            continue
        cls = classify_support(tr["rank_of_assigned"])
        if cls == "STRONG":
            strong.append(full_name)
        elif cls == "WEAK":
            weak.append(full_name)
        elif cls == "AGAINST":
            against.append(full_name)

    p("=== Summary (raw rank) ===")
    p(f"Of 6 truly-novel breeds (raw rank / 26, Hungarian included):")
    p(f"  - {len(strong)} STRONG (rank 1):     {strong}")
    p(f"  - {len(weak)}   WEAK   (rank 2-3):    {weak}")
    p(f"  - {len(against)}   AGAINST (rank >=4): {against}")

    # --------------------------------------------------------------------
    # Save JSON artifact + log
    # --------------------------------------------------------------------
    payload = {
        "tree_stats": {
            "n_tips_total": len(tips),
            "n_tips_mapped_to_clade": len(tip_to_clade),
            "target_tip_counts": target_tip_counts,
            "n_clades_in_tree_mapping": len(all_clades),
        },
        "tree_rankings": {
            code: {
                "breed_full": v["breed_full"],
                "assigned_clade": v["assigned_clade"],
                "n_tips": v["n_tips"],
                "rank_of_assigned": v["rank_of_assigned"],
                "n_clades_ranked": v["n_clades_ranked"],
                "top5": [{"clade": cl, "dist": d} for cl, d in v["ranked"][:5]],
                "ranked_all": [{"clade": cl, "dist": d} for cl, d in v["ranked"]],
                "rank_of_assigned_ex_hungarian": v["rank_of_assigned_ex_hungarian"],
                "n_clades_ranked_ex_hungarian": v["n_clades_ranked_ex_hungarian"],
                "top5_ex_hungarian": [
                    {"clade": cl, "dist": d} for cl, d in v["ranked_ex_hungarian"][:5]
                ],
                "support_class_raw": classify_support(v["rank_of_assigned"]),
                "support_class": classify_support(v["rank_of_assigned_ex_hungarian"]),
            }
            for code, v in tree_rankings.items()
        },
        "ibs_rankings": {
            code: {
                "breed_full": v["breed_full"],
                "assigned_clade": v["assigned_clade"],
                "rank_of_assigned": v["rank_of_assigned"],
                "n_clades_ranked": v["n_clades_ranked"],
                "top5": [{"clade": cl, "dist": d} for cl, d in v["ranked"][:5]],
                "ranked_all": [{"clade": cl, "dist": d} for cl, d in v["ranked"]],
                "support_class": classify_support(v["rank_of_assigned"]),
            }
            for code, v in ibs_rankings.items()
        },
        "cross_check": {
            "agreement_support_class": agreement,
            "total": len(TARGETS),
            "disagreements": disagreements,
        },
        "bootstrap": {
            "available": bootstrap_available,
            "reports": bootstrap_reports,
        },
        "summary_buckets": {
            "STRONG": strong,
            "WEAK": weak,
            "AGAINST": against,
        },
    }
    out_json = PROJECT_ROOT / "results" / "metrics" / "novel_breed_parker_support.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log_path = Path("/tmp/cbmf4761_agent_logs/parker_nj_novel_support.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote JSON: {out_json}")
    print(f"Wrote log:  {log_path}")
    return payload


if __name__ == "__main__":
    main()
