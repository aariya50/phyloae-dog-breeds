"""Shared library for CBMF4761 UMAP + ADMIXTURE project."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import phate
import plotly.express as px
import trimap
import umap
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NUM_CORES = os.cpu_count() or 1
ADMIX_ELBOW_MIN_K = 2
ADMIX_ELBOW_MAX_K = 15
ADMIX_ELBOW_DELTA_THRESHOLD = 0.002
ADMIX_GAP_MIN_K = 16
ADMIX_GAP_MAX_K = 22
ANALYSIS_SEED = 7
EPSILON = 1e-12

# ---------------------------------------------------------------------------
# 1) Dark theme
# ---------------------------------------------------------------------------

DEFAULT_THEME = {
    'fig_bg': '#0b1020',
    'ax_bg': '#111827',
    'grid': '#334155',
    'text': '#e5e7eb',
    'muted': '#94a3b8',
    'plotly_template': 'plotly_dark',
}


def apply_dark_theme(theme: dict | None = None) -> dict:
    """Apply dark matplotlib rcParams and return resolved theme dict."""
    t = {**DEFAULT_THEME, **(theme or {})}
    plt.rcParams.update({
        'figure.facecolor': t['fig_bg'],
        'axes.facecolor': t['ax_bg'],
        'savefig.facecolor': t['fig_bg'],
        'text.color': t['text'],
        'axes.labelcolor': t['text'],
        'axes.edgecolor': t['grid'],
        'axes.titlecolor': t['text'],
        'xtick.color': t['text'],
        'ytick.color': t['text'],
        'grid.color': t['grid'],
        'legend.facecolor': 'none',
        'legend.edgecolor': 'none',
        'font.size': 10,
    })
    return t


# ---------------------------------------------------------------------------
# 2) Shell helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path | None = None, capture: bool = False,
            quiet: bool = False) -> str:
    """Run a shell command."""
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=capture or quiet,
    )
    return (result.stdout or '') + (result.stderr or '') if (capture or quiet) else ''


def outputs_up_to_date(outputs: list[Path | str], inputs: list[Path | str] | None = None) -> bool:
    """Return True when all outputs exist and are newer than all existing inputs."""
    out_paths = [Path(p) for p in outputs]
    if not out_paths or any(not p.exists() for p in out_paths):
        return False
    in_paths = [Path(p) for p in (inputs or []) if Path(p).exists()]
    if not in_paths:
        return True
    newest_input = max(p.stat().st_mtime for p in in_paths)
    oldest_output = min(p.stat().st_mtime for p in out_paths)
    return oldest_output >= newest_input


def load_or_compute(cache_path: Path | str, compute_fn, force: bool = False,
                    fmt: str = 'parquet'):
    """Load cached object, or compute and persist it.

    Supported formats:
    - ``parquet``: pandas DataFrame
    - ``csv`` / ``tsv``: pandas DataFrame
    - ``npz``: dict[str, np.ndarray]
    """
    cache_path = Path(cache_path)
    fmt = str(fmt).lower()

    if cache_path.exists() and not force:
        if fmt == 'parquet':
            return pd.read_parquet(cache_path)
        if fmt == 'csv':
            return pd.read_csv(cache_path)
        if fmt == 'tsv':
            return pd.read_csv(cache_path, sep='\t')
        if fmt == 'npz':
            with np.load(cache_path, allow_pickle=False) as loaded:
                return {k: loaded[k] for k in loaded.files}
        raise ValueError(f'Unsupported cache format: {fmt}')

    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'parquet':
        result.to_parquet(cache_path)
    elif fmt == 'csv':
        result.to_csv(cache_path, index=False)
    elif fmt == 'tsv':
        result.to_csv(cache_path, sep='\t', index=False)
    elif fmt == 'npz':
        if not isinstance(result, dict):
            raise TypeError("npz cache expects a dict[str, array-like] result.")
        np.savez_compressed(cache_path, **{k: np.asarray(v) for k, v in result.items()})
    else:
        raise ValueError(f'Unsupported cache format: {fmt}')
    return result


def _resolve_tool(name: str, fallback: Path) -> str:
    found = shutil.which(name)
    if found:
        return found
    if fallback.exists() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise FileNotFoundError(f"Could not find '{name}' or fallback '{fallback}'.")


@dataclass
class ToolPaths:
    plink2: str
    admixture: str
    admixture_prefix: list[str]


def check_environment(root: Path, raw_prefix: Path) -> ToolPaths:
    """Resolve plink2/admixture binaries and verify raw inputs exist."""
    plink2 = _resolve_tool('plink2', root / '.tools' / 'bin' / 'plink2')
    admixture = _resolve_tool('admixture', root / '.tools' / 'bin' / 'admixture')

    run_cmd([plink2, '--version'], quiet=True)

    is_mac_arm = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    prefix = ['/usr/bin/arch', '-x86_64'] if is_mac_arm else []
    run_cmd([*prefix, admixture, '--help'], quiet=True)

    for ext in ['bed', 'bim', 'fam']:
        p = Path(f"{raw_prefix}.{ext}")
        if not p.exists():
            raise FileNotFoundError(f"Missing raw input: {p}")

    print('Complete.')
    return ToolPaths(plink2=plink2, admixture=admixture, admixture_prefix=prefix)


def find_repo_root(start: Path) -> Path:
    """Walk up from *start* to find the repo root (contains helpers/ + requirements.txt)."""
    for candidate in [start, *start.parents]:
        if (candidate / 'helpers').is_dir() and (candidate / 'requirements.txt').exists():
            return candidate
    raise RuntimeError('Could not find repository root.')


# ---------------------------------------------------------------------------
# 2.5) Manifest + preflight
# ---------------------------------------------------------------------------

def _sha256_sample(path: Path, sample_bytes: int = 1_048_576) -> str:
    """Fast fingerprint: hash first+last sample_bytes (or whole file if small)."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open('rb') as f:
        if size <= 2 * sample_bytes:
            h.update(f.read())
        else:
            h.update(f.read(sample_bytes))
            f.seek(max(size - sample_bytes, 0))
            h.update(f.read(sample_bytes))
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict:
    """Return stable metadata + sampled hash for reproducibility manifests."""
    st = path.stat()
    return {
        'path': str(path),
        'size': int(st.st_size),
        'mtime': float(st.st_mtime),
        'sha256_sample': _sha256_sample(path),
    }


def count_file_lines(path: Path) -> int:
    """Count lines in a text file without loading into memory."""
    n = 0
    with path.open('rb') as fh:
        for _ in fh:
            n += 1
    return n


def build_run_manifest(root: Path, raw_prefix: Path, pruned_prefix: Path, raw_matrix: Path,
                       admix_dir: Path, run_prefix: str, k_inventory: list[int],
                       seeds: list[int], output_path: Path, *,
                       method_summary_paths: dict[str, Path] | None = None) -> dict:
    """Build and persist run manifest for deterministic, cache-first analysis."""
    required_inputs = {
        'raw_bed': raw_prefix.with_suffix('.bed'),
        'raw_bim': raw_prefix.with_suffix('.bim'),
        'raw_fam': raw_prefix.with_suffix('.fam'),
        'pruned_bed': pruned_prefix.with_suffix('.bed'),
        'pruned_bim': pruned_prefix.with_suffix('.bim'),
        'pruned_fam': pruned_prefix.with_suffix('.fam'),
        'raw_matrix': raw_matrix,
    }
    fingerprints = {}
    for key, p in required_inputs.items():
        if not p.exists():
            raise FileNotFoundError(f'Missing required input for manifest: {p}')
        fingerprints[key] = file_fingerprint(p)

    available_q = sorted(
        _extract_k_from_q_path(p, run_prefix)
        for p in admix_dir.glob(f'{run_prefix}.*.Q')
        if _extract_k_from_q_path(p, run_prefix) is not None
    )
    available_q = [int(k) for k in available_q]
    method_paths = {k: str(v) for k, v in (method_summary_paths or {}).items()}

    manifest = {
        'root': str(root),
        'analysis_seed': ANALYSIS_SEED,
        'frozen_k_inventory': [int(k) for k in k_inventory],
        'frozen_seeds': [int(s) for s in seeds],
        'available_q_from_cache': available_q,
        'inputs': fingerprints,
        'method_summary_paths': method_paths,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def preflight_closed_form(pruned_prefix: Path, raw_matrix: Path, admix_dir: Path, run_prefix: str,
                          k_inventory: list[int], n_samples_expected: int,
                          method_summaries: dict[str, pd.DataFrame]) -> dict:
    """Fail-fast validation for invariant inputs and cache completeness."""
    checks: dict[str, object] = {}
    for ext in ['bed', 'bim', 'fam']:
        p = pruned_prefix.with_suffix(f'.{ext}')
        if not p.exists():
            raise FileNotFoundError(f'Missing LD-pruned input: {p}')
    if not raw_matrix.exists():
        raise FileNotFoundError(f'Missing additive matrix: {raw_matrix}')

    k_inventory = sorted(set(int(k) for k in k_inventory))
    missing_q = []
    missing_p = []
    missing_log = []
    bad_q_rows = []
    for k in k_inventory:
        q = admix_dir / f'{run_prefix}.{k}.Q'
        p = admix_dir / f'{run_prefix}.{k}.P'
        lg = admix_dir / f'log_k{k}.out'
        if not q.exists():
            missing_q.append(k)
        if not p.exists():
            missing_p.append(k)
        if not lg.exists():
            missing_log.append(k)
        if q.exists():
            q_rows = count_file_lines(q)
            if q_rows != n_samples_expected:
                bad_q_rows.append({'K': k, 'q_rows': q_rows, 'expected': n_samples_expected})
    if missing_q or missing_p or missing_log or bad_q_rows:
        raise ValueError(
            f'ADMIXTURE cache incomplete: missing_q={missing_q}, missing_p={missing_p}, '
            f'missing_log={missing_log}, bad_q_rows={bad_q_rows}'
        )

    summary_counts = {}
    missing_embeddings = {}
    for method, df in method_summaries.items():
        if 'embedding_csv' not in df.columns:
            raise ValueError(f'{method} summary missing embedding_csv column.')
        paths = df['embedding_csv'].astype(str).tolist()
        missing = [p for p in paths if p and not Path(p).exists()]
        summary_counts[method] = int(len(df))
        if missing:
            missing_embeddings[method] = missing[:20]
    if missing_embeddings:
        raise ValueError(f'Missing embedding cache files referenced by summaries: {missing_embeddings}')

    checks['k_inventory'] = k_inventory
    checks['n_samples_expected'] = int(n_samples_expected)
    checks['summary_rows'] = summary_counts
    checks['admixture_cache_ok'] = True
    checks['embedding_cache_ok'] = True
    return checks


# ---------------------------------------------------------------------------
# 3) Data loading & preprocessing
# ---------------------------------------------------------------------------

def run_plink_preprocessing(
    plink2: str, raw_prefix: Path, qc_prefix: Path, pruned_prefix: Path,
    raw_matrix: Path, *,
    geno: float, mind: float, maf: float,
    ld_window: int, ld_step: int, ld_r2: float,
) -> None:
    """Run the 4-step PLINK preprocessing pipeline (QC, LD prune, autosome filter, export)."""
    run_cmd([plink2, '--dog', '--bfile', str(raw_prefix), '--threads', '8',
             '--geno', str(geno), '--mind', str(mind), '--maf', str(maf),
             '--make-bed', '--out', str(qc_prefix)], quiet=True)

    run_cmd([plink2, '--dog', '--bfile', str(qc_prefix), '--threads', '8',
             '--indep-pairwise', str(ld_window), str(ld_step), str(ld_r2),
             '--out', str(qc_prefix)], quiet=True)

    run_cmd([plink2, '--dog', '--bfile', str(qc_prefix), '--threads', '8',
             '--extract', str(qc_prefix) + '.prune.in', '--autosome',
             '--make-bed', '--out', str(pruned_prefix)], quiet=True)

    run_cmd([plink2, '--dog', '--bfile', str(pruned_prefix), '--threads', '8',
             '--export', 'A', '--out', str(raw_matrix).replace('.raw', '')], quiet=True)

    print('Complete.')


META_COLUMNS = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']


def load_plink_raw(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load PLINK additive .raw matrix, return (meta, X) with NaN imputation."""
    df = pd.read_csv(path, sep=r'\s+')
    if '#FID' in df.columns:
        df = df.rename(columns={'#FID': 'FID'})
    for req in ['FID', 'IID']:
        if req not in df.columns:
            raise ValueError(f'Missing column: {req}')
    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(dtype=float, copy=True)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        rr, cc = np.where(np.isnan(X))
        X[rr, cc] = np.take(col_means, cc)
    return df[meta_cols].copy(), X


def prepare_features(meta: pd.DataFrame, X: np.ndarray, *, clade_csv=None):
    """Add breed columns, build color maps, scale features.

    When *clade_csv* is provided, breeds are grouped by Parker et al. clade
    and colors are assigned per-clade (one hue each, lightness varies within).
    Clade hue order is determined by hierarchical clustering of clade
    centroids in the scaled genotype space, so genetically-similar clades
    receive adjacent hues. Returns 8 values: meta, X_scaled, breed_order,
    color_mpl, color_plotly, clade_order, clade_color_mpl, clade_color_plotly.
    """
    import colorsys
    from scipy.cluster.hierarchy import linkage, leaves_list

    meta = meta.copy()
    labels = meta['FID'].astype(str)
    meta['breed'] = labels

    if clade_csv is not None:
        clade_df = pd.read_csv(clade_csv)
        clade_map = dict(zip(clade_df['breed_code'], clade_df['clade']))
        meta['breed_group'] = meta['breed'].map(
            lambda b: clade_map.get(b, clade_map.get(b.split('_')[0], 'Unassigned')))
    else:
        meta['breed_group'] = labels

    breed_order = sorted(meta['breed'].unique().tolist())
    X_scaled = StandardScaler().fit_transform(X)

    # Order clades by genetic similarity: hierarchical-cluster leaf order
    # over clade centroids in X_scaled, so similar clades get adjacent hues.
    clades_alpha = sorted(meta['breed_group'].unique().tolist())
    group_arr = meta['breed_group'].to_numpy()
    if len(clades_alpha) > 2:
        centroids = np.array([X_scaled[group_arr == c].mean(axis=0) for c in clades_alpha])
        Z = linkage(centroids, method='average', metric='euclidean')
        clade_order = [clades_alpha[i] for i in leaves_list(Z)]
    else:
        clade_order = clades_alpha

    # Build color scheme: one hue per clade, lightness varies within clade
    hues = np.linspace(0, 1, len(clade_order), endpoint=False)
    clade_hue = dict(zip(clade_order, hues))

    color_mpl: dict[str, tuple] = {}
    for clade in clade_order:
        breeds_in = sorted(meta.loc[meta['breed_group'] == clade, 'breed'].unique())
        h = clade_hue[clade]
        for j, b in enumerate(breeds_in):
            lightness = 0.35 + 0.35 * j / max(len(breeds_in) - 1, 1)
            r, g, b_ = colorsys.hls_to_rgb(h, lightness, 0.75)
            color_mpl[b] = (r, g, b_, 1.0)

    clade_color_mpl = {c: (*colorsys.hls_to_rgb(clade_hue[c], 0.5, 0.75), 1.0)
                       for c in clade_order}

    color_plotly = {b: mcolors.to_hex(c, keep_alpha=False) for b, c in color_mpl.items()}
    clade_color_plotly = {c: mcolors.to_hex(v, keep_alpha=False)
                          for c, v in clade_color_mpl.items()}

    return (meta, X_scaled, breed_order, color_mpl, color_plotly,
            clade_order, clade_color_mpl, clade_color_plotly)


# ---------------------------------------------------------------------------
# 4) Metrics
# ---------------------------------------------------------------------------

def silhouette_by_label(embedding: np.ndarray, labels: pd.Series) -> float:
    """Silhouette score filtering labels with < 2 samples."""
    counts = labels.value_counts()
    keep = labels.isin(counts[counts >= 2].index)
    if keep.sum() < 3:
        return float('nan')
    kept = labels[keep]
    if kept.nunique() < 2:
        return float('nan')
    return float(silhouette_score(embedding[keep.to_numpy()], kept.to_numpy()))


def knn_overlap_score(a: np.ndarray, b: np.ndarray, k: int = 15) -> float:
    """Mean k-NN overlap between two embeddings of the same data."""
    k_eff = min(k, a.shape[0] - 1)
    if k_eff <= 0:
        return float('nan')
    idx_a = NearestNeighbors(n_neighbors=k_eff + 1).fit(a).kneighbors(return_distance=False)[:, 1:]
    idx_b = NearestNeighbors(n_neighbors=k_eff + 1).fit(b).kneighbors(return_distance=False)[:, 1:]
    overlaps = [len(set(ra.tolist()) & set(rb.tolist())) / k_eff for ra, rb in zip(idx_a, idx_b)]
    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# 5) Embedding sweeps
# ---------------------------------------------------------------------------

def select_best_run(summary: pd.DataFrame, n_components: int = 2) -> dict:
    """Pick the best hyperparameter combo via equal-weight composite of all 3 metrics."""
    sub = summary[summary['n_components'] == n_components].copy()
    metrics = ['trustworthiness', 'silhouette_breed', 'seed_stability_vs_base']
    for m in metrics:
        sub[f'{m}_rank'] = sub[m].rank(pct=True)
    sub['composite'] = sum(sub[f'{m}_rank'] for m in metrics) / len(metrics)
    return sub.loc[sub['composite'].idxmax()].to_dict()


def _umap_run_id(metric: str, nn: int, md: float, seed: int, dims: int) -> str:
    return f'umap_{metric}_nn{nn}_md{str(md).replace(".", "p")}_seed{seed}_{dims}d'


def _pacmap_run_id(nn: int, mn: float, fp: float, distance: str, seed: int, dims: int) -> str:
    mn_t = str(mn).replace('.', 'p')
    fp_t = str(fp).replace('.', 'p')
    return f'pacmap_{distance}_nn{nn}_mn{mn_t}_fp{fp_t}_seed{seed}_{dims}d'


def _tsne_run_id(perplexity: float, metric: str, seed: int, dims: int) -> str:
    return f'tsne_{metric}_perp{int(perplexity)}_seed{seed}_{dims}d'


def _trimap_run_id(n_inliers: int, n_outliers: int, distance: str, seed: int, dims: int,
                   weight_temp: float = 0.5) -> str:
    base = f'trimap_{distance}_in{n_inliers}_out{n_outliers}_seed{seed}_{dims}d'
    if weight_temp != 0.5:
        wt_s = str(weight_temp).replace('.', 'p')
        base += f'_wt{wt_s}'
    return base


def _phate_run_id(knn: int, decay: int, knn_dist: str, seed: int, dims: int,
                  t='auto', gamma: int = 1) -> str:
    base = f'phate_{knn_dist}_knn{knn}_decay{int(decay)}_seed{seed}_{dims}d'
    if t != 'auto':
        base += f'_t{t}'
    if gamma != 1:
        base += f'_g{gamma}'
    return base


def _load_embedding_from_csv(csv_path: Path, prefix: str, n_components: int) -> np.ndarray:
    """Load just the embedding coordinates from a cached CSV."""
    df = pd.read_csv(csv_path)
    emb_cols = [f'{prefix}{i+1}' for i in range(n_components)]
    return df[emb_cols].to_numpy(dtype=float)


def _load_cached_one(X_scaled, breed_col, params, id_fn, prefix, key_order, embed_dir):
    """Load a single cached embedding and recompute its metrics."""
    rid = id_fn(**params)
    dims = params['n_components']
    key = tuple(params[k] for k in key_order) + (dims, params['seed'])

    emb = _load_embedding_from_csv(embed_dir / f'{rid}.csv', prefix, dims)

    trust = float(trustworthiness(X_scaled, emb, n_neighbors=min(15, X_scaled.shape[0] - 1)))
    sil = silhouette_by_label(emb, breed_col)

    record = {
        'run_id': rid, **params,
        'trustworthiness': trust,
        'silhouette_breed': sil,
        'seed_stability_vs_base': np.nan,
        'embedding_csv': str(embed_dir / f'{rid}.csv'),
    }
    return key, emb, record


def _run_one(X_scaled, meta_reset, breed_col, params, fit_fn, id_fn, prefix, key_order, embed_dir):
    """Run a single embedding config (called in parallel)."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter('ignore')
        rid = id_fn(**params)
        dims = params['n_components']
        key = tuple(params[k] for k in key_order) + (dims, params['seed'])

        try:
            emb = fit_fn(X_scaled, **params)
        except np.linalg.LinAlgError:
            emb = np.full((X_scaled.shape[0], dims), np.nan)
            record = {
                'run_id': rid, **params,
                'trustworthiness': np.nan,
                'silhouette_breed': np.nan,
                'seed_stability_vs_base': np.nan,
                'embedding_csv': '',
            }
            return key, emb, record

        emb_cols = [f'{prefix}{i+1}' for i in range(dims)]
        emb_df = pd.concat([meta_reset, pd.DataFrame(emb, columns=emb_cols)], axis=1)
        emb_path = embed_dir / f'{rid}.csv'
        tmp_path = emb_path.with_suffix('.csv.tmp')
        emb_df.to_csv(tmp_path, index=False)
        tmp_path.rename(emb_path)

        trust = float(trustworthiness(X_scaled, emb, n_neighbors=min(15, X_scaled.shape[0] - 1)))
        sil = silhouette_by_label(emb, breed_col)

        record = {
            'run_id': rid, **params,
            'trustworthiness': trust,
            'silhouette_breed': sil,
            'seed_stability_vs_base': np.nan,
            'embedding_csv': str(emb_path),
        }
        return key, emb, record


def _run_sweep(X_scaled, meta, *, param_grid, fit_fn, id_fn, prefix, embed_dir, metric_dir):
    """Generic parallel sweep runner with per-run caching."""
    seeds = param_grid['seeds']
    dims_list = param_grid['n_components']
    key_order = param_grid['key_order']
    meta_reset = meta.reset_index(drop=True)
    breed_col = meta['breed']

    # Build all param dicts
    all_params = []
    for combo in param_grid['combos']:
        for dims in dims_list:
            for seed in seeds:
                all_params.append({**combo, 'n_components': dims, 'seed': seed})

    # Partition into cached (CSV exists) vs new (needs computation)
    cached_params = []
    new_params = []
    for p in all_params:
        rid = id_fn(**p)
        if (embed_dir / f'{rid}.csv').exists():
            cached_params.append(p)
        else:
            new_params.append(p)

    total = len(all_params)
    n_cached = len(cached_params)
    n_new = len(new_params)
    n_jobs = min(NUM_CORES, max(n_cached, n_new, 1))
    print(f'{prefix} sweep: {total} total ({n_cached} cached, {n_new} to compute) ...', end=' ', flush=True)

    # Load cached embeddings in parallel
    cached_results = []
    if cached_params:
        cached_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_load_cached_one)(X_scaled, breed_col, p, id_fn, prefix, key_order, embed_dir)
            for p in cached_params
        )

    # Compute new embeddings in parallel
    new_results = []
    if new_params:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter('ignore')
            new_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_one)(X_scaled, meta_reset, breed_col, p, fit_fn, id_fn,
                                  prefix, key_order, embed_dir)
                for p in new_params
            )

    results = list(cached_results) + list(new_results)

    emb_dict = {}
    records = []
    for key, emb, record in results:
        emb_dict[key] = emb
        records.append(record)

    # Seed stability
    idx_map = {r['run_id']: i for i, r in enumerate(records)}
    base_seed = min(seeds)
    for combo in param_grid['combos']:
        for dims in dims_list:
            base_params = {**combo, 'n_components': dims, 'seed': base_seed}
            base_id = id_fn(**base_params)
            base_key = tuple(base_params[k] for k in key_order) + (dims, base_seed)
            records[idx_map[base_id]]['seed_stability_vs_base'] = 1.0
            for seed in seeds:
                if seed == base_seed:
                    continue
                alt_params = {**combo, 'n_components': dims, 'seed': seed}
                alt_id = id_fn(**alt_params)
                alt_key = tuple(alt_params[k] for k in key_order) + (dims, seed)
                records[idx_map[alt_id]]['seed_stability_vs_base'] = knn_overlap_score(
                    emb_dict[base_key], emb_dict[alt_key], k=15)

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(metric_dir / f'{prefix.lower()}_sweep_summary.csv', index=False)
    return emb_dict, summary


def summarize_sweep(summary: pd.DataFrame, group_cols: list[str]) -> None:
    """Print mean metrics grouped by each parameter as one compact table."""
    metrics = ['trustworthiness', 'silhouette_breed', 'seed_stability_vs_base']
    lines = [f'{"":>16}  {"trust":>7}  {"silhou":>7}  {"stabil":>7}']
    for col in group_cols:
        table = summary.groupby(col)[metrics].mean().round(4)
        lines.append(col)
        for val, row in table.iterrows():
            lines.append(f'  {str(val):>14}  {row.iloc[0]:7.4f}  {row.iloc[1]:7.4f}  {row.iloc[2]:7.4f}')
    print('\n'.join(lines))


def run_umap_sweep(X_scaled, meta, *, neighbors, min_dists, metrics,
                   n_components, seeds, embed_dir, metric_dir):
    """Run the full UMAP hyperparameter sweep."""
    combos = [{'n_neighbors': nn, 'min_dist': md, 'metric': m}
              for nn in neighbors for md in min_dists for m in metrics]

    def fit(X_scaled, n_neighbors, min_dist, metric, n_components, seed):
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                         n_components=n_components, random_state=seed).fit_transform(X_scaled)

    def make_id(n_neighbors, min_dist, metric, n_components, seed):
        return _umap_run_id(metric, n_neighbors, min_dist, seed, n_components)

    grid = {
        'combos': combos,
        'seeds': seeds,
        'n_components': n_components,
        'key_order': ['n_neighbors', 'min_dist', 'metric'],
    }
    emb_dict, summary = _run_sweep(
        X_scaled, meta, param_grid=grid, fit_fn=fit, id_fn=make_id,
        prefix='UMAP', embed_dir=embed_dir, metric_dir=metric_dir)

    manifest = {'neighbors': neighbors, 'min_dists': min_dists, 'metrics': metrics,
                'n_components': n_components, 'seeds': seeds, 'num_runs': len(summary)}
    (metric_dir / 'umap_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    # Re-key dict to (dims, nn, md, metric, seed) for backwards compat
    rekeyed = {}
    for (nn, md, m, dims, seed), v in emb_dict.items():
        rekeyed[(dims, nn, md, m, seed)] = v

    print(f'UMAP sweep complete: {len(summary)} runs')
    return rekeyed, summary


def run_pacmap_sweep(X_scaled, meta, *, neighbors, mn_ratios, fp_ratios,
                     distances, n_components, seeds, embed_dir, metric_dir):
    """Run the full PaCMAP hyperparameter sweep."""
    combos = [{'n_neighbors': nn, 'MN_ratio': mn, 'FP_ratio': fp, 'distance': d}
              for nn in neighbors for mn in mn_ratios for fp in fp_ratios for d in distances]

    def fit(X_scaled, n_neighbors, MN_ratio, FP_ratio, distance, n_components, seed):
        return pacmap.PaCMAP(n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio,
                             distance=distance, n_components=n_components,
                             random_state=seed).fit_transform(X_scaled)

    def make_id(n_neighbors, MN_ratio, FP_ratio, distance, n_components, seed):
        return _pacmap_run_id(n_neighbors, MN_ratio, FP_ratio, distance, seed, n_components)

    grid = {
        'combos': combos,
        'seeds': seeds,
        'n_components': n_components,
        'key_order': ['n_neighbors', 'MN_ratio', 'FP_ratio', 'distance'],
    }
    emb_dict, summary = _run_sweep(
        X_scaled, meta, param_grid=grid, fit_fn=fit, id_fn=make_id,
        prefix='PaCMAP', embed_dir=embed_dir, metric_dir=metric_dir)

    manifest = {'neighbors': neighbors, 'MN_ratios': mn_ratios, 'FP_ratios': fp_ratios,
                'distances': distances, 'n_components': n_components, 'seeds': seeds,
                'num_runs': len(summary)}
    (metric_dir / 'pacmap_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    # Re-key dict to (dims, nn, mn, fp, distance, seed)
    rekeyed = {}
    for (nn, mn, fp, d, dims, seed), v in emb_dict.items():
        rekeyed[(dims, nn, mn, fp, d, seed)] = v

    print(f'PaCMAP sweep complete: {len(summary)} runs')
    return rekeyed, summary


def run_tsne_sweep(X_scaled, meta, *, perplexities, metrics,
                   n_components, seeds, embed_dir, metric_dir):
    """Run the full t-SNE hyperparameter sweep."""
    combos = [{'perplexity': p, 'metric': m}
              for p in perplexities for m in metrics]

    def fit(X_scaled, perplexity, metric, n_components, seed):
        return TSNE(perplexity=perplexity, metric=metric, n_components=n_components,
                    random_state=seed, init='pca').fit_transform(X_scaled)

    def make_id(perplexity, metric, n_components, seed):
        return _tsne_run_id(perplexity, metric, seed, n_components)

    grid = {
        'combos': combos,
        'seeds': seeds,
        'n_components': n_components,
        'key_order': ['perplexity', 'metric'],
    }
    emb_dict, summary = _run_sweep(
        X_scaled, meta, param_grid=grid, fit_fn=fit, id_fn=make_id,
        prefix='TSNE', embed_dir=embed_dir, metric_dir=metric_dir)

    manifest = {'perplexities': perplexities, 'metrics': metrics,
                'n_components': n_components, 'seeds': seeds, 'num_runs': len(summary)}
    (metric_dir / 'tsne_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    rekeyed = {}
    for (p, m, dims, seed), v in emb_dict.items():
        rekeyed[(dims, p, m, seed)] = v

    print(f't-SNE sweep complete: {len(summary)} runs')
    return rekeyed, summary


def run_trimap_sweep(X_scaled, meta, *, n_inliers_list, n_outliers_list, distances,
                     n_components, seeds, embed_dir, metric_dir,
                     weight_temps=(0.5,)):
    """Run the full TriMAP hyperparameter sweep."""
    combos = [{'n_inliers': ni, 'n_outliers': no, 'distance': d, 'weight_temp': wt}
              for ni in n_inliers_list for no in n_outliers_list
              for d in distances for wt in weight_temps]

    def fit(X_scaled, n_inliers, n_outliers, distance, weight_temp, n_components, seed):
        return trimap.TRIMAP(n_dims=n_components, n_inliers=n_inliers, n_outliers=n_outliers,
                             distance=distance, weight_temp=weight_temp,
                             verbose=False).fit_transform(X_scaled)

    def make_id(n_inliers, n_outliers, distance, weight_temp, n_components, seed):
        return _trimap_run_id(n_inliers, n_outliers, distance, seed, n_components,
                              weight_temp=weight_temp)

    grid = {
        'combos': combos,
        'seeds': seeds,
        'n_components': n_components,
        'key_order': ['n_inliers', 'n_outliers', 'distance', 'weight_temp'],
    }
    emb_dict, summary = _run_sweep(
        X_scaled, meta, param_grid=grid, fit_fn=fit, id_fn=make_id,
        prefix='TriMAP', embed_dir=embed_dir, metric_dir=metric_dir)

    manifest = {'n_inliers': n_inliers_list, 'n_outliers': n_outliers_list,
                'distances': distances, 'weight_temps': list(weight_temps),
                'n_components': n_components, 'seeds': seeds,
                'num_runs': len(summary)}
    (metric_dir / 'trimap_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    rekeyed = {}
    for (ni, no, d, wt, dims, seed), v in emb_dict.items():
        rekeyed[(dims, ni, no, d, wt, seed)] = v

    print(f'TriMAP sweep complete: {len(summary)} runs')
    return rekeyed, summary


def run_phate_sweep(X_scaled, meta, *, knn_values, decays, knn_dists,
                    n_components, seeds, embed_dir, metric_dir,
                    t_values=('auto',), gammas=(1,)):
    """Run the full PHATE hyperparameter sweep."""
    combos = [{'knn': k, 'decay': dc, 'knn_dist': d, 't': tv, 'gamma': g}
              for k in knn_values for dc in decays for d in knn_dists
              for tv in t_values for g in gammas]

    def fit(X_scaled, knn, decay, knn_dist, t, gamma, n_components, seed):
        return phate.PHATE(n_components=n_components, knn=knn, decay=decay,
                           knn_dist=knn_dist, t=t, gamma=gamma,
                           random_state=seed,
                           verbose=0, n_jobs=1).fit_transform(X_scaled)

    def make_id(knn, decay, knn_dist, t, gamma, n_components, seed):
        return _phate_run_id(knn, decay, knn_dist, seed, n_components, t=t, gamma=gamma)

    grid = {
        'combos': combos,
        'seeds': seeds,
        'n_components': n_components,
        'key_order': ['knn', 'decay', 'knn_dist', 't', 'gamma'],
    }
    emb_dict, summary = _run_sweep(
        X_scaled, meta, param_grid=grid, fit_fn=fit, id_fn=make_id,
        prefix='PHATE', embed_dir=embed_dir, metric_dir=metric_dir)

    manifest = {'knn_values': knn_values, 'decays': decays, 'knn_dists': knn_dists,
                't_values': list(t_values), 'gammas': list(gammas),
                'n_components': n_components, 'seeds': seeds, 'num_runs': len(summary)}
    (metric_dir / 'phate_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    rekeyed = {}
    for (k, dc, d, tv, g, dims, seed), v in emb_dict.items():
        rekeyed[(dims, k, dc, d, tv, g, seed)] = v

    print(f'PHATE sweep complete: {len(summary)} runs')
    return rekeyed, summary


# --- Phylo-Autoencoder ---------------------------------------------------

def compute_phylo_distance_matrices(X_scaled, meta, pca_components=20,
                                     target_metric='euclidean'):
    """Compute normalized breed-level and clade-level distance matrices from PCA centroids."""
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist, squareform

    scipy_metric = 'cityblock' if target_metric == 'manhattan' else target_metric

    X_pca = PCA(n_components=pca_components, random_state=ANALYSIS_SEED).fit_transform(X_scaled)

    breed_labels = meta['breed'].values
    breed_names, breed_cents = _group_centroids(X_pca, breed_labels)
    breed_D = squareform(pdist(breed_cents, metric=scipy_metric))
    breed_D /= breed_D.max() + 1e-12

    clade_labels = meta['breed_group'].values
    clade_names, clade_cents = _group_centroids(X_pca, clade_labels)
    clade_D = squareform(pdist(clade_cents, metric=scipy_metric))
    clade_D /= clade_D.max() + 1e-12

    breed_to_idx = {b: i for i, b in enumerate(breed_names)}
    clade_to_idx = {c: i for i, c in enumerate(clade_names)}
    sample_breed_idx = np.array([breed_to_idx[b] for b in breed_labels])
    sample_clade_idx = np.array([clade_to_idx[c] for c in clade_labels])

    return {
        'breed_dist': breed_D, 'clade_dist': clade_D,
        'n_breeds': len(breed_names), 'n_clades': len(clade_names),
        'sample_breed_idx': sample_breed_idx, 'sample_clade_idx': sample_clade_idx,
    }


def _phyloae_run_id(hidden_dims: tuple, lr: float, lambda_breed: float,
                    lambda_clade: float, seed: int, dims: int,
                    target_metric: str = 'euclidean') -> str:
    hd_s = 'x'.join(str(h) for h in hidden_dims)
    lr_s = str(lr).replace('.', 'p').replace('-', 'n')
    lb_s = str(lambda_breed).replace('.', 'p')
    lc_s = str(lambda_clade).replace('.', 'p')
    if target_metric == 'euclidean':
        return f'phyloae_h{hd_s}_lr{lr_s}_lb{lb_s}_lc{lc_s}_seed{seed}_{dims}d'
    return f'phyloae_{target_metric}_h{hd_s}_lr{lr_s}_lb{lb_s}_lc{lc_s}_seed{seed}_{dims}d'


def _phyloae_centroid_dist(centroids, metric, dev):
    """Compute normalized pairwise distance matrix between centroids using the given metric."""
    import torch
    if metric == 'cosine':
        normed = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-12)
        sim = normed @ normed.t()
        d = 1.0 - sim
    elif metric == 'manhattan':
        d = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0)
    else:
        d = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0), p=2).squeeze(0)
    return d / (d.max() + 1e-12)


def run_phyloae_sweep(X_scaled, meta, *, hidden_dims_list, lrs, lambda_breeds,
                      lambda_clades, n_components, seeds, embed_dir, metric_dir,
                      target_metrics=None, max_epochs=500):
    """Run the full Phylo-Autoencoder hyperparameter sweep.

    Runs serially in-process (no joblib) because PyTorch segfaults in forked workers.
    Torch/MPS initialization is deferred until at least one run needs training,
    so a fully cached sweep never allocates GPU memory.
    """
    if target_metrics is None:
        target_metrics = ['euclidean']
    meta_reset = meta.reset_index(drop=True)
    breed_col = meta['breed']
    key_order = ['hidden_dims', 'lr', 'lambda_breed', 'lambda_clade']
    prefix = 'PhyloAE'

    all_params = []
    for tm in target_metrics:
        for hd in hidden_dims_list:
            for lr in lrs:
                for lb in lambda_breeds:
                    for lc in lambda_clades:
                        for dims in n_components:
                            for seed in seeds:
                                all_params.append({
                                    'hidden_dims': hd, 'lr': lr, 'lambda_breed': lb,
                                    'lambda_clade': lc, 'n_components': dims, 'seed': seed,
                                    'target_metric': tm,
                                })

    cached_params, new_params = [], []
    for p in all_params:
        rid = _phyloae_run_id(p['hidden_dims'], p['lr'], p['lambda_breed'],
                              p['lambda_clade'], p['seed'], p['n_components'],
                              p['target_metric'])
        if (embed_dir / f'{rid}.csv').exists():
            cached_params.append(p)
        else:
            new_params.append(p)

    # Lazy torch/MPS init: only when training is actually needed
    fit_one = None
    phylo_by_metric = {}
    if new_params:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        needed_metrics = sorted({p['target_metric'] for p in new_params})
        for tm in needed_metrics:
            phylo_by_metric[tm] = compute_phylo_distance_matrices(
                X_scaled, meta, target_metric=tm)

        if torch.cuda.is_available():
            dev = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dev = torch.device('mps')
        else:
            dev = torch.device('cpu')
        print(f'  PhyloAE device: {dev}', flush=True)

        X_t = torch.FloatTensor(X_scaled).to(dev)

        phylo_tensors = {}
        for tm, phylo in phylo_by_metric.items():
            breed_idx_t = torch.LongTensor(phylo['sample_breed_idx']).to(dev)
            clade_idx_t = torch.LongTensor(phylo['sample_clade_idx']).to(dev)
            breed_dist_t = torch.FloatTensor(phylo['breed_dist']).to(dev)
            clade_dist_t = torch.FloatTensor(phylo['clade_dist']).to(dev)
            nb_, nc_ = phylo['n_breeds'], phylo['n_clades']
            mask_b = torch.triu(torch.ones(nb_, nb_, dtype=torch.bool, device=dev), diagonal=1)
            mask_c = torch.triu(torch.ones(nc_, nc_, dtype=torch.bool, device=dev), diagonal=1)
            phylo_tensors[tm] = {
                'breed_idx': breed_idx_t, 'clade_idx': clade_idx_t,
                'breed_dist': breed_dist_t, 'clade_dist': clade_dist_t,
                'nb': nb_, 'nc': nc_, 'mask_b': mask_b, 'mask_c': mask_c,
            }

        def scatter_mean(z, idx, n_groups):
            sums = torch.zeros(n_groups, z.shape[1], dtype=z.dtype, device=dev)
            counts = torch.zeros(n_groups, 1, dtype=z.dtype, device=dev)
            sums.scatter_add_(0, idx.unsqueeze(1).expand_as(z), z)
            counts.scatter_add_(0, idx.unsqueeze(1), torch.ones(z.shape[0], 1, device=dev))
            return sums / counts.clamp(min=1)

        def _fit_one(hidden_dims, lr, lambda_breed, lambda_clade, n_comp, seed,
                     target_metric='euclidean'):
            torch.manual_seed(seed)
            np.random.seed(seed)
            pt = phylo_tensors[target_metric]
            breed_idx_t = pt['breed_idx']
            clade_idx_t = pt['clade_idx']
            breed_dist_t = pt['breed_dist']
            clade_dist_t = pt['clade_dist']
            nb_, nc_ = pt['nb'], pt['nc']
            mask_b, mask_c = pt['mask_b'], pt['mask_c']

            h1, h2 = hidden_dims
            encoder = nn.Sequential(
                nn.Linear(X_scaled.shape[1], h1), nn.BatchNorm1d(h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(),
                nn.Linear(h2, n_comp),
            ).to(dev)
            decoder = nn.Sequential(
                nn.Linear(n_comp, h2), nn.BatchNorm1d(h2), nn.ReLU(),
                nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(),
                nn.Linear(h1, X_scaled.shape[1]),
            ).to(dev)
            params = list(encoder.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)
            encoder.train(); decoder.train()
            for _ in range(max_epochs):
                z = encoder(X_t)
                x_hat = decoder(z)
                loss_recon = F.mse_loss(x_hat, X_t)
                bc = scatter_mean(z, breed_idx_t, nb_)
                bd = _phyloae_centroid_dist(bc, target_metric, dev)
                loss_breed = ((bd[mask_b] - breed_dist_t[mask_b]) ** 2).sum() / \
                             ((breed_dist_t[mask_b] ** 2).sum() + 1e-12)
                cc = scatter_mean(z, clade_idx_t, nc_)
                cd = _phyloae_centroid_dist(cc, target_metric, dev)
                loss_clade = ((cd[mask_c] - clade_dist_t[mask_c]) ** 2).sum() / \
                             ((clade_dist_t[mask_c] ** 2).sum() + 1e-12)
                loss = loss_recon + lambda_breed * loss_breed + lambda_clade * loss_clade
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            encoder.eval()
            with torch.no_grad():
                return encoder(X_t).cpu().numpy()

        fit_one = _fit_one

    total = len(all_params)
    n_cached = len(cached_params)
    n_new = len(new_params)
    print(f'{prefix} sweep: {total} total ({n_cached} cached, {n_new} to compute) ...', flush=True)

    emb_dict = {}
    records = []

    for p in cached_params:
        rid = _phyloae_run_id(p['hidden_dims'], p['lr'], p['lambda_breed'],
                              p['lambda_clade'], p['seed'], p['n_components'],
                              p['target_metric'])
        dims = p['n_components']
        key = (p['target_metric'],) + tuple(p[k] for k in key_order) + (dims, p['seed'])
        emb = _load_embedding_from_csv(embed_dir / f'{rid}.csv', prefix, dims)
        trust = float(trustworthiness(X_scaled, emb, n_neighbors=min(15, X_scaled.shape[0] - 1)))
        sil = silhouette_by_label(emb, breed_col)
        emb_dict[key] = emb
        records.append({'run_id': rid, **p, 'trustworthiness': trust,
                        'silhouette_breed': sil, 'seed_stability_vs_base': np.nan,
                        'embedding_csv': str(embed_dir / f'{rid}.csv')})

    for i, p in enumerate(new_params):
        rid = _phyloae_run_id(p['hidden_dims'], p['lr'], p['lambda_breed'],
                              p['lambda_clade'], p['seed'], p['n_components'],
                              p['target_metric'])
        dims = p['n_components']
        key = (p['target_metric'],) + tuple(p[k] for k in key_order) + (dims, p['seed'])
        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{n_new}] {rid}', flush=True)
            try:
                import subprocess
                _prog = f'{i+1}/{n_new} computing | {n_cached} cached | device={dev}'
                subprocess.run(['gsutil', '-q', 'cp', '-', 'gs://cbmf4761-phyloae/progress.txt'],
                               input=_prog.encode(), timeout=5, capture_output=True)
            except Exception:
                pass
        emb = fit_one(p['hidden_dims'], p['lr'], p['lambda_breed'],
                      p['lambda_clade'], dims, p['seed'], p['target_metric'])
        emb_cols = [f'{prefix}{j+1}' for j in range(dims)]
        emb_df = pd.concat([meta_reset, pd.DataFrame(emb, columns=emb_cols)], axis=1)
        emb_path = embed_dir / f'{rid}.csv'
        tmp_path = emb_path.with_suffix('.csv.tmp')
        emb_df.to_csv(tmp_path, index=False)
        tmp_path.rename(emb_path)
        trust = float(trustworthiness(X_scaled, emb, n_neighbors=min(15, X_scaled.shape[0] - 1)))
        sil = silhouette_by_label(emb, breed_col)
        emb_dict[key] = emb
        records.append({'run_id': rid, **p, 'trustworthiness': trust,
                        'silhouette_breed': sil, 'seed_stability_vs_base': np.nan,
                        'embedding_csv': str(emb_path)})

    # Seed stability
    idx_map = {r['run_id']: i for i, r in enumerate(records)}
    base_seed = min(seeds)
    for tm in target_metrics:
        for hd in hidden_dims_list:
            for lr in lrs:
                for lb in lambda_breeds:
                    for lc in lambda_clades:
                        for dims in n_components:
                            base_id = _phyloae_run_id(hd, lr, lb, lc, base_seed, dims, tm)
                            base_key = (tm, hd, lr, lb, lc, dims, base_seed)
                            records[idx_map[base_id]]['seed_stability_vs_base'] = 1.0
                            for seed in seeds:
                                if seed == base_seed:
                                    continue
                                alt_id = _phyloae_run_id(hd, lr, lb, lc, seed, dims, tm)
                                alt_key = (tm, hd, lr, lb, lc, dims, seed)
                                records[idx_map[alt_id]]['seed_stability_vs_base'] = \
                                    knn_overlap_score(emb_dict[base_key], emb_dict[alt_key], k=15)

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(metric_dir / f'{prefix.lower()}_sweep_summary.csv', index=False)

    manifest = {'hidden_dims_list': [list(h) for h in hidden_dims_list],
                'lrs': list(lrs), 'lambda_breeds': list(lambda_breeds),
                'lambda_clades': list(lambda_clades), 'max_epochs': max_epochs,
                'n_components': n_components, 'seeds': seeds,
                'target_metrics': target_metrics, 'num_runs': len(summary)}
    (metric_dir / 'phyloae_sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    rekeyed = {}
    for (tm, hd, lr, lb, lc, dims, seed), v in emb_dict.items():
        rekeyed[(dims, tm, hd, lr, lb, lc, seed)] = v

    print(f'{prefix} sweep complete: {len(summary)} runs')
    try:
        import subprocess
        subprocess.run(['gsutil', '-q', 'cp',
                        str(metric_dir / f'{prefix.lower()}_sweep_summary.csv'),
                        'gs://cbmf4761-phyloae/'], timeout=30, capture_output=True)
        subprocess.run(['gsutil', '-q', 'cp',
                        str(metric_dir / 'phyloae_sweep_manifest.json'),
                        'gs://cbmf4761-phyloae/'], timeout=30, capture_output=True)
    except Exception:
        pass
    return rekeyed, summary


# ---------------------------------------------------------------------------
# 6) Plotting
# ---------------------------------------------------------------------------

def _legend_params(n_breeds):
    cols = 1 if n_breeds <= 20 else 2 if n_breeds <= 60 else 3 if n_breeds <= 120 else 4
    font = 6.2 if n_breeds > 60 else 7.0
    return cols, font


def plot_comparison_grid(panels, meta, breed_order, color_mpl, fig_dir, theme,
                         *, clade_order=None, clade_color_mpl=None):
    """N-panel comparison grid. panels: list of (emb_2d, title_str)."""
    from matplotlib.lines import Line2D
    t = theme
    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows),
                             dpi=180, facecolor=t['fig_bg'])
    axes = np.atleast_1d(axes).flatten()

    for ax, (emb, title) in zip(axes, panels):
        ax.set_facecolor(t['ax_bg'])
        for b in breed_order:
            mask = (meta['breed'] == b).values
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1], s=7, alpha=0.72,
                       color=color_mpl[b], linewidths=0)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(alpha=0.3, linewidth=0.5, color=t['grid'])
        ax.tick_params(labelsize=8, colors=t['text'])
        for spine in ax.spines.values():
            spine.set_color(t['grid'])

    for ax in axes[n:]:
        ax.set_visible(False)

    # Clade-level legend on the last visible axis
    use_clade = clade_order is not None and clade_color_mpl is not None
    legend_items = clade_order if use_clade else breed_order
    legend_colors = clade_color_mpl if use_clade else color_mpl
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=legend_colors[c], markersize=5, label=c)
               for c in legend_items]
    cols, font = _legend_params(len(legend_items))
    axes[n - 1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                       ncol=cols, frameon=False, fontsize=font)

    label = 'clade' if use_clade else 'breed'
    fig.suptitle(f'Each point = one dog; color = {label}', fontsize=10.5, y=0.995, color=t['text'])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = fig_dir / 'all_methods_2d_comparison.png'
    fig.savefig(path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    print('Comparison grid:', path)
    return path


def plot_embedding_2d(emb_2d, meta, breed_order, color_mpl, *,
                      method_name, method_params, axis_prefix, fig_dir, theme,
                      clade_order=None, clade_color_mpl=None):
    """Standalone 2D embedding figure."""
    from matplotlib.lines import Line2D
    t = theme

    fig, ax = plt.subplots(figsize=(10.6, 8.1), dpi=180, facecolor=t['fig_bg'])
    ax.set_facecolor(t['ax_bg'])
    for b in breed_order:
        mask = (meta['breed'] == b).values
        if not mask.any():
            continue
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.8,
                   color=color_mpl[b], linewidths=0)

    _skip = {'seed', 'run_id', 'n_components', 'embedding_csv',
             'trustworthiness', 'silhouette_breed', 'seed_stability_vs_base',
             'trustworthiness_rank', 'silhouette_breed_rank',
             'seed_stability_vs_base_rank', 'composite'}
    param_str = ', '.join(f'{k}={v}' for k, v in method_params.items() if k not in _skip)
    ax.set_title(f'{method_name} 2D ({param_str})', pad=12, fontsize=12)
    ax.set_xlabel(f'{axis_prefix}1', fontsize=10)
    ax.set_ylabel(f'{axis_prefix}2', fontsize=10)
    ax.grid(alpha=0.35, linewidth=0.6, color=t['grid'])
    ax.tick_params(labelsize=9, colors=t['text'])
    for spine in ax.spines.values():
        spine.set_color(t['grid'])

    # Clade-level legend when available, else breed-level
    use_clade = clade_order is not None and clade_color_mpl is not None
    legend_items = clade_order if use_clade else breed_order
    legend_colors = clade_color_mpl if use_clade else color_mpl
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=legend_colors[c], markersize=5, label=c)
               for c in legend_items]
    cols, font = _legend_params(len(legend_items))
    label = 'Clade' if use_clade else 'Breed'
    ax.legend(handles=handles, title=label, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=cols, frameon=False, fontsize=font, title_fontsize=8.8)
    fig.tight_layout(rect=(0, 0.13, 1, 1.0))

    fname = f'{method_name.lower()}_2d_breed.png'
    path = fig_dir / fname
    fig.savefig(path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    print(f'{method_name} 2D:', path)
    return path


def plot_3d_interactive(emb_3d, meta, breed_order, color_plotly, fig_dir, theme,
                        *, method_name='UMAP', axis_prefix='UMAP',
                        clade_order=None, clade_color_plotly=None):
    """Interactive 3D scatter via Plotly."""
    t = theme
    c1, c2, c3 = f'{axis_prefix}1', f'{axis_prefix}2', f'{axis_prefix}3'
    df = meta[['FID', 'IID', 'breed', 'breed_group']].copy()
    df[c1] = emb_3d[:, 0]
    df[c2] = emb_3d[:, 1]
    df[c3] = emb_3d[:, 2]

    use_clade = clade_order is not None and clade_color_plotly is not None
    color_col = 'breed_group' if use_clade else 'breed_group'
    cmap = clade_color_plotly if use_clade else color_plotly
    cat_order = clade_order if use_clade else breed_order
    legend_title = 'Clade' if use_clade else 'Breed'

    fig = px.scatter_3d(
        df, x=c1, y=c2, z=c3,
        color=color_col, color_discrete_map=cmap,
        category_orders={color_col: cat_order},
        hover_name='IID',
        hover_data={'FID': True, 'breed': True, 'breed_group': True,
                    c1: ':.3f', c2: ':.3f', c3: ':.3f'},
        title=f'{method_name} 3D: each point = one dog; color = {legend_title.lower()}',
    )
    fig.update_traces(marker=dict(size=4, opacity=0.88, line=dict(width=0)))
    fig.update_layout(
        template=t['plotly_template'], paper_bgcolor=t['fig_bg'], plot_bgcolor=t['ax_bg'],
        margin=dict(l=0, r=0, b=0, t=64),
        legend=dict(title=legend_title, orientation='v', x=1.02, y=1, xanchor='left', yanchor='top',
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=t['text']),
                    title_font=dict(size=11, color=t['text']), itemsizing='constant'),
        title=dict(x=0.01, xanchor='left', font=dict(size=16, color=t['text'])),
        hoverlabel=dict(bgcolor=t['ax_bg'], bordercolor=t['grid'], font=dict(color=t['text'], size=12)),
    )
    for axis_kw in ['xaxis', 'yaxis', 'zaxis']:
        fig.update_scenes(**{axis_kw: dict(gridcolor=t['grid'], backgroundcolor=t['ax_bg'], zerolinecolor=t['grid'])})

    slug = method_name.lower()
    path = fig_dir / f'{slug}_3d_interactive.html'
    fig.write_html(str(path), include_plotlyjs='cdn')
    print(f'{method_name} 3D interactive: {path}')
    return path


def plot_admixture_barplot(q_values, sort_emb_2d, best_k, component_colors, component_names,
                           concordances, fig_dir, theme):
    """Stacked Q barplot sorted by an embedding's first axis. concordances: list of dicts."""
    t = theme
    sort_idx = np.argsort(sort_emb_2d[:, 0])
    Q_sorted = q_values[sort_idx]

    fig, ax = plt.subplots(figsize=(13.2, 5.1), dpi=180, facecolor=t['fig_bg'])
    ax.set_facecolor(t['ax_bg'])
    x = np.arange(Q_sorted.shape[0])
    bottom = np.zeros(Q_sorted.shape[0])
    for ci in range(best_k):
        ax.bar(x, Q_sorted[:, ci], bottom=bottom, width=1.0,
               color=component_colors[ci], label=component_names[ci], linewidth=0)
        bottom += Q_sorted[:, ci]

    ax.set_title('ADMIXTURE Q profiles, samples ordered by best embedding axis 1', fontsize=12, pad=10)
    ax.set_xlabel('Samples (sorted)', fontsize=10)
    ax.set_ylabel('Ancestry proportion (Q)', fontsize=10)
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.55, color=t['grid'])
    ax.tick_params(labelsize=9, colors=t['text'])
    for spine in ax.spines.values():
        spine.set_color(t['grid'])

    lines = ['Concordance']
    for c in concordances:
        lines.append(f"{c['method']:>7s}: ARI={c['ari']:.3f}  NMI={c['nmi']:.3f}  enrich={c['neighbor_enrichment_ratio']:.1f}x")
    panel = '\n'.join(lines)
    ax.text(1.005, 0.99, panel, transform=ax.transAxes, va='top', ha='left',
            fontsize=7.5, color=t['text'], family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=t['ax_bg'], alpha=0.94, edgecolor=t['grid']))
    ax.legend(title='Components', ncol=min(best_k, 6), bbox_to_anchor=(0.5, -0.16),
              loc='upper center', frameon=False, fontsize=8, title_fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 0.82, 1.0))
    path = fig_dir / 'admixture_barplot_best_k.png'
    fig.savefig(path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    print('ADMIXTURE barplot:', path)
    return path


# ---------------------------------------------------------------------------
# 7) Concordance & ADMIXTURE
# ---------------------------------------------------------------------------

def _parse_cv_error(log_path: Path, k: int) -> float:
    """Parse CV error from an ADMIXTURE log file."""
    text = log_path.read_text(encoding='utf-8')
    match = re.search(rf'CV error \(K={k}\):\s+([0-9.]+)', text)
    if match:
        return float(match.group(1))
    raise ValueError(f'Could not parse CV error for K={k} from {log_path}')


def _extract_k_from_q_path(path: Path, run_prefix: str) -> int | None:
    m = re.match(rf'^{re.escape(run_prefix)}\.(\d+)\.Q$', path.name)
    return int(m.group(1)) if m else None


def _extract_k_from_log_path(path: Path) -> int | None:
    m = re.match(r'^log_k(\d+)\.out$', path.name)
    return int(m.group(1)) if m else None


def _nearest_available_k(candidates: list[int], target: float) -> int:
    if not candidates:
        raise ValueError('No candidate K values available.')
    return sorted(candidates, key=lambda k: (abs(k - target), k))[0]


def _component_palette(k: int):
    if k <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(k)]
    else:
        cmap = plt.get_cmap('turbo')
        colors = [cmap(i / max(k - 1, 1)) for i in range(k)]
    names = [f'Component {i + 1}' for i in range(k)]
    return colors, names


def assert_ld_pruned_input(pruned_prefix: Path) -> dict:
    """Validate that ADMIXTURE input prefix points to an LD-pruned PLINK set."""
    req = {ext: pruned_prefix.with_suffix(f'.{ext}') for ext in ['bed', 'bim', 'fam']}
    exists = {ext: p.exists() for ext, p in req.items()}
    token_ok = 'ldpruned' in pruned_prefix.name.lower()
    passed = token_ok and all(exists.values())
    return {
        'prefix': str(pruned_prefix),
        'contains_ldpruned_token': token_ok,
        'required_files': {ext: str(p) for ext, p in req.items()},
        'required_files_exist': exists,
        'assertion_passed': passed,
    }


def build_admixture_cv_table(admix_dir: Path, run_prefix: str) -> pd.DataFrame:
    """Build canonical ADMIXTURE CV table from existing logs/Q files."""
    q_files: dict[int, Path] = {}
    for p in admix_dir.glob(f'{run_prefix}.*.Q'):
        k = _extract_k_from_q_path(p, run_prefix)
        if k is not None:
            q_files[k] = p

    log_files: dict[int, Path] = {}
    for p in admix_dir.glob('log_k*.out'):
        k = _extract_k_from_log_path(p)
        if k is not None:
            log_files[k] = p

    all_k = sorted(set(q_files) | set(log_files))
    if not all_k:
        raise FileNotFoundError(f'No ADMIXTURE logs/Q files found in {admix_dir}')

    records = []
    for k in all_k:
        q_path = q_files.get(k, admix_dir / f'{run_prefix}.{k}.Q')
        p_path = admix_dir / f'{run_prefix}.{k}.P'
        log_path = log_files.get(k, admix_dir / f'log_k{k}.out')
        cv_error = np.nan
        if log_path.exists():
            try:
                cv_error = _parse_cv_error(log_path, k)
            except ValueError:
                cv_error = np.nan
        records.append({
            'K': int(k),
            'cv_error': cv_error,
            'log_file': str(log_path) if log_path.exists() else '',
            'q_file': str(q_path) if q_path.exists() else '',
            'p_file': str(p_path) if p_path.exists() else '',
            'log_exists': bool(log_path.exists()),
            'q_exists': bool(q_path.exists()),
            'p_exists': bool(p_path.exists()),
        })

    cv_df = pd.DataFrame.from_records(records).sort_values('K').reset_index(drop=True)
    cv_prev = cv_df['cv_error'].shift(1)
    cv_df['delta_cv'] = cv_prev - cv_df['cv_error']
    cv_df['pct_improvement'] = (cv_df['delta_cv'] / cv_prev) * 100.0
    cv_df['delta2_cv'] = cv_df['delta_cv'] - cv_df['delta_cv'].shift(1)

    cv_df.to_csv(admix_dir / 'cv_errors_all.tsv', sep='\t', index=False)

    # Compatibility table
    compat = cv_df[['K', 'cv_error', 'log_file']].copy()
    compat = compat[compat['cv_error'].notna()].reset_index(drop=True)
    compat.to_csv(admix_dir / 'cv_errors.tsv', sep='\t', index=False)
    return cv_df


def select_admixture_k_values(cv_df: pd.DataFrame, *,
                              elbow_threshold: float = ADMIX_ELBOW_DELTA_THRESHOLD) -> dict:
    """Deterministic K selection with gap-aware elbow logic.

    Definitions:
    - delta_cv(K) = cv_error(K-1) - cv_error(K) for sequential K.
      Larger delta_cv means larger fit improvement from adding one component.
    """
    valid = cv_df[cv_df['cv_error'].notna()].sort_values('K').reset_index(drop=True)
    if valid.empty:
        raise ValueError('No valid CV values found in ADMIXTURE table.')

    available_k = valid['K'].astype(int).tolist()
    k_to_cv = {int(r.K): float(r.cv_error) for r in valid.itertuples(index=False)}

    k_cv_min = int(valid.loc[valid['cv_error'].idxmin(), 'K'])

    seg = valid[(valid['K'] >= ADMIX_ELBOW_MIN_K) & (valid['K'] <= ADMIX_ELBOW_MAX_K)].copy()
    if seg.empty:
        raise ValueError('No K in elbow segment 2..15 with valid CV values.')
    seg = seg.sort_values('K').reset_index(drop=True)
    seg_delta = seg[seg['delta_cv'].notna()].copy()
    if seg_delta.empty:
        raise ValueError('No usable delta_cv values in elbow segment 2..15.')

    below = seg_delta[seg_delta['delta_cv'] < elbow_threshold]
    if not below.empty:
        k_elbow = int(below.iloc[0]['K'])
        elbow_rule = 'first_k_with_delta_cv_below_threshold'
    else:
        # Fallback 1: smallest K after which all remaining delta_cv values stay below threshold.
        k_elbow = None
        for k in seg_delta['K'].astype(int).tolist():
            tail = seg_delta.loc[seg_delta['K'] > k, 'delta_cv'].dropna()
            if tail.empty or (tail < elbow_threshold).all():
                k_elbow = int(k)
                elbow_rule = 'fallback_smallest_k_after_remaining_deltas_below_threshold'
                break
        # Fallback 2: if still none, maximize curvature via delta2.
        if k_elbow is None:
            seg_delta2 = seg_delta[seg_delta['delta2_cv'].notna()].copy()
            if not seg_delta2.empty:
                k_elbow = int(seg_delta2.sort_values(['delta2_cv', 'K'], ascending=[False, True]).iloc[0]['K'])
                elbow_rule = 'fallback_max_delta2_curvature_in_2_15'
            else:
                k_elbow = int(seg_delta.sort_values(['K']).iloc[-1]['K'])
                elbow_rule = 'fallback_last_k_in_segment'

    broad_pool = [k for k in available_k if 4 <= k <= 8]
    k_broad_default = _nearest_available_k(broad_pool or available_k, k_elbow)

    k_fine_target = min(max(2 * k_elbow, 10), 15)
    fine_pool = [k for k in available_k if 10 <= k <= 15]
    k_fine_default = _nearest_available_k(fine_pool or available_k, k_fine_target)

    continuous_expected = set(range(ADMIX_ELBOW_MIN_K, ADMIX_ELBOW_MAX_K + 1))
    continuous_missing = sorted(continuous_expected - set(available_k))
    auxiliary_high_k = [k for k in available_k if k >= 23]

    return {
        'available_k': available_k,
        'k_cv_min': k_cv_min,
        'k_cv_min_note': 'fit-optimal, potentially overparameterized',
        'k_elbow': int(k_elbow),
        'k_elbow_rule': elbow_rule,
        'k_elbow_threshold_delta_cv': float(elbow_threshold),
        'k_broad_default': int(k_broad_default),
        'k_fine_default': int(k_fine_default),
        'k_fine_target': int(k_fine_target),
        'elbow_segment_min_k': ADMIX_ELBOW_MIN_K,
        'elbow_segment_max_k': ADMIX_ELBOW_MAX_K,
        'continuous_2_15_missing_k': continuous_missing,
        'auxiliary_high_k': auxiliary_high_k,
        'excluded_gap_for_elbow': [ADMIX_GAP_MIN_K, ADMIX_GAP_MAX_K],
        'cv_at_k_cv_min': k_to_cv.get(k_cv_min),
        'cv_at_k_elbow': k_to_cv.get(int(k_elbow)),
        'cv_at_k_broad_default': k_to_cv.get(int(k_broad_default)),
        'cv_at_k_fine_default': k_to_cv.get(int(k_fine_default)),
    }


def write_k_selection_summary(admix_dir: Path, k_selection: dict, ld_status: dict) -> Path:
    """Write K-selection summary JSON (including LD-prune assertion)."""
    payload = {**k_selection, 'ld_pruned_input_assertion': ld_status}
    path = admix_dir / 'k_selection_summary.json'
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Compatibility output for older downstream code.
    (admix_dir / 'best_k.txt').write_text(str(int(k_selection['k_cv_min'])), encoding='utf-8')
    return path


def build_k_interpretability(admix_dir: Path, run_prefix: str, *, n_samples: int | None = None,
                             k_min: int = ADMIX_ELBOW_MIN_K,
                             k_max: int = ADMIX_ELBOW_MAX_K) -> pd.DataFrame:
    """Compute per-K interpretability diagnostics on continuous 2..15 segment."""
    ks = list(range(k_min, k_max + 1))
    labels: dict[int, np.ndarray] = {}
    rows = []

    for k in ks:
        q_path = admix_dir / f'{run_prefix}.{k}.Q'
        if not q_path.exists():
            continue
        q_df = pd.read_csv(q_path, sep=r'\s+', header=None)
        if n_samples is not None and q_df.shape[0] != n_samples:
            raise ValueError(f'Q rows ({q_df.shape[0]}) != samples ({n_samples}) for K={k}')
        q = q_df.to_numpy(dtype=float)
        labels[k] = q.argmax(axis=1)
        rows.append({
            'K': k,
            'mean_max_q': float(np.max(q, axis=1).mean()),
            'components_used': int(np.unique(labels[k]).size),
            'n_samples': int(q.shape[0]),
        })

    if not rows:
        raise FileNotFoundError(f'No Q files found in {admix_dir} for K={k_min}..{k_max}')

    out = pd.DataFrame(rows).sort_values('K').reset_index(drop=True)
    nmi_vals = []
    for k in out['K'].astype(int):
        k_next = k + 1
        if k_next in labels:
            nmi_vals.append(float(normalized_mutual_info_score(labels[k], labels[k_next])))
        else:
            nmi_vals.append(np.nan)
    out['adjacent_argmax_nmi_k_to_kplus1'] = nmi_vals

    dominance_floor = 0.55
    nmi_floor = 0.70
    out['likely_overfragmented'] = (
        (out['mean_max_q'] < dominance_floor) |
        (out['adjacent_argmax_nmi_k_to_kplus1'].notna() &
         (out['adjacent_argmax_nmi_k_to_kplus1'] < nmi_floor))
    )
    out['dominance_floor'] = dominance_floor
    out['adjacent_nmi_floor'] = nmi_floor
    out.to_csv(admix_dir / 'k_interpretability.csv', index=False)
    return out


def load_q_matrix(admix_dir: Path, run_prefix: str, best_k: int, n_samples: int):
    """Load Q matrix and derive labels and component colors."""
    q_path = admix_dir / f'{run_prefix}.{best_k}.Q'
    if not q_path.exists():
        raise FileNotFoundError(f'Missing Q file: {q_path}')
    Q = pd.read_csv(q_path, sep=r'\s+', header=None)
    if Q.shape[0] != n_samples:
        raise ValueError(f'Q rows ({Q.shape[0]}) != samples ({n_samples})')
    q_values = Q.to_numpy(dtype=float)
    admix_label = q_values.argmax(axis=1)
    colors, names = _component_palette(best_k)
    return q_values, admix_label, colors, names


def load_q_matrices(admix_dir: Path, run_prefix: str, k_values: list[int], n_samples: int) -> dict[int, np.ndarray]:
    """Load all requested Q matrices from cache."""
    q_by_k: dict[int, np.ndarray] = {}
    for k in sorted(set(int(v) for v in k_values)):
        q_path = admix_dir / f'{run_prefix}.{k}.Q'
        if not q_path.exists():
            raise FileNotFoundError(f'Missing Q file for K={k}: {q_path}')
        q = pd.read_csv(q_path, sep=r'\s+', header=None).to_numpy(dtype=float)
        if q.shape[0] != n_samples:
            raise ValueError(f'Q rows ({q.shape[0]}) != samples ({n_samples}) for K={k}')
        q_by_k[k] = q
    return q_by_k


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        return float('nan')
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < EPSILON or sb < EPSILON:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


def _shade_hex(hex_color: str, step: int) -> str:
    """Create deterministic shade variants for split children."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = min(0.90, max(0.18, l + 0.10 * step))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
    return mcolors.to_hex((r2, g2, b2), keep_alpha=False)


def _global_palette(n: int = 512) -> list[str]:
    vals = []
    for i in range(n):
        hue = (0.61803398875 * i) % 1.0  # golden-ratio spacing
        r, g, b = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
        vals.append(mcolors.to_hex((r, g, b), keep_alpha=False))
    return vals


def align_components_across_k(q_by_k: dict[int, np.ndarray], k_values: list[int]) -> dict:
    """Align ADMIXTURE components across K with Hungarian matching on 1-corr."""
    ks = sorted(set(int(k) for k in k_values if int(k) in q_by_k))
    if not ks:
        raise ValueError('No Q matrices available to align.')

    palette = _global_palette()
    palette_i = 0

    colors_by_k: dict[int, list[str]] = {}
    names_by_k: dict[int, list[str]] = {}
    parent_k_by_k: dict[int, int | None] = {}
    parent_map_by_k: dict[int, dict[int, int | None]] = {}
    corr_by_k: dict[int, dict[int, float]] = {}

    for idx, k in enumerate(ks):
        qk = q_by_k[k]
        n_comp = qk.shape[1]
        parent_map_by_k[k] = {}
        corr_by_k[k] = {}
        parent_k = (k - 1) if (k - 1) in q_by_k else None
        parent_k_by_k[k] = parent_k

        if idx == 0 or parent_k is None:
            cols = []
            for _ in range(n_comp):
                cols.append(palette[palette_i % len(palette)])
                palette_i += 1
            colors_by_k[k] = cols
            names_by_k[k] = [f'K{k}_C{c + 1}' for c in range(n_comp)]
            for c in range(n_comp):
                parent_map_by_k[k][c] = None
                corr_by_k[k][c] = float('nan')
            continue

        q_prev = q_by_k[parent_k]
        n_prev = q_prev.shape[1]
        corr_m = np.zeros((n_comp, n_prev), dtype=float)
        for ci in range(n_comp):
            for pj in range(n_prev):
                corr_m[ci, pj] = _safe_corr(qk[:, ci], q_prev[:, pj])
        cost = 1.0 - corr_m
        rows, cols = linear_sum_assignment(cost)
        matched = {int(r): int(c) for r, c in zip(rows, cols)}

        # group children by parent for split shading
        children_by_parent: dict[int, list[int]] = {}
        for child, par in matched.items():
            children_by_parent.setdefault(par, []).append(child)
        for par, children in children_by_parent.items():
            children.sort(key=lambda c: (-corr_m[c, par], c))

        colors = [None] * n_comp
        names = [f'K{k}_C{c + 1}' for c in range(n_comp)]
        prev_colors = colors_by_k[parent_k]
        for par, children in sorted(children_by_parent.items()):
            base = prev_colors[par]
            for s_idx, child in enumerate(children):
                colors[child] = base if s_idx == 0 else _shade_hex(base, s_idx)
                parent_map_by_k[k][child] = par
                corr_by_k[k][child] = float(corr_m[child, par])
                names[child] = f'K{k}_C{child + 1}<=K{parent_k}_C{par + 1}'

        # unmatched children are truly new components
        for child in range(n_comp):
            if colors[child] is None:
                colors[child] = palette[palette_i % len(palette)]
                palette_i += 1
                parent_map_by_k[k][child] = None
                corr_by_k[k][child] = float('nan')
                names[child] = f'K{k}_C{child + 1}(new)'

        colors_by_k[k] = [str(c) for c in colors]
        names_by_k[k] = names

    return {
        'k_values': ks,
        'colors_by_k': colors_by_k,
        'names_by_k': names_by_k,
        'parent_k_by_k': parent_k_by_k,
        'parent_map_by_k': parent_map_by_k,
        'corr_by_k': corr_by_k,
    }


def choose_best_concordant_embedding(emb_2d: dict[str, np.ndarray], q_broad: np.ndarray, *,
                                     seed: int = ANALYSIS_SEED) -> tuple[str, pd.DataFrame]:
    """Choose best embedding method by NMI against broad-K dominant labels."""
    y = q_broad.argmax(axis=1)
    rows = []
    for method, emb in emb_2d.items():
        km = KMeans(n_clusters=q_broad.shape[1], random_state=seed, n_init=10)
        pred = km.fit_predict(emb)
        rows.append({
            'method': method,
            'ari': float(adjusted_rand_score(y, pred)),
            'nmi': float(normalized_mutual_info_score(y, pred)),
        })
    df = pd.DataFrame(rows).sort_values(['nmi', 'ari', 'method'], ascending=[False, False, True]).reset_index(drop=True)
    return str(df.iloc[0]['method']), df


def build_master_sample_order(meta: pd.DataFrame, q_broad: np.ndarray, best_embedding: np.ndarray,
                              best_method: str, out_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Build one canonical sample order reused for every K barplot."""
    q_max = q_broad.max(axis=1)
    dominant = q_broad.argmax(axis=1).astype(int)
    df = pd.DataFrame({
        'sample_idx': np.arange(len(meta), dtype=int),
        'IID': meta['IID'].astype(str).values,
        'FID': meta['FID'].astype(str).values,
        'dominant_component_broad': dominant,
        'q_max_broad': q_max,
        'best_embedding_axis1': best_embedding[:, 0],
        'best_embedding_method': best_method,
    })
    df = df.sort_values(
        ['dominant_component_broad', 'q_max_broad', 'best_embedding_axis1', 'IID'],
        ascending=[True, False, True, True],
        kind='mergesort',
    ).reset_index(drop=True)
    df['sample_order_rank'] = np.arange(1, len(df) + 1, dtype=int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    order_idx = df['sample_idx'].to_numpy(dtype=int)
    return df, order_idx


def _entropy_norm(q: np.ndarray) -> np.ndarray:
    k = q.shape[1]
    ent = -(q * np.log(q + EPSILON)).sum(axis=1)
    return ent / max(np.log(max(k, 2)), EPSILON)


def _k_eff(q: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum((q ** 2).sum(axis=1), EPSILON)


def compute_admixture_metrics_by_k(q_by_k: dict[int, np.ndarray], cv_df: pd.DataFrame, k_selection: dict,
                                   alignment: dict, out_path: Path) -> pd.DataFrame:
    """Compute per-K and per-component ADMIXTURE metrics."""
    rows = []
    cv_map = {
        int(r.K): float(r.cv_error)
        for r in cv_df[cv_df['cv_error'].notna()].itertuples(index=False)
    }
    ks = sorted(q_by_k.keys())

    for k in ks:
        q = q_by_k[k]
        dom = q.argmax(axis=1)
        q_max = q.max(axis=1)
        h_norm = _entropy_norm(q)
        k_eff = _k_eff(q)

        next_k = k + 1 if (k + 1) in q_by_k else None
        mean_parent_corr = np.nan
        dom_nmi_next = np.nan
        dom_preserved_frac = np.nan
        if next_k is not None:
            pmap = alignment['parent_map_by_k'].get(next_k, {})
            corr_map = alignment['corr_by_k'].get(next_k, {})
            corr_vals = [v for v in corr_map.values() if not np.isnan(v)]
            if corr_vals:
                mean_parent_corr = float(np.mean(corr_vals))
            dom_next = q_by_k[next_k].argmax(axis=1)
            mapped = np.array([
                (-1 if pmap.get(int(c), -1) is None else int(pmap.get(int(c), -1)))
                for c in dom_next
            ], dtype=int)
            keep = mapped >= 0
            if keep.any():
                dom_nmi_next = float(normalized_mutual_info_score(dom[keep], mapped[keep]))
                dom_preserved_frac = float(np.mean(dom[keep] == mapped[keep]))

        rows.append({
            'row_type': 'k_summary',
            'K': int(k),
            'component': '',
            'cv_error': cv_map.get(int(k), np.nan),
            'delta_cv': float(cv_df.loc[cv_df['K'] == k, 'delta_cv'].iloc[0]) if (cv_df['K'] == k).any() else np.nan,
            'pct_improvement': float(cv_df.loc[cv_df['K'] == k, 'pct_improvement'].iloc[0]) if (cv_df['K'] == k).any() else np.nan,
            'delta2_cv': float(cv_df.loc[cv_df['K'] == k, 'delta2_cv'].iloc[0]) if (cv_df['K'] == k).any() else np.nan,
            'mean_q_max': float(np.mean(q_max)),
            'median_q_max': float(np.median(q_max)),
            'qmax_p10': float(np.percentile(q_max, 10)),
            'qmax_p25': float(np.percentile(q_max, 25)),
            'qmax_p75': float(np.percentile(q_max, 75)),
            'qmax_p90': float(np.percentile(q_max, 90)),
            'mean_h_norm': float(np.mean(h_norm)),
            'median_h_norm': float(np.median(h_norm)),
            'mean_k_eff': float(np.mean(k_eff)),
            'median_k_eff': float(np.median(k_eff)),
            'adjacent_mean_parent_corr': mean_parent_corr,
            'adjacent_dominant_nmi': dom_nmi_next,
            'adjacent_dominant_preserved_frac': dom_preserved_frac,
            'k_cv_min': int(k_selection['k_cv_min']),
            'k_elbow': int(k_selection['k_elbow']),
            'k_broad_default': int(k_selection['k_broad_default']),
            'k_fine_default': int(k_selection['k_fine_default']),
        })

        comp_mass = q.mean(axis=0)
        comp_q50 = (q >= 0.5).sum(axis=0)
        comp_q20 = (q >= 0.2).sum(axis=0)
        comp_order = np.argsort(-comp_mass)
        rank_map = {int(c): int(i + 1) for i, c in enumerate(comp_order)}
        for c in range(q.shape[1]):
            rows.append({
                'row_type': 'component',
                'K': int(k),
                'component': int(c + 1),
                'component_mass_mean_q': float(comp_mass[c]),
                'component_n_q_ge_0_5': int(comp_q50[c]),
                'component_n_q_ge_0_2': int(comp_q20[c]),
                'component_rank_within_k': rank_map[c],
                'component_embedding_separability': np.nan,
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    return df


def rank_k_candidates(admix_df: pd.DataFrame, embedding_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Apply deterministic K ranking over k_summary rows."""
    k_rows = admix_df[admix_df['row_type'] == 'k_summary'].copy()
    emb_k = embedding_df.groupby('K', as_index=False)['nmi'].max().rename(columns={'nmi': 'embedding_nmi_max'})
    k_rows = k_rows.merge(emb_k, on='K', how='left')
    k_rows['embedding_nmi_max'] = k_rows['embedding_nmi_max'].fillna(-1.0)
    k_rows['is_broad_or_fine_default'] = (
        (k_rows['K'] == k_rows['k_broad_default']) | (k_rows['K'] == k_rows['k_fine_default'])
    ).astype(int)
    k_rows['adjacent_stability_key'] = k_rows['adjacent_dominant_preserved_frac'].fillna(-1.0)
    k_rows = k_rows.sort_values(
        ['is_broad_or_fine_default', 'mean_h_norm', 'adjacent_stability_key', 'embedding_nmi_max', 'cv_error', 'K'],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)
    k_rows['k_rank'] = np.arange(1, len(k_rows) + 1, dtype=int)

    keep_cols = ['K', 'k_rank', 'is_broad_or_fine_default', 'embedding_nmi_max']
    merged = admix_df.merge(k_rows[keep_cols], on='K', how='left')
    merged.to_csv(out_path, sep='\t', index=False)
    return merged


def apply_component_ranking(admix_df: pd.DataFrame, q_by_k: dict[int, np.ndarray],
                            embedding_for_sep: np.ndarray, out_path: Path,
                            *, seed: int = ANALYSIS_SEED) -> pd.DataFrame:
    """Finalize component ranking with separability (mass, occupancy, separability)."""
    df = admix_df.copy()
    comp_mask = df['row_type'] == 'component'
    if not comp_mask.any():
        df.to_csv(out_path, sep='\t', index=False)
        return df

    for k in sorted(set(df.loc[comp_mask, 'K'].astype(int).tolist())):
        q = q_by_k.get(int(k))
        if q is None:
            continue
        dom = q.argmax(axis=1).astype(int)
        sub_idx = df.index[(df['row_type'] == 'component') & (df['K'] == k)].tolist()
        if not sub_idx:
            continue
        for idx in sub_idx:
            c = int(df.at[idx, 'component']) - 1
            y = (dom == c).astype(int)
            sep = float('nan')
            if y.sum() >= 20 and (len(y) - y.sum()) >= 20:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                clf = LogisticRegression(max_iter=800, class_weight='balanced', solver='lbfgs')
                sep = float(np.mean(cross_val_score(
                    clf, embedding_for_sep, y, scoring='balanced_accuracy', cv=cv, n_jobs=1
                )))
            df.at[idx, 'component_embedding_separability'] = sep

        rank_block = df.loc[sub_idx, [
            'component_mass_mean_q', 'component_n_q_ge_0_5', 'component_embedding_separability'
        ]].copy()
        rank_block['component_embedding_separability'] = rank_block['component_embedding_separability'].fillna(-1.0)
        order = rank_block.sort_values(
            ['component_mass_mean_q', 'component_n_q_ge_0_5', 'component_embedding_separability'],
            ascending=[False, False, False],
        ).index.tolist()
        for r, idx in enumerate(order, start=1):
            df.at[idx, 'component_rank_within_k'] = int(r)

    df.to_csv(out_path, sep='\t', index=False)
    return df


def plot_cv_diagnostics_standard(cv_df: pd.DataFrame, k_selection: dict, fig_path: Path, theme: dict) -> Path:
    """CV diagnostics figure with required naming."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = theme
    valid = cv_df[cv_df['cv_error'].notna()].sort_values('K').reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180, facecolor=t['fig_bg'])
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(t['ax_bg'])
        ax.grid(alpha=0.28, linewidth=0.55, color=t['grid'])
        ax.tick_params(labelsize=9, colors=t['text'])
        for spine in ax.spines.values():
            spine.set_color(t['grid'])
    ax1.plot(valid['K'], valid['cv_error'], marker='o', linewidth=1.7, color='#60a5fa', markersize=4)
    ax1.set_title('CV error by K', fontsize=11)
    ax1.set_xlabel('K')
    ax1.set_ylabel('CV error')
    for key, color in [('k_cv_min', '#f97316'), ('k_elbow', '#22c55e'),
                       ('k_broad_default', '#a78bfa'), ('k_fine_default', '#eab308')]:
        k = int(k_selection[key])
        row = valid[valid['K'] == k]
        if row.empty:
            continue
        y = float(row.iloc[0]['cv_error'])
        ax1.scatter([k], [y], s=62, color=color, zorder=4)
        ax1.text(k, y, f' {key}', color=t['text'], fontsize=7, va='bottom')
    ax2.bar(valid['K'], valid['delta_cv'].fillna(0.0), color='#93c5fd')
    ax2.axhline(y=float(k_selection.get('k_elbow_threshold_delta_cv', ADMIX_ELBOW_DELTA_THRESHOLD)),
                color='#f43f5e', linestyle='--', linewidth=1.0)
    ax2.set_title('delta_cv by K', fontsize=11)
    ax2.set_xlabel('K')
    ax2.set_ylabel('delta_cv = CV(K-1) - CV(K)')
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


def plot_admixture_barplots_master_order(q_by_k: dict[int, np.ndarray], sample_order_idx: np.ndarray, alignment: dict,
                                         fig_dir: Path, theme: dict, *, broad_k: int, fine_k: int) -> dict:
    """Produce one standardized barplot per K and broad/fine panel."""
    t = theme
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {'per_k': [], 'panel': None}
    ks = sorted(q_by_k.keys())

    for k in ks:
        q_sorted = q_by_k[k][sample_order_idx]
        colors = alignment['colors_by_k'][k]
        fig, ax = plt.subplots(figsize=(13.4, 4.9), dpi=180, facecolor=t['fig_bg'])
        ax.set_facecolor(t['ax_bg'])
        x = np.arange(q_sorted.shape[0])
        bottom = np.zeros(q_sorted.shape[0])
        for c in range(q_sorted.shape[1]):
            ax.bar(x, q_sorted[:, c], bottom=bottom, width=1.0, color=colors[c], linewidth=0)
            bottom += q_sorted[:, c]
        ax.set_title(f'Estimated ancestry proportions under model K={k}', fontsize=11)
        ax.set_xlim(0, len(x))
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Samples (fixed master order)')
        ax.set_ylabel('Q')
        ax.set_xticks([])
        ax.grid(alpha=0.22, linewidth=0.5, color=t['grid'])
        for spine in ax.spines.values():
            spine.set_color(t['grid'])
        fp = fig_dir / f'barplot_K{k}.png'
        fig.tight_layout()
        fig.savefig(fp, bbox_inches='tight', facecolor=t['fig_bg'])
        plt.close(fig)
        out['per_k'].append(fp)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.9), dpi=180, facecolor=t['fig_bg'])
    for ax, k in zip(axes, [broad_k, fine_k]):
        q_sorted = q_by_k[k][sample_order_idx]
        colors = alignment['colors_by_k'][k]
        ax.set_facecolor(t['ax_bg'])
        x = np.arange(q_sorted.shape[0])
        bottom = np.zeros(q_sorted.shape[0])
        for c in range(q_sorted.shape[1]):
            ax.bar(x, q_sorted[:, c], bottom=bottom, width=1.0, color=colors[c], linewidth=0)
            bottom += q_sorted[:, c]
        ax.set_title(f'K={k}', fontsize=11)
        ax.set_xlim(0, len(x))
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.grid(alpha=0.22, linewidth=0.5, color=t['grid'])
        for spine in ax.spines.values():
            spine.set_color(t['grid'])
    fig.suptitle('Broad vs Fine K (fixed sample order, inherited component colors)', fontsize=11, color=t['text'])
    fig.tight_layout()
    panel_fp = fig_dir / 'barplot_panel_broad_fine.png'
    fig.savefig(panel_fp, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    out['panel'] = panel_fp
    return out


def compute_concordance(emb_2d, q_values, admix_label, best_k, n_samples, method_name, seed=7):
    """Compute ARI, NMI, and neighborhood enrichment for one embedding vs ADMIXTURE."""
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    clusters = km.fit_predict(emb_2d)
    ari = adjusted_rand_score(admix_label, clusters)
    nmi = normalized_mutual_info_score(admix_label, clusters)

    k_nn = min(15, n_samples - 1)
    nn_idx = NearestNeighbors(n_neighbors=k_nn + 1).fit(emb_2d).kneighbors(return_distance=False)[:, 1:]
    neighbor_mean = float(np.abs(q_values[nn_idx] - q_values[:, None, :]).sum(axis=2).mean())

    rng = np.random.default_rng(seed)
    n_pairs = int(nn_idx.size)
    rand_dist = np.abs(q_values[rng.integers(0, n_samples, n_pairs)] -
                       q_values[rng.integers(0, n_samples, n_pairs)]).sum(axis=1)
    random_mean = float(rand_dist.mean())
    ratio = random_mean / neighbor_mean if neighbor_mean > 0 else float('nan')

    print(f'{method_name}: ARI={ari:.4f}  NMI={nmi:.4f}  enrichment={ratio:.2f}x')
    return {'method': method_name, 'best_k': best_k, 'ari': ari, 'nmi': nmi,
            'neighbor_mean_l1_q': neighbor_mean, 'random_mean_l1_q': random_mean,
            'neighbor_enrichment_ratio': ratio}


def compute_concordance_all_k(emb_2d: dict[str, np.ndarray], q_by_k: dict[int, np.ndarray],
                              seed: int = ANALYSIS_SEED) -> pd.DataFrame:
    """Compute ARI and NMI for every method at every available K."""
    rows = []
    for k in sorted(q_by_k.keys()):
        q = q_by_k[k]
        y = q.argmax(axis=1)
        for method, emb in emb_2d.items():
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            pred = km.fit_predict(emb)
            rows.append({
                'K': k, 'method': method,
                'ari': float(adjusted_rand_score(y, pred)),
                'nmi': float(normalized_mutual_info_score(y, pred)),
            })
    return pd.DataFrame(rows)


def plot_concordance_vs_k(conc_df: pd.DataFrame, fig_path: Path, theme: dict) -> Path:
    """Plot ARI and NMI vs K for each embedding method."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = theme
    methods = sorted(conc_df['method'].unique())
    method_colors = {m: c for m, c in zip(methods, [
        '#f97316', '#22c55e', '#60a5fa', '#a78bfa', '#eab308', '#f43f5e',
        '#06b6d4', '#d946ef', '#84cc16', '#fb7185',
    ])}
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=180, facecolor=t['fig_bg'])
    for ax in axes:
        ax.set_facecolor(t['ax_bg'])
        ax.grid(alpha=0.28, linewidth=0.55, color=t['grid'])
        ax.tick_params(labelsize=9, colors=t['text'])
        for spine in ax.spines.values():
            spine.set_color(t['grid'])
    for metric, ax, title in [('ari', axes[0], 'ARI vs K'), ('nmi', axes[1], 'NMI vs K')]:
        for method in methods:
            sub = conc_df[conc_df['method'] == method].sort_values('K')
            ax.plot(sub['K'], sub[metric], marker='o', linewidth=1.7, markersize=4,
                    color=method_colors[method], label=method)
        ax.set_title(title, fontsize=11, color=t['text'])
        ax.set_xlabel('K', color=t['text'])
        ax.set_ylabel(metric.upper(), color=t['text'])
        ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


def estimate_cluster_count(emb_2d: dict[str, np.ndarray], k_range: range = range(2, 51),
                           seed: int = ANALYSIS_SEED) -> pd.DataFrame:
    """Sweep KMeans K on each embedding, return silhouette scores."""
    from sklearn.metrics import silhouette_score
    rows = []
    for method, emb in emb_2d.items():
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(emb)
            sil = float(silhouette_score(emb, labels))
            rows.append({'method': method, 'K': k, 'silhouette': sil})
        print(f'  {method}: done (K={k_range.start}-{k_range.stop - 1})')
    return pd.DataFrame(rows)


def plot_cluster_estimation(cluster_df: pd.DataFrame, fig_path: Path, theme: dict,
                            *, parker_k: int = 26,
                            parker_label: str | None = None) -> Path:
    """Plot silhouette vs K for each method with Parker reference count marked."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    t = theme
    methods = sorted(cluster_df['method'].unique())
    method_colors = {m: c for m, c in zip(methods, [
        '#f97316', '#22c55e', '#60a5fa', '#a78bfa', '#eab308', '#f43f5e',
        '#06b6d4', '#d946ef', '#84cc16', '#fb7185',
    ])}
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=180, facecolor=t['fig_bg'])
    ax.set_facecolor(t['ax_bg'])
    ax.grid(alpha=0.28, linewidth=0.55, color=t['grid'])
    ax.tick_params(labelsize=9, colors=t['text'])
    for spine in ax.spines.values():
        spine.set_color(t['grid'])
    for method in methods:
        sub = cluster_df[cluster_df['method'] == method].sort_values('K')
        ax.plot(sub['K'], sub['silhouette'], marker='o', linewidth=1.7, markersize=3,
                color=method_colors[method], label=method)
        peak_k = int(sub.loc[sub['silhouette'].idxmax(), 'K'])
        peak_sil = sub['silhouette'].max()
        ax.scatter([peak_k], [peak_sil], s=80, color=method_colors[method],
                   edgecolors='white', linewidths=1.2, zorder=5)
    ax.axvline(x=parker_k, color='#f43f5e', linestyle='--', linewidth=1.2, alpha=0.7)
    label = parker_label or f'Parker ref: 23 clades + singletons + wolf (K={parker_k})'
    ax.text(parker_k + 0.5, ax.get_ylim()[1] * 0.95, label,
            color='#f43f5e', fontsize=9, va='top')
    ax.set_title('Estimated cluster count (silhouette vs K)', fontsize=11, color=t['text'])
    ax.set_xlabel('K (number of clusters)', color=t['text'])
    ax.set_ylabel('Silhouette score', color=t['text'])
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


def _method_slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', name.lower())


def _local_purity(embedding: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    labels = np.asarray(labels)
    k_eff = min(k, embedding.shape[0] - 1)
    if k_eff <= 0:
        return float('nan')
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(embedding)
    _, idx = nn.kneighbors(embedding)
    return float((labels[idx[:, 1:]] == labels[:, None]).mean())


def _logistic_separability(embedding: np.ndarray, labels: np.ndarray, *, seed: int = ANALYSIS_SEED) -> float:
    labels = np.asarray(labels, dtype=int)
    out = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for c in np.unique(labels):
        y = (labels == c).astype(int)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            continue
        clf = LogisticRegression(max_iter=800, class_weight='balanced', solver='lbfgs')
        scores = cross_val_score(clf, embedding, y, scoring='balanced_accuracy', cv=cv, n_jobs=1)
        out.append(float(np.mean(scores)))
    if not out:
        return float('nan')
    return float(np.mean(out))


def _plot_embedding_by_labels(embedding: np.ndarray, labels: np.ndarray, color_map: dict[int, str],
                              fig_path: Path, title: str, theme: dict):
    t = theme
    fig, ax = plt.subplots(figsize=(8.8, 7.4), dpi=180, facecolor=t['fig_bg'])
    ax.set_facecolor(t['ax_bg'])
    for c in sorted(np.unique(labels).tolist()):
        m = labels == c
        ax.scatter(embedding[m, 0], embedding[m, 1], s=10, alpha=0.82,
                   color=color_map.get(int(c), '#9ca3af'), linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(alpha=0.25, linewidth=0.55, color=t['grid'])
    for sp in ax.spines.values():
        sp.set_color(t['grid'])
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


def _plot_embedding_by_continuous(embedding: np.ndarray, values: np.ndarray,
                                  fig_path: Path, title: str, theme: dict, cmap: str = 'viridis'):
    t = theme
    fig, ax = plt.subplots(figsize=(8.8, 7.4), dpi=180, facecolor=t['fig_bg'])
    ax.set_facecolor(t['ax_bg'])
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=values, s=10, alpha=0.82,
                    cmap=cmap, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.outline.set_edgecolor(t['grid'])
    cbar.ax.tick_params(colors=t['text'], labelsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(alpha=0.25, linewidth=0.55, color=t['grid'])
    for sp in ax.spines.values():
        sp.set_color(t['grid'])
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


def compute_embedding_metrics_and_figures(emb_2d: dict[str, np.ndarray], meta: pd.DataFrame,
                                          q_by_k: dict[int, np.ndarray], target_ks: list[int],
                                          alignment: dict, out_path: Path, fig_dir: Path,
                                          theme: dict, *, seed: int = ANALYSIS_SEED) -> pd.DataFrame:
    """Compute embedding concordance/continuity metrics and export required figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    label_meta = meta['FID'].astype(str).to_numpy()

    for method, emb in emb_2d.items():
        slug = _method_slug(method)
        for k in target_ks:
            q = q_by_k[k]
            dom = q.argmax(axis=1).astype(int)
            h_norm = _entropy_norm(q)

            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            pred = km.fit_predict(emb)
            ari = float(adjusted_rand_score(dom, pred))
            nmi = float(normalized_mutual_info_score(dom, pred))
            local_dom = _local_purity(emb, dom, k=15)
            local_meta = _local_purity(emb, label_meta, k=15)
            log_sep = _logistic_separability(emb, dom, seed=seed)

            rows.append({
                'method': method,
                'method_slug': slug,
                'K': int(k),
                'ari': ari,
                'nmi': nmi,
                'local_purity_dominant': local_dom,
                'local_purity_metadata': local_meta,
                'logistic_separability': log_sep,
            })

            cmap = {int(i): alignment['colors_by_k'][k][i] for i in range(len(alignment['colors_by_k'][k]))}
            _plot_embedding_by_labels(
                emb, dom, cmap,
                fig_dir / f'embedding_{slug}_by_dominantK{k}.png',
                f'Geometric arrangement of samples; colors denote dominant ancestry component (K={k}, {method})',
                theme,
            )
            _plot_embedding_by_continuous(
                emb, h_norm,
                fig_dir / f'embedding_{slug}_by_entropyK{k}.png',
                f'Geometric arrangement of samples; colors denote normalized entropy (K={k}, {method})',
                theme,
                cmap='magma',
            )

    df = pd.DataFrame(rows)
    # Embedding method ranking at broad K
    broad_k = min(target_ks)
    broad = df[df['K'] == broad_k].copy()
    broad = broad.sort_values(['nmi', 'local_purity_dominant', 'logistic_separability', 'ari', 'method'],
                              ascending=[False, False, False, False, True]).reset_index(drop=True)
    broad['embedding_method_rank'] = np.arange(1, len(broad) + 1, dtype=int)
    df = df.merge(broad[['method', 'embedding_method_rank']], on='method', how='left')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    return df


def _extract_splits(tree) -> set[frozenset]:
    tips = [t.name for t in tree.tips()]
    all_leaves = set(tips)
    n = len(all_leaves)
    splits = set()
    for node in tree.non_tips():
        subset = {t.name for t in node.tips()}
        if 1 < len(subset) < (n - 1):
            canonical = subset if len(subset) <= (n / 2) else (all_leaves - subset)
            splits.add(frozenset(canonical))
    return splits


def _group_centroids(X: np.ndarray, labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    centroids = pd.DataFrame(np.asarray(X)).groupby(np.asarray(labels).astype(str), sort=True).mean()
    return centroids.index.tolist(), centroids.to_numpy()


def _draw_tree_panel(ax, tree, title: str, theme: dict):
    """Minimal matplotlib renderer for skbio TreeNode."""
    t = theme
    ax.set_facecolor(t['ax_bg'])
    tips = list(tree.tips())
    y_map = {tip.name: float(i) for i, tip in enumerate(tips)}
    x_map = {}

    def _set_x(node, x0):
        x_map[node] = x0
        for ch in node.children:
            bl = float(ch.length) if ch.length is not None else 1.0
            _set_x(ch, x0 + bl)

    _set_x(tree, 0.0)

    def _node_y(node):
        if node.is_tip():
            return y_map[node.name]
        ys = [_node_y(ch) for ch in node.children]
        return float(np.mean(ys))

    y_node = {node: _node_y(node) for node in tree.postorder()}
    for node in tree.preorder():
        if node.is_tip():
            continue
        x = x_map[node]
        ys = [y_node[ch] for ch in node.children]
        ax.plot([x, x], [min(ys), max(ys)], color=t['text'], linewidth=0.75, alpha=0.9)
        for ch in node.children:
            xc = x_map[ch]
            yc = y_node[ch]
            ax.plot([x, xc], [yc, yc], color=t['text'], linewidth=0.75, alpha=0.9)
            if ch.is_tip():
                ax.text(xc + 0.03, yc, str(ch.name), va='center', ha='left', fontsize=5.5, color=t['text'])
    ax.set_title(title, fontsize=9.5, color=t['text'])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(t['grid'])


def build_nj_tree_figure(X_scaled: np.ndarray, meta: pd.DataFrame, fig_path: Path, theme: dict, *,
                         n_bootstrap: int = 100, seed: int = ANALYSIS_SEED) -> dict:
    """Build bootstrapped NJ trees for breed and clade centroids and plot one figure."""
    try:
        from skbio import DistanceMatrix
        from skbio.tree import nj
    except Exception as exc:
        raise ImportError(
            'scikit-bio is required for NJ tree generation. Install with `pip install scikit-bio`.'
        ) from exc

    rng = np.random.default_rng(seed)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    def _build(labels: np.ndarray, label_name: str):
        groups, cent = _group_centroids(X_scaled, labels)
        D = squareform(pdist(cent, metric='euclidean'))
        base_tree = nj(DistanceMatrix(D, ids=groups))
        base_splits = _extract_splits(base_tree)
        split_counts = {s: 0 for s in base_splits}

        n_features = X_scaled.shape[1]
        for _ in range(int(n_bootstrap)):
            cols = rng.integers(0, n_features, n_features)
            Xb = X_scaled[:, cols]
            _, cent_b = _group_centroids(Xb, labels)
            Db = squareform(pdist(cent_b, metric='euclidean'))
            tree_b = nj(DistanceMatrix(Db, ids=groups))
            splits_b = _extract_splits(tree_b)
            for s in base_splits:
                if s in splits_b:
                    split_counts[s] += 1

        supports = {s: 100.0 * split_counts[s] / max(int(n_bootstrap), 1) for s in base_splits}
        sup_vals = np.array(list(supports.values())) if supports else np.array([])
        stats = {
            'label_name': label_name,
            'n_groups': int(len(groups)),
            'n_internal_splits': int(len(base_splits)),
            'n_support_ge_50': int((sup_vals >= 50).sum()) if sup_vals.size else 0,
            'n_support_ge_70': int((sup_vals >= 70).sum()) if sup_vals.size else 0,
            'n_support_ge_90': int((sup_vals >= 90).sum()) if sup_vals.size else 0,
            'pct_support_ge_70': float((sup_vals >= 70).mean()) if sup_vals.size else 0.0,
        }
        return base_tree, stats

    breed_tree, breed_stats = _build(meta['FID'].astype(str).to_numpy(), 'breed')
    clade_tree, clade_stats = _build(meta['breed_group'].astype(str).to_numpy(), 'clade')

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 9.8), dpi=180, facecolor=theme['fig_bg'])
    _draw_tree_panel(axes[0], breed_tree, 'NJ tree (breed centroids)', theme)
    _draw_tree_panel(axes[1], clade_tree, 'NJ tree (clade centroids)', theme)
    fig.suptitle('Distance-based grouping with bootstrap support (NJ)', fontsize=11, color=theme['text'])
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=theme['fig_bg'])
    plt.close(fig)
    return {'breed': breed_stats, 'clade': clade_stats}


def build_parker_comparison_table(admix_df: pd.DataFrame, embedding_df: pd.DataFrame, tree_stats: dict,
                                  k_selection: dict, out_path: Path) -> pd.DataFrame:
    """Create deterministic Parker-style comparison table and classification flag."""
    k_b = int(k_selection['k_broad_default'])
    k_f = int(k_selection['k_fine_default'])
    k_rows = admix_df[admix_df['row_type'] == 'k_summary'].copy()
    rb = k_rows[k_rows['K'] == k_b].iloc[0]
    rf = k_rows[k_rows['K'] == k_f].iloc[0]

    emb_b = embedding_df[embedding_df['K'] == k_b].sort_values('nmi', ascending=False).iloc[0]
    emb_f = embedding_df[embedding_df['K'] == k_f].sort_values('nmi', ascending=False).iloc[0]

    tree_high = bool(tree_stats['breed']['pct_support_ge_70'] >= 0.5)
    mean_qmax = float(rb['mean_q_max'])
    mean_h = float(rb['mean_h_norm'])
    local_purity = float(emb_b['local_purity_dominant'])
    ari_b = float(emb_b['ari'])
    nmi_b = float(emb_b['nmi'])

    if tree_high and (mean_qmax >= 0.80) and (mean_h <= 0.25) and (local_purity >= 0.80):
        flag = 'match_discrete'
    elif (float(rb['mean_q_max']) >= 0.75) and (float(rf['mean_h_norm']) > float(rb['mean_h_norm']) + 0.05) and \
            (float(rf['adjacent_dominant_preserved_frac']) < float(rb['adjacent_dominant_preserved_frac']) - 0.05):
        flag = 'match_hierarchical'
    elif (mean_qmax < 0.65) and (mean_h > 0.40) and (ari_b < 0.20) and (0.20 <= nmi_b <= 0.65) and (0.50 <= local_purity < 0.80):
        flag = 'match_continuous'
    else:
        flag = 'mixed_pattern'

    rows = [
        {
            'feature': 'tree_discreteness',
            'parker_pattern': 'supported internal clades in bootstrapped NJ tree',
            'our_value': json.dumps(tree_stats['breed']),
            'comparison_flag': flag,
        },
        {
            'feature': 'assignment_discreteness',
            'parker_pattern': 'island-like breeds imply high assignment purity',
            'our_value': json.dumps({'mean_q_max': mean_qmax, 'mean_h_norm': mean_h, 'mean_k_eff': float(rb['mean_k_eff'])}),
            'comparison_flag': flag,
        },
        {
            'feature': 'within_vs_between_cohesion',
            'parker_pattern': 'within-clade sharing much larger than across-clade',
            'our_value': 'distance-only fallback: ordered PCA-distance heatmap + within/between proxy',
            'comparison_flag': flag,
        },
        {
            'feature': 'migration_admixture',
            'parker_pattern': 'Treemix migration edges + haplotype sharing',
            'our_value': 'not run; distance-only fallback used (Treemix/IBD unavailable in this run)',
            'comparison_flag': flag,
        },
        {
            'feature': 'embedding_separability',
            'parker_pattern': 'tight separable clusters',
            'our_value': json.dumps({
                'broad': {'K': k_b, 'ari': ari_b, 'nmi': nmi_b, 'local_purity': local_purity},
                'fine': {'K': k_f, 'ari': float(emb_f['ari']), 'nmi': float(emb_f['nmi']),
                         'local_purity': float(emb_f['local_purity_dominant'])},
            }),
            'comparison_flag': flag,
        },
    ]
    df = pd.DataFrame(rows, columns=['feature', 'parker_pattern', 'our_value', 'comparison_flag'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    return df


def plot_comparison_dashboard(cv_df: pd.DataFrame, k_selection: dict, admix_df: pd.DataFrame,
                              embedding_df: pd.DataFrame, parker_df: pd.DataFrame,
                              fig_path: Path, theme: dict) -> Path:
    """Compact dashboard summarizing K selection + concordance + Parker flag."""
    t = theme
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    k_rows = admix_df[admix_df['row_type'] == 'k_summary'].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), dpi=180, facecolor=t['fig_bg'])
    ax1, ax2, ax3, ax4 = axes.flatten()
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(t['ax_bg'])
        ax.grid(alpha=0.24, linewidth=0.5, color=t['grid'])
        ax.tick_params(colors=t['text'], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(t['grid'])

    valid = cv_df[cv_df['cv_error'].notna()].sort_values('K')
    ax1.plot(valid['K'], valid['cv_error'], marker='o', color='#60a5fa', linewidth=1.6)
    ax1.set_title('predictive error by K', fontsize=10)
    ax1.set_xlabel('K')
    ax1.set_ylabel('CV error')

    sel_ks = [int(k_selection['k_broad_default']), int(k_selection['k_fine_default']), int(k_selection['k_cv_min'])]
    sub = k_rows[k_rows['K'].isin(sel_ks)].copy().sort_values('K')
    ax2.bar(sub['K'].astype(str), sub['mean_q_max'], label='mean Qmax', color='#34d399')
    ax2_t = ax2.twinx()
    ax2_t.plot(sub['K'].astype(str), sub['mean_h_norm'], marker='o', color='#f59e0b', label='mean H_norm')
    ax2.set_title('discreteness vs continuity', fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2_t.set_ylim(0, 1.0)

    emb_b = embedding_df[embedding_df['K'] == int(k_selection['k_broad_default'])].sort_values('nmi', ascending=False)
    ax3.bar(emb_b['method'], emb_b['nmi'], color='#a78bfa')
    ax3.set_title('embedding NMI at broad K', fontsize=10)
    ax3.set_ylabel('NMI')
    ax3.tick_params(axis='x', rotation=25)

    ax4.axis('off')
    flag = parker_df['comparison_flag'].iloc[0] if not parker_df.empty else 'mixed_pattern'
    lines = [
        'Parker comparison summary',
        f'classification: {flag}',
        f"k_broad={int(k_selection['k_broad_default'])}, "
        f"k_fine={int(k_selection['k_fine_default'])}, "
        f"k_cv_min={int(k_selection['k_cv_min'])}",
        '',
        'Claims:',
        '- predictive error by K',
        '- estimated ancestry proportions under model K',
        '- geometric arrangement of samples',
        '- distance-based grouping with bootstrap support',
    ]
    ax4.text(0.02, 0.98, '\n'.join(lines), transform=ax4.transAxes, va='top', ha='left',
             fontsize=8.3, color=t['text'], family='monospace')

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight', facecolor=t['fig_bg'])
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# 8) Summary export
# ---------------------------------------------------------------------------
