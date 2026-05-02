"""Fetch Parker 2017 supplementary assets used by the analysis.

The breed-clade mapping (`parker_clades.csv`) and the NJ tree
(`parker_tree/parker_2017_nj.nex`) are derived from the supplementary
materials of:

    Parker HG et al. (2017). Genomic Analyses Reveal the Influence of
    Geographic Origin, Migration, and Hybridization on Modern Dog Breed
    Development. Cell Reports 19(4):697-708.
    DOI: 10.1016/j.celrep.2017.03.079

These small files are not committed to the repo. The notebook calls
`ensure_parker_assets()` once during setup; if a local copy exists under
`data/`, it is used as-is. Otherwise a download is attempted; on failure,
explicit manual-download instructions are raised.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

PARKER_ARTICLE_URL = "https://www.cell.com/cell-reports/fulltext/S2211-1247(17)30456-4"
MIRROR_BASE = "https://github.com/aariya50/phyloae-dog-breeds/releases/download/data-v1"


def _download(url: str, target: Path, timeout: int = 30) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        target.write_bytes(r.read())


def ensure_parker_clades(data_dir: Path) -> Path:
    """Return local path to parker_clades.csv; download from mirror if missing."""
    target = Path(data_dir) / "parker_clades.csv"
    if target.exists():
        return target
    url = f"{MIRROR_BASE}/parker_clades.csv"
    try:
        _download(url, target)
    except (urllib.error.URLError, OSError) as e:
        raise FileNotFoundError(
            f"Parker breed-clade table not found at {target} and download from "
            f"{url} failed ({e}). The file is derived from Parker 2017 "
            f"supplementary materials; see {PARKER_ARTICLE_URL} or data/README.md."
        ) from e
    return target


def ensure_parker_nj_tree(data_dir: Path) -> Path:
    """Return local path to parker_2017_nj.nex; download from mirror if missing."""
    target = Path(data_dir) / "parker_tree" / "parker_2017_nj.nex"
    if target.exists():
        return target
    url = f"{MIRROR_BASE}/parker_2017_nj.nex"
    try:
        _download(url, target)
    except (urllib.error.URLError, OSError) as e:
        raise FileNotFoundError(
            f"Parker NJ tree not found at {target} and download from "
            f"{url} failed ({e}). The file is mmc5 from Parker 2017; see "
            f"{PARKER_ARTICLE_URL} or data/README.md."
        ) from e
    return target


def ensure_parker_assets(data_dir: Path) -> tuple[Path, Path]:
    """Ensure both Parker assets are available locally."""
    return ensure_parker_clades(data_dir), ensure_parker_nj_tree(data_dir)
