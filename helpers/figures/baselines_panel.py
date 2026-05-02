"""Crop the 6-panel comparison grid to remove the PhyloAE panel (bottom-right)."""

from pathlib import Path
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "results" / "figures" / "presentation" / "all_methods_2d_comparison.png"
DST = PROJECT / "results" / "figures" / "presentation" / "baselines_5panel.png"


def main() -> None:
    img = Image.open(SRC)
    w, h = img.size
    col_w = w // 3
    row_h = h // 2
    draw = ImageDraw.Draw(img)
    draw.rectangle([2 * col_w, row_h, w, h], fill=(0x0E, 0x11, 0x17))
    img.save(DST)
    print(f"Saved {DST}  ({img.size[0]}x{img.size[1]}, {DST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
