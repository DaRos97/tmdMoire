"""Combined w1p/w1d L2 distance heatmaps for S11 and S3.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads two .npz files produced by scripts/export_edc_gamma_data.py.

Produces:
  SM_interlayer.pdf  -- 1x2 subplots: S11 L2 distance (left), S3 L2 distance (right)

Usage:
    python plot_interlayer_comparison.py <s11.npz> <s3.npz>
    python plot_interlayer_comparison.py <s11.npz> <s3.npz> --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python plot_interlayer_comparison.py <s11_data.npz> <s3_data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_paths = [Path(args[0]), Path(args[1])]
    output_dir = Path(__file__).resolve().parent / "figures"

    i = 2
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    labels = ["S11", "S3"]
    contour_targets = [(-1220.0, 460.0), (-1200.0, 455.0)]
    contour_labels = [
        r"$w_1^{p_z}$ = $-$1.220 eV" + "\n" + r"$w_1^{d_{z^2}}$ = +0.460 eV",
        r"$w_1^{p_z}$ = $-$1.200 eV" + "\n" + r"$w_1^{d_{z^2}}$ = +0.455 eV",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, dp, label, target, clabel in zip(axes, data_paths, labels, contour_targets, contour_labels):
        d = np.load(dp)

        w1p_edges_meV = d["w1p_edges_meV"]
        w1d_edges_meV = d["w1d_edges_meV"]
        dist_sep_w_2d_meV = d["dist_sep_w_2d_meV"]

        im = ax.pcolormesh(w1p_edges_meV, w1d_edges_meV, dist_sep_w_2d_meV,
                           cmap="plasma_r", shading="flat")

        if target is not None:
            ax.plot(target[0], target[1], "k", marker="s", markersize=12,
                    markeredgewidth=2.5, markerfacecolor="none", zorder=10)

        ax.set_xlabel(r"$w_1^{p_z}$ (meV)", fontsize=12)
        ax.set_ylabel(r"$w_1^{d_{z^2}}$ (meV)", fontsize=12)
        ax.text(0.95, 0.95, f"{label}\n{clabel}", transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    cbar = fig.colorbar(im, ax=axes, pad=0.02, shrink=0.92)
    cbar.set_label("Min distance $f$ (meV)", fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_fn = output_dir / "SM_interlayer.pdf"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")


if __name__ == "__main__":
    main()
