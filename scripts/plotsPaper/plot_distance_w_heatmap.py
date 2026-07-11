"""Standalone w1p/w1d distance heatmaps for EDC Gamma results.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py.

Produces:
  distance_w_heatmap.png  -- 1x2 subplots (L1 + separation) over (w1p, w1d)

Usage:
    python plot_distance_w_heatmap.py <data.npz>
    python plot_distance_w_heatmap.py data.npz --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_distance_w_heatmap.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path(".")

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path)

    run_id = str(d["run_id"])
    w1p_edges_meV = d["w1p_edges_meV"]
    w1d_edges_meV = d["w1d_edges_meV"]
    dist_w_2d_meV = d["dist_w_2d_meV"]
    dist_sep_w_2d_meV = d["dist_sep_w_2d_meV"]

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    for ax, d2d, title in [
        (ax1, dist_w_2d_meV, r"L1 distance: $\Sigma\,|c_i - E_i^{\mathrm{exp}}|$"),
        (ax2, dist_sep_w_2d_meV, r"Separation: $\Sigma\,|\Delta E - \Delta E^{\mathrm{exp}}|$"),
    ]:
        im = ax.pcolormesh(w1p_edges_meV, w1d_edges_meV, d2d,
                           cmap="viridis_r", shading="flat")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Min distance (meV)", fontsize=11)
        ax.set_xlabel(r"$w_{1p}$ (meV)", fontsize=12)
        ax.set_ylabel(r"$w_{1d}$ (meV)", fontsize=12)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"EDC Gamma: min distance over Vg, phiG  |  Run: {run_id}", fontsize=13, y=1.02)
    fig.savefig(output_dir / "distance_w_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_w_heatmap.png'}")


if __name__ == "__main__":
    main()
