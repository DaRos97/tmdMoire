"""Standalone w1p/w1d distance heatmap for EDC Gamma results.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py.

Produces:
  distance_w_heatmap.png  -- 2D heatmap of min distance over (w1p, w1d)

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

    if "w1p_vals_meV" not in d:
        print("Error: .npz does not contain w1p/w1d distance data.")
        print("This .npz was exported with an older version of export_edc_gamma_data.py.")
        print("Re-export with the current version.")
        sys.exit(1)

    run_id = str(d["run_id"])
    w1p_vals_meV = d["w1p_vals_meV"]
    w1d_vals_meV = d["w1d_vals_meV"]
    dist_w_2d_meV = d["dist_w_2d_meV"]
    w1p_edges_meV = d["w1p_edges_meV"]
    w1d_edges_meV = d["w1d_edges_meV"]
    best_w1p_ev = float(d["best_w1p_ev"])
    best_w1d_ev = float(d["best_w1d_ev"])
    best_dist_meV = float(d["best_dist_meV"])

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    im = ax.pcolormesh(
        w1p_edges_meV, w1d_edges_meV, dist_w_2d_meV,
        cmap="viridis_r", shading="flat",
    )

    ax.scatter(
        best_w1p_ev * 1000, best_w1d_ev * 1000,
        marker="*", s=200, c="red", edgecolors="white", linewidths=1.0, zorder=6,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Min distance (meV)", fontsize=11)

    ax.set_xlabel(r"$w_{1p}$ (meV)", fontsize=12)
    ax.set_ylabel(r"$w_{1d}$ (meV)", fontsize=12)
    ax.set_title(
        f"EDC Gamma: min distance over Vg, phiG\n"
        f"Run: {run_id}  |  best: {best_dist_meV:.1f} meV",
        fontsize=12,
    )

    fig.savefig(output_dir / "distance_w_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_w_heatmap.png'}")


if __name__ == "__main__":
    main()
