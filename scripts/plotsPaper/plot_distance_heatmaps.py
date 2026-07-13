"""Standalone distance heatmap plots for EDC Gamma results.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py.

Produces:
  distance_heatmap.png       -- 1x2 subplots (L1 + separation) over (Vg, phiG)

Usage:
    python plot_distance_heatmaps.py <data.npz>
    python plot_distance_heatmaps.py data.npz --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_heatmap(ax, phi_edges, Vg_edges_meV, d2d_meV, title, cmap="viridis_r"):
    im = ax.pcolormesh(phi_edges, Vg_edges_meV, d2d_meV,
                       cmap=cmap, shading="flat")
    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    vg_line_vals = np.arange(2, 21, 2)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.6)
    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_xticks([160, 170, 180, 190, 200])
    ax.set_yticks(np.arange(0, 21, 2))
    ax.set_ylim(0, 20)
    ax.set_xlim(160, 200)
    ax.set_title(title, fontsize=11)
    return im


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_distance_heatmaps.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path(__file__).resolve().parent / "figures"

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path)

    run_id = str(d["run_id"])
    phi_edges = d["phi_edges"]
    Vg_edges_meV = d["Vg_edges_meV"]
    dist_2d_meV = d["dist_2d_meV"]
    dist_sep_2d_meV = d["dist_sep_2d_meV"]

    output_dir.mkdir(parents=True, exist_ok=True)

    titles = [
        (dist_2d_meV, r"L1 distance: $\Sigma\,|c_i - E_i^{\mathrm{exp}}|$", "viridis_r"),
        (dist_sep_2d_meV, r"Separation: $\Sigma\,|\Delta E - \Delta E^{\mathrm{exp}}|$", "plasma_r"),
    ]

    # ── Full-range ────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    for ax, (d2d, title, cmap) in zip(axes, titles):
        im = _make_heatmap(ax, phi_edges, Vg_edges_meV, d2d, title, cmap)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Min distance (meV)", fontsize=11)

    fig.suptitle(f"EDC Gamma: {run_id}", fontsize=13, y=1.02)
    fig.savefig(output_dir / f"distance_heatmap_{run_id}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / f'distance_heatmap_{run_id}.png'}")


if __name__ == "__main__":
    main()
