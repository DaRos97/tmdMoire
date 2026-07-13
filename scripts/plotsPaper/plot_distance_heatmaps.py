"""Standalone distance heatmap plots for EDC Gamma results.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py.

Produces:
  distance_heatmap.png       -- 1x2 subplots (L1 + separation) over (Vg, phiG)
  distance_heatmap_zoom.png  -- same, zoomed to phiG in [160, 200]

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


def _make_heatmap(ax, phi_edges, Vg_edges_meV, d2d_meV, Vg_vals_meV, title):
    im = ax.pcolormesh(phi_edges, Vg_edges_meV, d2d_meV,
                       cmap="viridis_r", shading="flat")
    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    vg_line_vals = np.arange(8, float(np.nanmax(Vg_vals_meV)) + 0.1, 2)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.6)
    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_xticks([160, 170, 180, 190, 200])
    ax.set_ylim(bottom=None, top=float(np.nanmax(Vg_vals_meV)))
    ax.set_xlim(160, 200)
    ax.set_title(title, fontsize=11)
    return im


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_distance_heatmaps.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path("figures")

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path)

    run_id = str(d["run_id"])
    Vg_vals_meV = d["Vg_vals_meV"]
    phi_edges = d["phi_edges"]
    Vg_edges_meV = d["Vg_edges_meV"]
    dist_2d_meV = d["dist_2d_meV"]
    dist_sep_2d_meV = d["dist_sep_2d_meV"]
    phiG_vals = d["phiG_vals_deg"]

    output_dir.mkdir(parents=True, exist_ok=True)

    titles = [
        (dist_2d_meV, r"L1 distance: $\Sigma\,|c_i - E_i^{\mathrm{exp}}|$"),
        (dist_sep_2d_meV, r"Separation: $\Sigma\,|\Delta E - \Delta E^{\mathrm{exp}}|$"),
    ]

    # ── Full-range ────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)

    for ax, (d2d, title) in zip(axes, titles):
        im = _make_heatmap(ax, phi_edges, Vg_edges_meV, d2d, Vg_vals_meV, title)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Min distance (meV)", fontsize=11)

    fig.suptitle(f"EDC Gamma: {run_id}", fontsize=13, y=1.02)
    fig.savefig(output_dir / "distance_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_heatmap.png'}")

    # ── Zoomed ────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    for ax, (d2d, title) in zip(axes, titles):
        im = _make_heatmap(ax, phi_edges, Vg_edges_meV, d2d, Vg_vals_meV, title)
        ax.set_xlim(160, 200)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Min distance (meV)", fontsize=11)

    fig.suptitle(f"EDC Gamma (zoom): {run_id}", fontsize=13, y=1.02)
    fig.savefig(output_dir / "distance_heatmap_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_heatmap_zoom.png'}")


if __name__ == "__main__":
    main()
