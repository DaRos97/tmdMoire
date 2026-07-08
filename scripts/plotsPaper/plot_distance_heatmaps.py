"""Standalone distance heatmap plots for EDC Gamma results.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py.

Produces:
  distance_heatmap.png       -- full-range 2D heatmap
  distance_heatmap_zoom.png  -- zoomed to phiG in [150, 210]

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


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_distance_heatmaps.py <data.npz> [--output-dir <dir>]")
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
    Vg_vals_meV = d["Vg_vals_meV"]
    dist_2d_meV = d["dist_2d_meV"]
    phi_edges = d["phi_edges"]
    Vg_edges_meV = d["Vg_edges_meV"]
    best_Vg_meV = float(d["best_Vg_meV"])
    best_phiG_deg = float(d["best_phiG_deg"])
    best_dist_meV = float(d["best_dist_meV"])
    best_w1p_ev = float(d["best_w1p_ev"])
    best_w1d_ev = float(d["best_w1d_ev"])

    output_dir.mkdir(parents=True, exist_ok=True)

    vg_line_vals = np.arange(8, float(np.nanmax(Vg_vals_meV)) + 0.1, 2)
    vg_max = float(np.nanmax(Vg_vals_meV))

    # ── Full-range heatmap ───────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    im = ax.pcolormesh(
        phi_edges, Vg_edges_meV, dist_2d_meV,
        cmap="viridis_r", shading="flat",
    )

    ax.scatter(
        best_phiG_deg, best_Vg_meV,
        marker="*", s=200, c="red", edgecolors="white", linewidths=1.0, zorder=6,
    )

    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.5, alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Min distance (meV)", fontsize=11)

    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_ylim(bottom=None, top=vg_max)
    ax.set_title(
        f"EDC Gamma: min distance over interlayer params\n"
        f"Run: {run_id}  |  best: {best_dist_meV:.1f} meV  |  "
        f"w1p={best_w1p_ev:.3f}, w1d={best_w1d_ev:.3f}",
        fontsize=12,
    )

    fig.savefig(output_dir / "distance_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_heatmap.png'}")

    # ── Zoomed heatmap ───────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    im = ax.pcolormesh(
        phi_edges, Vg_edges_meV, dist_2d_meV,
        cmap="viridis_r", shading="flat",
    )

    ax.scatter(
        best_phiG_deg, best_Vg_meV,
        marker="*", s=200, c="red", edgecolors="white", linewidths=1.0, zorder=6,
    )

    for deg in [60, 180, 300]:
        ax.axvline(x=deg, color="red", ls="--", lw=1, alpha=0.6)
    for v in vg_line_vals:
        ax.axhline(y=v, color="gray", ls="--", lw=0.3, alpha=0.6, zorder=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Min distance (meV)", fontsize=11)

    ax.set_xlabel(r"$\phi_G$ (deg)", fontsize=12)
    ax.set_ylabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_xlim(150, 210)
    ax.set_ylim(bottom=None, top=vg_max)
    ax.set_title(
        f"EDC Gamma: distance heatmap (zoom)\n"
        f"Run: {run_id}  |  best: {best_dist_meV:.1f} meV",
        fontsize=12,
    )

    fig.savefig(output_dir / "distance_heatmap_zoom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'distance_heatmap_zoom.png'}")


if __name__ == "__main__":
    main()
