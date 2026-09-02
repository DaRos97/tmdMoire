"""Plot heterobilayer schematic and zone-folded parabolic moire bands side-by-side.

Left panel (0.5 width): schematic of hexagonal lattice with 19 hexagons
Right panel (0.5 width): parabolic moire bands along x-axis in mini-BZ

Output: Figures/fig_stacking_moire.png (half A4 width)
"""
import sys
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.parabolic import compute_parabolic_bands
from tmdmoire.utils.kpoints import R_z

import os
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

FIGURES_DIR = Path("Figures")
OUTPUT_FILE = FIGURES_DIR / "fig_stacking_moire.svg"

A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
FIG_WIDTH = A4_WIDTH_INCH / 2
FIG_HEIGHT = FIG_WIDTH / 2

cmap = plt.get_cmap("viridis")
ring_colors = [cmap(i) for i in [0, 0.5, 1.0]]


def draw_hexagon(ax, center, radius, color, alpha=1.0, zorder=1):
    """Draw a single hexagon centered at `center` with given `radius`."""
    angles = np.linspace(0, 2 * np.pi, 7)
    vertices = np.array([[center[0] + radius * np.cos(a),
                          center[1] + radius * np.sin(a)] for a in angles])
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=1.2, alpha=alpha, zorder=zorder)


def draw_heterobilayer_schematic(ax):
    """Draw schematic of a hexagonal lattice with 19 hexagons."""
    a_red = 2.8
    r_red = a_red / np.sqrt(3)
    r_draw = r_red * 0.95

    centers_by_ring = {0: [(0, 0)], 1: [], 2: []}
    g1 = np.array([a_red / 2 * np.sqrt(3), a_red / 2])
    for i in range(6):
        centers_by_ring[1].append(tuple(R_z(np.pi / 3 * i) @ g1))
    g1 = np.array([r_red * 3, 0])
    g2 = np.array([a_red * np.sqrt(3), a_red])
    for i in range(6):
        centers_by_ring[2].append(tuple(R_z(np.pi / 3 * i) @ g1))
        centers_by_ring[2].append(tuple(R_z(np.pi / 3 * i) @ g2))
    for ring in range(3):
        color = ring_colors[ring]
        for cx, cy in centers_by_ring[ring]:
            draw_hexagon(ax, (cx, cy), r_draw, color=color, alpha=1.0, zorder=1)

    geo = MoireGeometry(0.0)
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    G1 = G1 / np.linalg.norm(G1) * a_red
    G2 = G2 / np.linalg.norm(G2) * a_red
    ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=2)
    ax.arrow(0, 0, G1[0], G1[1],
             head_width=0.12, head_length=0.1, fc="black", ec="black", lw=1.2, zorder=3)
    ax.arrow(0, 0, G2[0], G2[1],
             head_width=0.12, head_length=0.1, fc="black", ec="black", lw=1.2, zorder=3)
    ax.text(G1[0] + 0.25, G1[1], r"$G_1$", fontsize=7, zorder=4)
    ax.text(G2[0] + 0.25, G2[1], r"$G_2$", fontsize=7, zorder=4)

    max_extent = a_red * 2.5
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-k", type=int, default=400, help="Number of k-points")
    parser.add_argument("--n-shells", type=int, default=2, help="Number of moire shells")
    parser.add_argument("--theta", type=float, default=0.0, help="Twist angle in degrees")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.35, 0.65], wspace=0.05)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    draw_heterobilayer_schematic(ax_left)

    print("Computing parabolic moire bands...")
    geo = MoireGeometry(args.theta)
    G_M = geo.reciprocal_vectors()
    K_mag = np.linalg.norm((G_M[1] + G_M[2]) / 3)
    k_vals, evals, _, sort_idx = compute_parabolic_bands(
        np.linspace(-5 * K_mag, 5 * K_mag, args.n_k),
        args.n_shells,
        geo,
    )

    c_center, c_shell1, c_shell2 = ring_colors

    n_bands = evals.shape[1]

    for i in range(n_bands):
        for j in range(len(k_vals) - 1):
            cell_idx = sort_idx[j, i]
            if cell_idx == 0:
                color = c_center
            elif cell_idx <= 6:
                color = c_shell1
            else:
                color = c_shell2
            ax_right.plot(k_vals[j:j + 2], evals[j:j + 2, i], color=color, lw=1.0)

    for k in np.arange(-4 * K_mag, 5 * K_mag, K_mag):
        ax_right.axvline(k, color="gray", lw=0.5, ls="--")

    ax_right.set_xlim(-5 * K_mag, 5 * K_mag)

    ax_right.set_xticks([-4 * K_mag, -3 * K_mag, -2 * K_mag, -K_mag, 0, K_mag, 2 * K_mag, 3 * K_mag, 4 * K_mag])
    ax_right.set_xticklabels(["K'", r"$\Gamma$", "K", "K'", r"$\Gamma$", "K", "K'", r"$\Gamma$", "K"], fontsize=7)
    ax_right.set_xlim(-5 * K_mag, 5 * K_mag)
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", labelsize=7)
    ax_right.set_ylim(-0.006, 0.0005)

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
