"""Theta cartoon: how the twist angle reshapes the moire bands.

Same physics as 2D_cartoon.py (parabolic dispersion + nearest-neighbor
moire coupling V*exp(i*phi)) but sweeps the twist angle to show how the
mini-BZ size and the resulting band structure evolve with theta.

Subplot (1) top-left: moire length L_M (left axis) and mini-BZ rotation
eta (right axis) as a function of twist angle.
Subplot (2) top-right: bands along K'->Gamma->K for theta = 0.5 deg
at V=0 (red lines) and V=0.4 meV (blue circles sized by central-cell
weight).
Bottom row: reserved (empty).

Total figure width matches 2D_cartoon.py (6.75 in).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.constants import M_LIST
from tmdmoire.utils.kpoints import R_z


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
m = 3.5 * m0
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def build_hamiltonian(k_vec, geo, n_shells, V, phi):
    """Build the (n_cells x n_cells) Hamiltonian at a given 2D k-point.

    Diagonal: parabolic E = -alpha * |k + G_c|^2 with alpha from 2D_cartoon.py
    Off-diagonal: V*exp(i*phi) for nearest-neighbor cells, following the
    alternating-conjugate pattern from MoireHamiltonian._build_moire_potential.
    """
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    lu = MoireGeometry.lu_table(n_shells)
    n_cells = len(lu)

    G_cells = np.array([lu[c][0] * G1 + lu[c][1] * G2 for c in range(n_cells)])

    H = np.zeros((n_cells, n_cells), dtype=complex)
    for c in range(n_cells):
        k_sq = np.sum((k_vec + G_cells[c]) ** 2)
        H[c, c] = -alpha * k_sq

    for s in range(n_cells):
        for g_idx, m_vec in enumerate(M_LIST):
            nn_coord = (lu[s][0] + m_vec[0], lu[s][1] + m_vec[1])
            try:
                nn = lu.index(nn_coord)
            except ValueError:
                continue
            v_s = V * np.exp(1j * phi) if g_idx % 2 else V * np.exp(-1j * phi)
            H[s, nn] += v_s

    return H


def compute_bands(theta, n_shells, n_k, k_range_factor, V_off, V_on, phi):
    geo = MoireGeometry(theta)
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    K_mag = np.linalg.norm((G1 + G2) / 3)

    n_cells = MoireGeometry.n_cells(n_shells)
    k_vals = np.linspace(-k_range_factor * K_mag, k_range_factor * K_mag, n_k)

    evals_off = np.empty((n_k, n_cells))
    evals_on = np.empty((n_k, n_cells))
    evecs_on = np.empty((n_k, n_cells, n_cells), dtype=complex)
    for i, k in enumerate(k_vals):
        k_vec = np.array([k, 0.0])
        H_off = build_hamiltonian(k_vec, geo, n_shells, V_off, phi)
        evals_off[i] = np.linalg.eigvalsh(H_off)
        H_on = build_hamiltonian(k_vec, geo, n_shells, V_on, phi)
        w, v = np.linalg.eigh(H_on)
        evals_on[i] = w
        evecs_on[i] = v

    return k_vals, K_mag, n_cells, evals_off, evals_on, evecs_on


def decorate_main_axis(ax):
    for k in np.arange(-3, 4):
        ax.axvline(k, color="gray", lw=0.5, ls="--")
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([r"$\Gamma$", r"$K$", r"$K'$",
                        r"$\Gamma$", r"$K$", r"$K'$", r"$\Gamma$"])
    ax.set_xlim(-3, 3)
    ax.set_yticks([])
    ax.set_ylim(-30, 5)


def plot_main_panel(ax, k_vals, K_mag, n_cells, evals_off, evals_on, evecs_on,
                    max_size=60.0):
    for band in range(n_cells):
        ax.plot(k_vals / K_mag, evals_off[:, band], color="red", lw=0.2, zorder=5)

    central_weight = np.abs(evecs_on[:, 0, :]) ** 2
    for band in range(n_cells):
        ax.scatter(
            k_vals / K_mag,
            evals_on[:, band],
            s=central_weight[:, band] * max_size,
            c="C0",
            linewidths=0,
            zorder=3,
        )

    decorate_main_axis(ax)


def plot_geometry_vs_theta(ax, theta_max=5.0, n_pts=200):
    """Plot moire length L_M (left axis) and mini-BZ rotation eta (right axis)
    as a function of twist angle theta.
    """
    thetas = np.linspace(0.01, theta_max, n_pts)
    geo = [MoireGeometry(t) for t in thetas]
    L_M = np.array([g.moire_length for g in geo])
    eta = np.array([g.mini_bz_rotation for g in geo])

    color_L = "#0072B2"
    ax.plot(thetas, L_M, color=color_L, lw=1.5, label=r"$L_M$")
    ax.set_xlabel(r"$\theta$ (deg)", labelpad=0)
    ax.set_ylabel(r"$a_{\mathrm{moir\'e}}$ ($\mathrm{\AA}$)", color=color_L)
    ax.tick_params(axis="y", labelcolor=color_L)

    ax2 = ax.twinx()
    color_eta = "#E69F00"
    ax2.plot(thetas, np.degrees(eta), color=color_eta, lw=1.5, ls="--",
             label=r"$\eta$")
    ax2.set_ylabel(r"$\eta$ (deg)", color=color_eta)
    ax2.tick_params(axis="y", labelcolor=color_eta)
    ax2.axhline(0.0, color="gray", lw=0.5, ls=":")

    ax.axvline(0.8, color="gray", lw=1.0, ls="--")
    ax.set_xlim(0, theta_max)


def _draw_hexagon(ax, center, radius, color, alpha=1.0, zorder=1):
    angles = np.linspace(0, 2 * np.pi, 7)
    vertices = np.array([[center[0] + radius * np.cos(a),
                          center[1] + radius * np.sin(a)] for a in angles])
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=1.2,
            alpha=alpha, zorder=zorder)


def draw_hexagon_cartoon(ax, a_red=5.0):
    """19-hexagon schematic of the moire supercell (from plot_stacking_moire.py),
    rotated as a single rigid group by eta(theta=0.8)."""
    cmap = plt.get_cmap("viridis")
    ring_colors = [cmap(i) for i in [0, 0.5, 1.0]]

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

    eta = MoireGeometry(0.8).mini_bz_rotation
    rot = R_z(eta)

    for ring in range(2):
        color = ring_colors[ring]
        for cx, cy in centers_by_ring[ring]:
            angles = np.linspace(0, 2 * np.pi, 7)
            vertices = np.array([[cx + r_draw * np.cos(a),
                                 cy + r_draw * np.sin(a)] for a in angles])
            rotated_v = (rot @ vertices.T).T
            ax.plot(rotated_v[:, 0], rotated_v[:, 1], color=color,
                    linewidth=1.2, alpha=1.0, zorder=1)

    geo = MoireGeometry(0.0)
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    G1 = G1 / np.linalg.norm(G1) * a_red
    G2 = G2 / np.linalg.norm(G2) * a_red
    G1 = rot @ G1
    G2 = rot @ G2
    ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=2)
    ax.arrow(0, 0, G1[0], G1[1],
             head_width=0.12, head_length=0.1, fc="black", ec="black", lw=1.2, zorder=3)
    ax.arrow(0, 0, G2[0], G2[1],
             head_width=0.12, head_length=0.1, fc="black", ec="black", lw=1.2, zorder=3)

    max_extent = a_red * 1.7
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    n_shells = 1
    n_k = 800
    V_off = 0.0
    V_on = 0.8
    theta_band = 0.8

    panel_band = compute_bands(theta_band, n_shells, n_k, 3, V_off, V_on, 0.0)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
    })

    fig = plt.figure(figsize=(3.4, 3.5))
    gs_outer = fig.add_gridspec(2, 1, hspace=0.35, height_ratios=[1, 1])

    ax_geom = fig.add_subplot(gs_outer[0])
    ax_bands = fig.add_subplot(gs_outer[1])

    plot_geometry_vs_theta(ax_geom)
    ax_geom.text(0.20, 0.15, r"$\theta = 0.8^\circ$", transform=ax_geom.transAxes,
                 ha="left", va="center", color="black",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="black", lw=0.8))
    ax_hex = ax_geom.inset_axes([0.55, 0.28, 0.45, 0.52])
    draw_hexagon_cartoon(ax_hex, a_red=3.5)
    plot_main_panel(ax_bands, *panel_band, max_size=30.0)
    ax_bands.set_ylabel("Energy")

    fig.subplots_adjust(left=0.18, right=0.82, top=0.96, bottom=0.09)

    out = Path(__file__).with_name("figures") / "fig_2D_theta.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
