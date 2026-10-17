"""Twist-angle dependence of moire geometry and parabolic bands.

The top panel shows moire length and mini-BZ rotation versus twist angle,
with a mini-BZ construction redrawn as a vector inset. The bottom panel
compares parabolic bands with and without moire coupling.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import ConnectionPatch, PathPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometry, MoireGeometryFixed
from tmdmoire.constants import M_LIST


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
m = 1.19 * m0
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20

minibz_x_limits = (13.0, 78.0)
minibz_y_limits = (46.0, 5.0)


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


def compute_bands(theta, n_shells, n_k, k_range_factor, V_off, V_on, phi,
                  a_moire_override=None):
    geo = MoireGeometryFixed(theta, a_moire_override=a_moire_override)
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    K_mag = np.linalg.norm((G1 + G2) / 3)

    n_cells = MoireGeometryFixed.n_cells(n_shells)
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
    ax.set_ylim(-50, 5)


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


def plot_geometry_vs_theta(ax, theta_selected=0.8, theta_max=5.0, n_pts=200):
    """Plot moire length L_M (left axis) and mini-BZ rotation eta (right axis)
    as a function of twist angle theta.
    """
    thetas = np.linspace(0.01, theta_max, n_pts)
    geo = [MoireGeometry(t) for t in thetas]
    L_M = np.array([g.moire_length for g in geo])
    eta = np.array([g.mini_bz_rotation for g in geo])

    color_L = "#0072B2"
    ax.plot(thetas, L_M, color=color_L, lw=1.5, label=r"$L_M$")
    ax.set_xlabel(r"$\theta$ [deg]", labelpad=0)
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", top=True, labeltop=True,
                   bottom=False, labelbottom=False, pad=1)
    ax.set_ylabel(r"$a_M$ [$\mathrm{\AA}$]", color=color_L)
    ax.spines["left"].set_color(color_L)
    ax.tick_params(axis="y", left=True, right=False, colors=color_L)

    ax2 = ax.twinx()
    color_eta = "#E69F00"
    ax2.plot(thetas, np.degrees(eta), color=color_eta, lw=1.5, ls="--",
             label=r"$\eta$")
    ax2.set_ylabel(r"$\eta$ [deg]", color=color_eta)
    ax2.spines["right"].set_color(color_eta)
    ax2.spines["left"].set_visible(False)
    tick_label_size = ax.xaxis.label.get_size() - 1
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax2.tick_params(axis="y", left=False, right=True, colors=color_eta,
                    labelsize=tick_label_size)
    ax2.tick_params(axis="x", which="both", top=False, labeltop=False,
                    bottom=False, labelbottom=False)

    ax.axvline(theta_selected, color="gray", lw=1.0, ls="--")
    ax.set_xlim(0, theta_max)
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= 0) & (ticks <= theta_max)]
    ticks = np.unique(np.append(ticks, theta_selected))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])


def draw_minibz_diagram(ax, label_fontsize):
    """Draw a vector mini-BZ/twist schematic based on figures/miniBZ.svg.

    Coordinates follow the SVG's drawing coordinate system (y increases
    downward), making the relative placement of its hexagon, labels, arrows,
    reference lines, and angle arcs easy to retain.
    """
    color = "black"
    line_width = 0.65

    # Reference directions and the rotated mini-BZ hexagon.
    ax.plot([18.5557, 68.19597], [38.49802, 38.49802],
            color=color, lw=line_width, ls=(0, (2.4, 2.4)))
    ax.plot([18.528534, 67.741878], [38.504802, 33.809279],
            color=color, lw=line_width, ls=(0, (2.4, 2.4)))

    center = np.array([58.132, 26.498])
    radius = 12.06
    angles = np.deg2rad(37.34 + np.arange(6) * 60.0)
    vertices = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    vertices = np.vstack((vertices, vertices[0]))
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, lw=0.8)

    # Reciprocal-vector arrows from the mini-BZ center.
    ax.annotate("", xy=(50.075, 7.196), xytext=center,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.75,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(70.807, 9.858), xytext=center,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.75,
                                mutation_scale=6, shrinkA=0, shrinkB=0))

    # Curved angle indicators, represented by the SVG's cubic Bezier arcs.
    angle_arcs = [
        [(47.833636, 35.689433), (48.644331, 36.450524),
         (48.916950, 37.338347), (48.031110, 38.499020)],
        [(66.537161, 34.176907), (67.707786, 35.449303),
         (67.945351, 36.887891), (67.302773, 38.483247)],
    ]
    for points in angle_arcs:
        patch = PathPatch(
            MplPath(points, [MplPath.MOVETO, MplPath.CURVE4,
                             MplPath.CURVE4, MplPath.CURVE4]),
            fill=False, edgecolor=color, lw=0.65,
        )
        ax.add_patch(patch)

    # Math-text labels mirror the SVG while keeping Greek glyphs in math mode.
    labels = [
        (18.15, 36.85, r"$\mathbf{\Gamma}$", "center"),
        (67.13, 17.91, r"$\mathbf{G}_1$", "left"),
        (53.41, 12.5, r"$\mathbf{G}_2$", "left"),
        (68.66, 33.41, r"$\mathbf{K}_2$", "left"),
        (55.33, 43.0, r"$\mathbf{K}_1$", "left"),
        (44.0, 34.0, r"$\theta$", "left"),
        (69.15, 37.20, r"$\eta$", "left"),
    ]
    for x, y, label, alignment in labels:
        ax.text(x, y, label, ha=alignment, va="center", fontsize=label_fontsize,
                color=color, clip_on=False)

    ax.set_xlim(*minibz_x_limits)
    ax.set_ylim(*minibz_y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    ax.patch.set_edgecolor("none")
    ax.patch.set_linewidth(0)
    ax.patch.set_alpha(0.7)
    ax.patch.set_visible(True)


def main():
    n_shells = 1
    n_k = 800
    V_off = 0.0
    V_on = 2.0
    theta_band = 0.4
    a_moire = 76.0

    panel_band = compute_bands(theta_band, n_shells, n_k, 3, V_off, V_on, 0.0,
                               a_moire_override=a_moire)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "text.usetex": False,
    })

    fig = plt.figure(figsize=(3.4, 3.5))
    gs_outer = fig.add_gridspec(2, 1, hspace=0.40, height_ratios=[1, 1])

    ax_geom = fig.add_subplot(gs_outer[0])
    ax_bands = fig.add_subplot(gs_outer[1])

    plot_geometry_vs_theta(ax_geom, theta_selected=theta_band)
    plot_main_panel(ax_bands, *panel_band, max_size=30.0)
    ax_bands.set_ylabel("Energy")

    ax_geom.text(-0.18, 1.05, "a.", transform=ax_geom.transAxes,
                 fontsize=9, fontweight="bold", va="top", ha="left")
    ax_bands.text(-0.18, 1.05, "b.", transform=ax_bands.transAxes,
                  fontsize=9, fontweight="bold", va="top", ha="left")

    fig.subplots_adjust(left=0.15, right=0.87, top=0.91, bottom=0.06)

    inset_x, inset_y, inset_width = 0.2, -0.395, 0.62
    parent_pos = ax_geom.get_position()
    parent_aspect = (parent_pos.width * fig.get_figwidth()) / (
        parent_pos.height * fig.get_figheight()
    )
    diagram_aspect = abs(
        (minibz_x_limits[1] - minibz_x_limits[0]) /
        (minibz_y_limits[1] - minibz_y_limits[0])
    )
    inset_height = inset_width * parent_aspect / diagram_aspect
    inset_bounds = [
        parent_pos.x0 + inset_x * parent_pos.width,
        parent_pos.y0 + inset_y * parent_pos.height,
        inset_width * parent_pos.width,
        inset_height * parent_pos.height,
    ]
    minibz_ax = fig.add_axes(inset_bounds, zorder=20)
    draw_minibz_diagram(minibz_ax, label_fontsize=ax_geom.yaxis.label.get_size())

    for corner_x in (0.0, 1.0):
        connector = ConnectionPatch(
            xyA=(theta_band, 0.0),
            coordsA=ax_geom.get_xaxis_transform(),
            xyB=(corner_x, 1.0),
            coordsB=ax_bands.transAxes,
            axesA=ax_geom,
            axesB=ax_bands,
            color="gray",
            linewidth=0.5,
            clip_on=False,
            zorder=15,
        )
        fig.add_artist(connector)

    out = Path(__file__).with_name("figures") / "fig_2D_theta.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
