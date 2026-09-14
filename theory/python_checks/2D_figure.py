"""2D moire figure: combined cartoon + phase sweep + V/phi sweeps of observables.

Left column (width 0.8): bands along K'->Gamma->K for theta=0 (top)
and theta=1 (bottom) at V=0 (red lines) and V=0.4 meV (blue circles sized
by central-cell weight).

Right column, top (width 1.2 split into 4): same as left top but for
moire phases phi = 0, 30, 60, 90 degrees over a small k-range around Gamma.

Right column, bottom (split into 4): 4 observables loaded from
data/2D_analysis_sweeps.npz, with twin x-axes:
  - bottom: V (meV) for the V-sweep curve (phi = 180 deg)
  - top:    phi (deg) for the phi-sweep curve (V = 0.2 meV)

Observables (renamed):
  (a) Delta   = minimum gap
  (b) chi     = Delta chi at Gamma (top band - highest-w band excl. top, vs V=0)
  (c) lambda  = weight ratio at k=-2 K_M (same k)
  (d) rho     = weight ratio at k_cross < -2 K_M

Total figure width matches 1D_figure.py (6.75 in).
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.constants import M_LIST


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
m = 3.5 * m0
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def build_hamiltonian(k_vec, geo, n_shells, V, phi):
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
            rasterized=True,
        )

    legend_elements = [
        Line2D([0], [0], color="red", lw=1.0, label=r"$V = 0$ meV"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
               markersize=8, label=r"$V = 0.4$ meV"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", frameon=False)

    decorate_main_axis(ax)


def plot_inset(ax, k_vals, K_mag, n_cells, evals_off, evals_on, evecs_on,
               max_size=60.0, marker_boost=5.0):
    for band in range(n_cells):
        ax.plot(k_vals / K_mag, evals_off[:, band], color="red", lw=0.2, zorder=5)

    central_weight = np.abs(evecs_on[:, 0, :]) ** 2
    for band in range(n_cells):
        ax.scatter(
            k_vals / K_mag,
            evals_on[:, band],
            s=central_weight[:, band] * max_size * marker_boost,
            c="C0",
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-12, -8)
    ax.axvline(0.0, color="gray", lw=0.5, ls="--")
    ax.text(0.0, -0.05, r"$\Gamma$", transform=ax.get_xaxis_transform(),
            ha="center", va="top")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    n_shells = 1
    n_k = 800
    V_off = 0.0
    V_on = 0.4
    thetas = [0.0, 1.0]
    phase_degs = [0, 20, 40, 60]

    panels = [compute_bands(theta, n_shells, n_k, 3, V_off, V_on, 0.0)
              for theta in thetas]

    phase_panels = []
    for phase_deg in phase_degs:
        phi = phase_deg * np.pi / 180.0
        phase_panels.append(
            compute_bands(0.0, n_shells, n_k, 0.5, V_off, V_on, phi)
        )

    sweeps_path = Path(__file__).with_name("data") / "2D_analysis.npz"
    if sweeps_path.exists():
        sw = np.load(sweeps_path)
        Vs = sw["V_V"]
        phi_degs = sw["phi_deg_phi"]
        Delta_v, chi_v = sw["delta_V"], sw["chi_V"]
        lambda_v, rho_v = sw["lambda_V"], sw["rho_V"]
        Delta_p, chi_p = sw["delta_phi"], sw["chi_phi"]
        lambda_p, rho_p = sw["lambda_phi"], sw["rho_phi"]
    else:
        print(f"WARNING: {sweeps_path} not found; bottom-right panels will be empty.")
        Vs = np.array([0.0, 1.0])
        phi_degs = np.array([0.0, 120.0])
        Delta_v = chi_v = lambda_v = rho_v = np.zeros(2)
        Delta_p = chi_p = lambda_p = rho_p = np.zeros(2)

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig = plt.figure(figsize=(6.75, 4.5))
    gs_outer = fig.add_gridspec(2, 2, width_ratios=[0.7, 1.3], wspace=0.05,
                                hspace=0.45, height_ratios=[2.0, 1.0])

    ax_left_top = fig.add_subplot(gs_outer[0, 0])

    plot_main_panel(ax_left_top, *panels[0], max_size=20.0)
    ax_left_top.set_ylabel("Energy")

    gs_right_top = gs_outer[0, 1].subgridspec(1, 4, wspace=0.20)
    ax_right_top = [fig.add_subplot(gs_right_top[0, i]) for i in range(4)]
    for ax, panel, phase_deg in zip(ax_right_top, phase_panels, phase_degs):
        plot_inset(ax, *panel)
        ax.set_title(rf"$\phi = {phase_deg}^\circ$", fontsize=10)

    gs_bottom = gs_outer[1, :].subgridspec(1, 4, wspace=0.20)
    ax_right_bot = [fig.add_subplot(gs_bottom[0, i]) for i in range(4)]
    ax_Delta, ax_chi, ax_lam, ax_rho = ax_right_bot

    label_box = dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                     edgecolor="black", linewidth=0.8)
    ax_Delta.text(0.5, 0.05, r"$\Delta$", transform=ax_Delta.transAxes,
                  ha="center", va="bottom", fontsize=12, bbox=label_box)
    ax_chi.text(0.5, 0.95, r"$\chi$", transform=ax_chi.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)
    ax_lam.text(0.5, 0.95, r"$\lambda$", transform=ax_lam.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)
    ax_rho.text(0.5, 0.95, r"$\rho$", transform=ax_rho.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)

    color_v = "#0072B2"
    color_p = "#E69F00"

    quantities = [
        (ax_Delta, Delta_v, Delta_p, "linear", True),
        (ax_chi, chi_v, chi_p, "linear", True),
        (ax_lam, lambda_v, lambda_p, "linear", True),
        (ax_rho, rho_v, rho_p, "linear", True),
    ]

    for ax, y_v, y_p, yscale, show_yticks in quantities:
        ax.plot(Vs, y_v, color=color_v, lw=1.5)
        ax.set_xlim(Vs[0], Vs[-1])
        ax.set_xlabel("V (meV)")
        ax.tick_params(axis="both", labelsize=7)
        if not show_yticks:
            ax.set_yticks([])
        if ax is ax_lam or ax is ax_rho:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_visible(False)
        ax.axhline(0.0, color="k", lw=0.5, ls=":")

        ax2 = ax.twiny()
        ax2.plot(phi_degs, y_p, color=color_p, lw=1.5)
        ax2.set_xlim(phi_degs[0], phi_degs[-1])
        ax2.set_xticks([0, 30, 60, 90, 120])
        ax2.set_xlabel(r"$\phi$ (deg)")
        ax2.tick_params(axis="both", labelsize=7)
        if not show_yticks:
            ax2.set_yticks([])

        if ax is ax_Delta:
            delta_legend = [
                Line2D([0], [0], color=color_p, lw=1.5,
                       label=r"$V = 0.4$ meV"),
                Line2D([0], [0], color=color_v, lw=1.5,
                       label=r"$\phi = 180^\circ$"),
            ]
            ax.legend(handles=delta_legend, loc="upper left", frameon=False,
                      fontsize=8)

        y_all = np.concatenate([np.atleast_1d(y_v), np.atleast_1d(y_p)])
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-12)
            if y_max > 0:
                ax.set_ylim(y_min, y_max * 1.5)
            else:
                ax.set_ylim(y_min * 1.5 if y_min < 0 else y_min, y_max)
        else:
            pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.1
            ax.set_ylim(y_min - pad, y_max + pad)

        if ax is ax_lam or ax is ax_rho:
            lim_max = max(abs(y_min), abs(y_max))
            if lim_max > 0:
                exponent = int(np.floor(np.log10(lim_max)))
                if exponent != 0:
                    ax.text(0.02, 0.95, rf"$\times 10^{{{exponent}}}$",
                            transform=ax.transAxes,
                            fontsize=7, ha="left", va="top")

    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12)

    out = Path(__file__).with_name("figures") / "fig_2D_theory.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
