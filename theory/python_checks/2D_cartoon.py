"""2D moire cartoon: parabolic bands + nearest-neighbor moire coupling.

Builds the same kind of zone-folded Hamiltonian as the main code
(tmdmoire.bilayer.hamiltonian.MoireHamiltonian.build_supercell) but with
parabolic bands (using the same m, hbar as 1D_cartoon.py) and a single
moire coupling V*exp(i*phi) between nearest-neighbor mini-BZ cells.

Left column: bands along K'->Gamma->K for theta=0 (top) and theta=1 (bottom)
at V=0 (red lines) and V=0.4 meV (blue circles sized by central-cell weight).
Right column (top row): same as left top inset but for moire phases
phi = 0, 60, 120, 180 degrees over a small k-range around Gamma.

Total figure width matches 1D_figure.py (6.75 in).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.constants import M_LIST


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
m = 3.5 * m0
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def build_hamiltonian(k_vec, geo, n_shells, V, phi):
    """Build the (n_cells x n_cells) Hamiltonian at a given 2D k-point.

    Diagonal: parabolic E = -alpha * |k + G_c|^2 with alpha from 1D_cartoon.py
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
        )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-12, -8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    n_shells = 1
    n_k = 800
    V_off = 0.0
    V_on = 0.4
    thetas = [0.0, 1.0]
    phase_degs = [0, 60, 120, 180]

    panels = [compute_bands(theta, n_shells, n_k, 3, V_off, V_on, 0.0)
              for theta in thetas]

    phase_panels = []
    for phase_deg in phase_degs:
        phi = phase_deg * np.pi / 180.0
        phase_panels.append(
            compute_bands(0.0, n_shells, n_k, 0.5, V_off, V_on, phi)
        )

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig = plt.figure(figsize=(6.75, 6.75 / 2.0))
    gs_outer = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.05,
                                hspace=0.30)

    ax_left_top = fig.add_subplot(gs_outer[0, 0])
    ax_left_bot = fig.add_subplot(gs_outer[1, 0])

    plot_main_panel(ax_left_top, *panels[0])
    ax_left_top.set_ylabel("Energy")
    plot_main_panel(ax_left_bot, *panels[1])
    ax_left_bot.set_ylabel("Energy")

    gs_right = gs_outer[0, 1].subgridspec(1, 4, wspace=0.20)
    ax_right = [fig.add_subplot(gs_right[0, i]) for i in range(4)]

    for ax, panel, phase_deg in zip(ax_right, phase_panels, phase_degs):
        plot_inset(ax, *panel)
        ax.set_title(rf"$\phi = {phase_deg}^\circ$", fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.08)

    out = Path(__file__).with_name("figures") / "2D_cartoon.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
