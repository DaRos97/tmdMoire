"""2D moire figure: combined cartoon + phase sweep + V/phi sweeps of observables.

Left column (width 0.8): bands along K'->Gamma->K for theta = theta_data
(top) and theta = 1 deg (bottom) at V = 0 (red lines) and V = 2 meV
(blue circles sized by central-cell weight). theta_data and a_moire_data
are read from data/2D_analysis.npz so the cartoon matches the analysis.

Right column, top (width 1.2 split into 4): same as left top but for
moire phases phi = 0, 20, 40, 60 deg over a small k-range around Gamma.

Right column, bottom (split into 4): 4 observables loaded from
data/2D_analysis.npz, with twin x-axes:
  - bottom: V (meV) for the V-sweep curve (phi = 180 deg)
  - top:    phi (deg) for the phi-sweep curve (V = 2 meV)

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
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmdmoire.bilayer.geometry import MoireGeometryFixed
from tmdmoire.constants import M_LIST

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _2D_common import compute_gap, compute_lambda, compute_rho


hbar_si = 1.054571817e-34
m0 = 9.1093837e-31
m = 1.19 * m0
eV = 1.602176634e-19

alpha = (hbar_si ** 2 / (2.0 * m)) / (eV * 1e-3) * 1e20


def build_hamiltonian(k_vec, geo, n_shells, V, phi):
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    lu = MoireGeometryFixed.lu_table(n_shells)
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
    ax.set_ylim(-80, 5)
    ax.tick_params(axis="both", labelsize=9)


def plot_main_panel(ax, k_vals, K_mag, n_cells, evals_off, evals_on, evecs_on,
                    max_size=60.0):
    for band in range(n_cells):
        ax.plot(k_vals / K_mag, evals_off[:, band], color="red", lw=0.1, zorder=5)

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
        Line2D([0], [0], color="red", lw=0.8, label=r"$V = 0$ meV"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
               markersize=5, label=r"$V = 2$ meV"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", frameon=True,
              fontsize=9, fancybox=True, shadow=False, edgecolor="black",
              facecolor="white", framealpha=1.0)

    zoom_box = Rectangle((-0.3, -75), 0.6, 15, fill=False,
                         edgecolor="black", linewidth=1.2, zorder=10)
    ax.add_patch(zoom_box)

    decorate_main_axis(ax)


def plot_inset(ax, k_vals, K_mag, n_cells, evals_off, evals_on, evecs_on,
               max_size=60.0, marker_boost=5.0):
    for band in range(n_cells):
        ax.plot(k_vals / K_mag, evals_off[:, band], color="red", lw=0.1, zorder=5)

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
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-75, -60)
    ax.axvline(0.0, color="gray", lw=0.5, ls="--", zorder=5)
    ax.tick_params(axis="x", which="both", direction="out", length=3,
                   bottom=True, top=False, labelbottom=False)
    ax.set_xticks([0])
    ax.text(0.0, -0.05, r"$\Gamma$", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9)
    ax.set_yticks([])


def annotate_main_panel(ax, k_vals, K_mag, n_cells, evals, evecs):
    """Add Δ, χ, λ, ρ arrows and labels to the main band panel, as in
    1D_figure.py subplot a."""
    i_zero = int(np.argmin(np.abs(k_vals)))
    k_lambda = -1.5 * K_mag
    i_lambda = int(np.argmin(np.abs(k_vals - k_lambda)))

    (delta, k_star, active, _, _, _,
     e_top, e_second, e_third,
     chi, _, band_2nd_weight, _) = \
        compute_gap(k_vals, evals, evecs, i_zero)

    if active == "top2":
        e_lo, e_hi = e_second, e_top
    else:
        e_lo, e_hi = e_third, e_second
    ax.annotate(
        "",
        xy=(k_star / K_mag, e_hi),
        xytext=(k_star / K_mag, e_lo),
        arrowprops=dict(arrowstyle="<->", color="black", lw=0.5,
                        mutation_scale=5, shrinkA=0, shrinkB=0),
        zorder=7,
    )
    ax.text(
        k_star / K_mag + 0.10, 0.5 * (e_hi + e_lo), r"$\Delta$",
        color="black", fontsize=10, va="center", ha="left",
    )

    e_top_0 = float(evals[i_zero, -1])
    e_2nd_w = float(evals[i_zero, band_2nd_weight])
    ax.annotate(
        "",
        xy=(0.0, e_top_0),
        xytext=(0.0, e_2nd_w),
        arrowprops=dict(arrowstyle="<->", color="black", lw=0.7,
                        mutation_scale=12, shrinkA=0, shrinkB=0),
        zorder=7,
    )
    ax.text(
        0.08, 0.5 * (e_top_0 + e_2nd_w), r"$\chi$",
        color="black", fontsize=10, va="center",
    )

    lam, main_band_idx, side_band_idx, e_main, e_side, _ = \
        compute_lambda(evals, evecs, i_lambda)
    k_mark = k_vals[i_lambda] / K_mag
    ax.scatter([k_mark], [e_main], s=40, c="black", edgecolors="black",
               linewidths=0.5, zorder=6)
    if side_band_idx >= 0 and not np.isnan(e_side):
        ax.scatter([k_mark], [e_side], s=40, c="black", edgecolors="black",
                   linewidths=0.5, zorder=6)
        ax.plot([k_mark, k_mark], [e_main, e_side], color="black",
                lw=0.8, ls="--", zorder=5)
        ax.text(k_mark - 0.08, 0.5 * (e_main + e_side), r"$\lambda$",
                color="black", fontsize=10, va="center", ha="right")
    else:
        ax.text(k_mark - 0.08, e_main + 3, r"$\lambda$",
                color="black", fontsize=10, ha="right")

    rho, side_k_idx, side_band_idx_rho, _, e_side_rho = \
        compute_rho(evals, evecs, k_vals, i_lambda, main_band_idx)
    if side_k_idx >= 0 and not np.isnan(e_side_rho):
        k_rho = k_vals[side_k_idx] / K_mag
        ax.scatter([k_rho], [e_side_rho], s=40, c="black",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.plot([k_mark, k_rho], [e_main, e_main], color="black",
                lw=0.8, ls="--", zorder=5)
        k_mid = 0.5 * (k_mark + k_rho)
        ax.text(k_mid, e_main + 3, r"$\rho$",
                color="black", fontsize=10, ha="center")


def main():
    n_shells = 1
    n_k = 1000
    V_off = 0.0
    V_on = 2.0
    theta_bottom = 1.0
    phase_degs = [0, 20, 40, 60]

    sweeps_path = Path(__file__).with_name("data") / "2D_analysis.npz"
    if sweeps_path.exists():
        sw = np.load(sweeps_path)
        Vs = sw["V_V"]
        phi_degs = sw["phi_deg_phi"]
        Delta_v, chi_v = sw["delta_V"], sw["chi_V"]
        lambda_v, rho_v = sw["lambda_V"], sw["rho_V"]
        Delta_p, chi_p = sw["delta_phi"], sw["chi_phi"]
        lambda_p, rho_p = sw["lambda_phi"], sw["rho_phi"]
        theta = float(sw["theta"]) if "theta" in sw.files else 0.0
        a_moire_target = (
            float(sw["a_moire"]) if "a_moire" in sw.files else 50.0
        )
        print(
            f"Loaded sweeps from {sweeps_path}: theta = {theta} deg, "
            f"a_moire = {a_moire_target} A"
        )
    else:
        print(f"WARNING: {sweeps_path} not found; using defaults.")
        theta = 0.0
        a_moire_target = 50.0
        Vs = np.array([0.0, 1.0])
        phi_degs = np.array([0.0, 120.0])
        Delta_v = chi_v = lambda_v = rho_v = np.zeros(2)
        Delta_p = chi_p = lambda_p = rho_p = np.zeros(2)

    thetas = [theta, theta_bottom]

    panels = [compute_bands(th, n_shells, n_k, 3, V_off, V_on, 0.0,
                            a_moire_override=a_moire_target)
              for th in thetas]

    geo_a = MoireGeometryFixed(theta, a_moire_override=a_moire_target)
    g_mag = float(np.linalg.norm(geo_a.reciprocal_vectors()[1]))
    A = alpha * g_mag ** 2 / 4.0
    A2 = A * A
    print(f"A = alpha*|G_M|²/4 = {A:.6f} meV  (|G_M| = {g_mag:.6f} 1/Å, "
          f"a_moiré = {geo_a.moire_length:.4f} Å)")

    phase_panels = []
    for phase_deg in phase_degs:
        phi = phase_deg * np.pi / 180.0
        phase_panels.append(
            compute_bands(theta, n_shells, n_k, 0.5, V_off, V_on, phi,
                          a_moire_override=a_moire_target)
        )

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig = plt.figure(figsize=(6.75, 4.5))
    gs_outer = fig.add_gridspec(2, 2, width_ratios=[0.7, 1.3], wspace=0.05,
                                hspace=0.45, height_ratios=[2.0, 1.0])

    ax_left_top = fig.add_subplot(gs_outer[0, 0])

    plot_main_panel(ax_left_top, *panels[0], max_size=5.0)
    k_vals_a, K_mag_a, n_cells_a, evals_off_a, evals_on_a, evecs_on_a = panels[0]
    annotate_main_panel(ax_left_top, k_vals_a, K_mag_a, n_cells_a,
                        evals_on_a, evecs_on_a)
    ax_left_top.set_ylim(-110, 10)
    ax_left_top.set_yticks(np.arange(-100, 11, 25))
    ax_left_top.tick_params(axis="y", labelsize=7)
    ax_left_top.set_ylabel("Energy [meV]", fontsize=9, labelpad=2)
    ax_left_top.text(-0.15, 1.05, r"$\mathbf{a.}$",
                     transform=ax_left_top.transAxes,
                     fontsize=9, fontweight="bold", va="bottom", ha="left")

    gs_right_top = gs_outer[0, 1].subgridspec(1, 4, wspace=0.20)
    ax_right_top = [fig.add_subplot(gs_right_top[0, i]) for i in range(4)]
    for ax, panel, phase_deg in zip(ax_right_top, phase_panels, phase_degs):
        plot_inset(ax, *panel, marker_boost=20.0)
        ax.set_title(rf"$\phi = {phase_deg}^\circ$", fontsize=9)
    ax_right_top[0].text(-0.15, 1.05, r"$\mathbf{b.}$",
                          transform=ax_right_top[0].transAxes,
                          fontsize=9, fontweight="bold", va="bottom", ha="left")

    gs_bottom = gs_outer[1, :].subgridspec(1, 4, wspace=0.40)
    ax_right_bot = [fig.add_subplot(gs_bottom[0, i]) for i in range(4)]
    ax_Delta, ax_chi, ax_lam, ax_rho = ax_right_bot

    label_box = dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                     edgecolor="black", linewidth=0.8)
    ax_Delta.text(0.85, 0.45, r"$\Delta$", transform=ax_Delta.transAxes,
                  ha="center", va="bottom", fontsize=12, bbox=label_box)
    ax_chi.text(0.5, 0.90, r"$\chi$", transform=ax_chi.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)
    ax_lam.text(0.5, 0.90, r"$\lambda$", transform=ax_lam.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)
    ax_rho.text(0.5, 0.90, r"$\rho$", transform=ax_rho.transAxes,
                ha="center", va="top", fontsize=12, bbox=label_box)

    ax_Delta.set_ylabel("[meV]", fontsize=9, labelpad=2)
    ax_chi.set_ylabel("[meV]", fontsize=9, labelpad=2)
    ax_lam.set_ylabel("%", fontsize=9, labelpad=2)
    ax_rho.set_ylabel("%", fontsize=9, labelpad=2)

    subplot_labels = [(ax_Delta, "c."), (ax_chi, "d."),
                      (ax_lam, "e."), (ax_rho, "f.")]
    for ax, lab in subplot_labels:
        ax.text(-0.18, 1.15, rf"$\mathbf{{{lab}}}$",
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")

    color_v = "#0072B2"
    color_p = "#E69F00"

    lambda_v_s = lambda_v * 100.0
    lambda_p_s = lambda_p * 100.0

    quantities = [
        (ax_Delta, Delta_v, Delta_p, "linear", True),
        (ax_chi, chi_v, chi_p, "linear", True),
        (ax_lam, lambda_v_s, lambda_p_s, "linear", True),
        (ax_rho, rho_v, rho_p, "linear", True),
    ]

    for ax, y_v, y_p, yscale, show_yticks in quantities:
        ax.plot(Vs, y_v, color=color_v, lw=1.5)
        ax.set_xlim(Vs[0], Vs[-1])
        ax.set_xlabel("V [meV]", fontsize=9)
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
        ax2.set_xlabel(r"$\phi$ [deg]", fontsize=9)
        ax2.tick_params(axis="both", labelsize=7)
        if not show_yticks:
            ax2.set_yticks([])

        if ax is ax_Delta:
            delta_legend = [
                Line2D([0], [0], color=color_p, lw=1.5,
                       label=r"$V = 2$ meV"),
                Line2D([0], [0], color=color_v, lw=1.5,
                       label=r"$\phi = 180^\circ$"),
            ]
            ax.legend(handles=delta_legend, loc="upper left",
                      bbox_to_anchor=(-0.02, 1.02), frameon=False,
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

        if ax is ax_rho:
            ax.set_yticks([0, 0.005, 0.01])

        if ax is ax_Delta:
            coeffs = np.polyfit(Vs, y_v, 1)
            print(f"Δ(V) = {coeffs[0]:.4f} V + {coeffs[1]:.4f}")
        elif ax in (ax_chi, ax_lam, ax_rho):
            y0 = float(y_v[0])
            y_shifted = y_v - y0
            design = np.column_stack([np.ones_like(Vs), Vs ** 2])
            a_q, b_q = np.linalg.lstsq(design, y_shifted, rcond=None)[0]
            name = {ax_chi: "χ", ax_lam: "λ", ax_rho: "ρ"}[ax]
            A_sq_b = A2 * b_q
            print(
                f"{name}(V) - {name}(0) = {a_q:.6e} + {b_q:.6e} V²"
                f"   | A²·b = {A_sq_b:.6e}"
            )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12)

    out = Path(__file__).with_name("figures") / "fig_2D_theory.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
