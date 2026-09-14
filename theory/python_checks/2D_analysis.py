"""2D moire analysis: parameter sweeps with band-weight plots and gap analysis.

Sweeps a single-band moire Hamiltonian (parabolic dispersion + V*exp(i*phi)
nearest-neighbor coupling).

Two sweeps:
  (1) V = 0.2 meV, phi in [0, 120] deg
  (2) phi = 180 deg, V in [0, 1] meV

For each (V, phi):
  - 1000 k-points are sampled from -5 K_M to 0
  - gap is computed as: k* = argmin_k (E_{n-1}(k) - E_{n-3}(k)), then
    delta = max(E_{n-1}(k*) - E_{n-2}(k*), E_{n-2}(k*) - E_{n-3}(k*))
  - chi at k=0 is the distance between the top band and the band with
    the second-highest central-cell weight |psi_{cell=0}|^2
  - lambda at k=-2 K_M: ratio of the highest-weight band with energy
    larger than the main (highest-weight) band, divided by the main
    band's weight
  - rho: at k=-2 K_M the main band has energy E_main. For each of the two
    highest-energy bands, find the k' < -2K_M where that band's energy is
    closest to E_main; rho is then the ratio of the higher-weight candidate
    to the main band's weight.
  - a band-weight plot is saved to figures/temp/ with marker size proportional
    to the central-cell weight |psi_k(Gamma)|^2 of each eigenstate. An inset
    in the top-left zooms around k*; the two energies that define the
    active sub-gap (either E_{n-1}(k*) & E_{n-2}(k*), or E_{n-2}(k*) &
    E_{n-3}(k*)) are highlighted in red inside the inset only. A side panel
    lists all bands at k=0 (idx, E, |w|^2) with the two bands used for chi
    shown in red. Green markers on the main plot show the main and side
    bands at k=-2 K_M used for lambda; an orange marker shows the side
    band used for rho at k' < -2 K_M.

A summary figure with 4 subplots is also saved to figures/2D_analysis.pdf:
  Row 1: Delta = max(Delta_top2, Delta_2_3) at k*    (twin V / phi axes)
  Row 2: chi at k=0                                  (twin V / phi axes)
  Row 3: lambda at k=-2 K_M                          (twin V / phi axes)
  Row 4: rho at k' < -2 K_M (side band at same E as main)  (twin V/phi axes)
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
    """Build the n_cells x n_cells zone-folded Hamiltonian at k_vec.

    Diagonal: parabolic E = -alpha * |k + G_c|^2.
    Off-diagonal: V*exp(+i*phi) for even-index M_LIST neighbors,
                  V*exp(-i*phi) for odd-index M_LIST neighbors.
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


def compute_bands(geo, n_shells, k_vals, V, phi):
    """Diagonalize H(k) along the k-path; return (evals, evecs)."""
    n_cells = MoireGeometry.n_cells(n_shells)
    n_k = len(k_vals)
    evals = np.empty((n_k, n_cells))
    evecs = np.empty((n_k, n_cells, n_cells), dtype=complex)
    for i_k, k in enumerate(k_vals):
        k_vec = np.array([k, 0.0])
        H = build_hamiltonian(k_vec, geo, n_shells, V, phi)
        w, v = np.linalg.eigh(H)
        evals[i_k] = w
        evecs[i_k] = v
    return evals, evecs


def compute_gap(k_vals, evals, evecs, i_zero):
    """Compute gap statistics and chi from band eigenvalues/eigenvectors.

    Gap definition:
        k*      = argmin_k (E_{n-1}(k) - E_{n-3}(k))
        delta   = max(E_{n-1}(k*) - E_{n-2}(k*),
                      E_{n-2}(k*) - E_{n-3}(k*))
        active  = "top2" if the top-second sub-gap is the max,
                  "2_3"   if the second-third sub-gap is the max.

    Chi definition (at k = k_vals[i_zero], typically k=0):
        weights      = |psi_{cell=0, band}|^2 at k=k_vals[i_zero]
        sorted_bands = argsort(weights) in descending order
        band_2nd     = sorted_bands[1]              # 2nd highest weight
        chi          = E_{n-1}(k0) - E_{band_2nd}(k0)

    Returns
    -------
    delta, k_star, active, delta_top2, delta_2_3, bandwidth,
    e_top, e_second, e_third,
    chi, weights_zero, band_2nd_weight, e_2nd_weight
    """
    gap_main = evals[:, -1] - evals[:, -3]
    i_star = int(np.argmin(gap_main))
    k_star = float(k_vals[i_star])
    e_top = float(evals[i_star, -1])
    e_second = float(evals[i_star, -2])
    e_third = float(evals[i_star, -3])

    gap_top2_at_kstar = e_top - e_second
    gap_2_3_at_kstar = e_second - e_third
    delta = float(max(gap_top2_at_kstar, gap_2_3_at_kstar))
    active = "top2" if gap_top2_at_kstar >= gap_2_3_at_kstar else "2_3"

    gap_top2 = evals[:, -1] - evals[:, -2]
    delta_top2 = float(np.min(gap_top2))

    gap_2_3 = evals[:, -2] - evals[:, -3]
    delta_2_3 = float(np.min(gap_2_3))

    bandwidth = float(evals.max() - evals.min())

    weights_zero = np.abs(evecs[i_zero, 0, :]) ** 2
    sorted_by_weight = np.argsort(weights_zero)[::-1]
    band_2nd_weight = int(sorted_by_weight[1])
    e_top_zero = float(evals[i_zero, -1])
    e_2nd_weight = float(evals[i_zero, band_2nd_weight])
    chi = e_top_zero - e_2nd_weight

    return (delta, k_star, active, delta_top2, delta_2_3, bandwidth,
            e_top, e_second, e_third,
            chi, weights_zero, band_2nd_weight, e_2nd_weight)


def compute_lambda(evals, evecs, i_minus_2K):
    """At k = k_vals[i_minus_2K], compute the weight ratio lambda.

    main  = band with the highest central-cell weight
    side  = band with the highest central-cell weight among those with
            E > E_main (i.e. above the main band in energy)
    lambda = |psi_side|^2 / |psi_main|^2

    Returns
    -------
    lam : float
        weight ratio (0.0 if no side band exists)
    main_band_idx, side_band_idx : int
        indices into the band array (side_band_idx = -1 if no side band)
    e_main, e_side : float
        energies of the main and side bands (NaN for e_side if no side band)
    weights_m2K : ndarray
        full weight array at k=-2K (for diagnostic use)
    """
    weights_m2K = np.abs(evecs[i_minus_2K, 0, :]) ** 2
    main_band_idx = int(np.argmax(weights_m2K))
    e_main = float(evals[i_minus_2K, main_band_idx])
    w_main = float(weights_m2K[main_band_idx])

    higher_E_mask = evals[i_minus_2K, :] > e_main
    if higher_E_mask.any():
        weights_higher = weights_m2K.copy()
        weights_higher[~higher_E_mask] = -1.0
        side_band_idx = int(np.argmax(weights_higher))
        e_side = float(evals[i_minus_2K, side_band_idx])
        w_side = float(weights_m2K[side_band_idx])
        lam = w_side / w_main if w_main > 0 else 0.0
    else:
        side_band_idx = -1
        e_side = float("nan")
        lam = 0.0

    return lam, main_band_idx, side_band_idx, e_main, e_side, weights_m2K


def compute_rho(evals, evecs, k_vals, i_minus_2K, main_band_idx):
    """Compute the rho ratio from the two highest-energy bands at k' < -2K_M.

    Procedure:
      1. E_main = main band energy at k = k_vals[i_minus_2K].
      2. For each of the two highest-energy bands (indices n_cells-1 and
         n_cells-2 in the ascending-sorted band array), find the k'< -2K_M
         where that band's energy is closest to E_main.
      3. From the two resulting (k', band) candidates, pick the one with the
         highest central-cell weight.
      4. rho = weight of the chosen side band / weight of the main band.

    Returns
    -------
    rho : float
        weight ratio (0.0 if no candidate found)
    side_k_idx : int
        index in k_vals where the rho side band lives
    side_band_idx : int
        band index of the rho side band at side_k_idx
    side_weight : float
        central-cell weight of the rho side band
    e_side_rho : float
        energy of the rho side band (close to E_main)
    """
    E_main = float(evals[i_minus_2K, main_band_idx])
    w_main = float(np.abs(evecs[i_minus_2K, 0, main_band_idx]) ** 2)

    if w_main <= 0 or i_minus_2K <= 0:
        return 0.0, -1, -1, 0.0, float("nan")

    n_cells = evals.shape[1]
    tracked_bands = [n_cells - 1, n_cells - 2]

    candidates = []
    for band_idx in tracked_bands:
        diffs = np.abs(evals[:i_minus_2K, band_idx] - E_main)
        i_k_min = int(np.argmin(diffs))
        w = float(np.abs(evecs[i_k_min, 0, band_idx]) ** 2)
        e_band_at_kmin = float(evals[i_k_min, band_idx])
        candidates.append((i_k_min, band_idx, w, e_band_at_kmin))

    best_k_idx, best_band_idx, side_weight, e_side_rho = max(
        candidates, key=lambda c: c[2]
    )
    rho = side_weight / w_main if w_main > 0 else 0.0
    return rho, best_k_idx, best_band_idx, side_weight, e_side_rho


def decorate_axis(ax, k_vals, K_mag):
    """Add k-axis ticks/labels for the [-5 K_M, 0] range with Gamma highlighted."""
    k_min = k_vals[0] / K_mag
    k_max = k_vals[-1] / K_mag
    k_int_lo = int(np.ceil(k_min))
    k_int_hi = int(np.floor(k_max))
    positions = np.arange(k_int_lo, k_int_hi + 1)
    for k in positions:
        ax.axvline(k, color="gray", lw=0.5, ls="--")
    ax.set_xticks(positions)
    if k_min <= 0 <= k_max:
        gamma_idx = int(np.argmin(np.abs(positions)))
        labels = [r"$\Gamma$"] * len(positions)
        for i, p in enumerate(positions):
            if i == gamma_idx:
                continue
            dist_from_gamma = abs(p - positions[gamma_idx])
            if dist_from_gamma % 3 == 0:
                labels[i] = r"$\Gamma$"
            elif (dist_from_gamma % 3 == 1 and p < positions[gamma_idx]) or \
                 (dist_from_gamma % 3 == 2 and p > positions[gamma_idx]):
                labels[i] = r"$K$"
            else:
                labels[i] = r"$K'$"
        ax.set_xticklabels(labels)
    ax.set_xlim(k_min, k_max)
    ax.set_ylabel("Energy (meV)")
    ax.set_ylim(-30, 5)


def save_band_weight_plot(out_path, k_vals, K_mag, evals, evecs, V, phi_deg,
                          delta, k_star, active,
                          e_top, e_second, e_third,
                          chi, weights_zero, band_2nd_weight, e_2nd_weight,
                          i_zero, lam, main_band_idx, side_band_idx,
                          e_main_m2K, e_side_m2K, i_minus_2K,
                          rho, side_k_idx_rho, side_band_idx_rho,
                          e_side_rho, max_size=60.0):
    """Save a band-weight plot; inset highlights gap points, side lists k=0 bands."""
    import matplotlib.pyplot as plt

    n_k, n_cells = evals.shape
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    central_weight = np.abs(evecs[:, 0, :]) ** 2
    for band in range(n_cells):
        ax.scatter(
            k_vals / K_mag,
            evals[:, band],
            s=central_weight[:, band] * max_size,
            c="C0",
            linewidths=0,
            zorder=3,
        )

    k_m2K = k_vals[i_minus_2K] / K_mag
    ax.axvline(k_m2K, color="green", lw=0.7, ls="--", alpha=0.5, zorder=4)
    ax.scatter(
        [k_m2K], [e_main_m2K],
        s=130, c="limegreen", zorder=11, marker="o",
        edgecolors="darkgreen", linewidths=1.5,
    )
    if side_band_idx >= 0 and not np.isnan(e_side_m2K):
        ax.scatter(
            [k_m2K], [e_side_m2K],
            s=130, c="limegreen", zorder=11, marker="s",
            edgecolors="darkgreen", linewidths=1.5,
        )

    if side_k_idx_rho >= 0 and side_band_idx_rho >= 0 and not np.isnan(e_side_rho):
        k_rho = k_vals[side_k_idx_rho] / K_mag
        ax.scatter(
            [k_rho], [e_side_rho],
            s=130, c="orange", zorder=11, marker="D",
            edgecolors="darkorange", linewidths=1.5,
        )

    decorate_axis(ax, k_vals, K_mag)
    ax.set_title(
        rf"$V = {V:.2f}$ meV, $\phi = {phi_deg:.0f}^\circ$, "
        rf"$\Delta = {delta:.3f}$ meV, $k^\star = {k_star/K_mag:.2f}\,K_M$, "
        rf"$\chi = {chi:.3f}$ meV, $\lambda = {lam:.3f}$, $\rho = {rho:.3f}$",
        fontsize=9,
    )
    ax.set_xlabel(r"$k / K_M$")

    inset_ax = ax.inset_axes([0.05, 0.55, 0.4, 0.4])
    k_half = 0.3
    mask = np.abs(k_vals / K_mag - k_star / K_mag) <= k_half
    for band in range(n_cells):
        inset_ax.scatter(
            k_vals[mask] / K_mag,
            evals[mask, band],
            s=central_weight[mask, band] * max_size,
            c="C0",
            linewidths=0,
            zorder=3,
        )
    if active == "top2":
        inset_ax.scatter(
            [k_star / K_mag], [e_top],
            s=120, c="red", zorder=10, marker="o",
            edgecolors="darkred", linewidths=1.5,
        )
        inset_ax.scatter(
            [k_star / K_mag], [e_second],
            s=120, c="red", zorder=10, marker="^",
            edgecolors="darkred", linewidths=1.5,
        )
    else:
        inset_ax.scatter(
            [k_star / K_mag], [e_second],
            s=120, c="red", zorder=10, marker="^",
            edgecolors="darkred", linewidths=1.5,
        )
        inset_ax.scatter(
            [k_star / K_mag], [e_third],
            s=120, c="red", zorder=10, marker="s",
            edgecolors="darkred", linewidths=1.5,
        )
    inset_ax.axvline(k_star / K_mag, color="red", lw=0.8, ls="--", alpha=0.6)
    inset_ax.axvline(-1.0, color="black", lw=1.2, ls="-", alpha=0.85)
    if active == "top2":
        e_center = 0.5 * (e_top + e_second)
    else:
        e_center = 0.5 * (e_second + e_third)
    e_half = max(1.5, 3.0 * abs(delta))
    inset_ax.set_xlim(k_star / K_mag - k_half, k_star / K_mag + k_half)
    inset_ax.set_ylim(e_center - e_half, e_center + e_half)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    y_top = inset_ax.get_ylim()[1]
    y_lo = inset_ax.get_ylim()[0]
    inset_ax.text(
        -1.0, y_top - 0.04 * (y_top - y_lo), r"$-K$",
        ha="center", va="top", fontsize=8, color="black", fontweight="bold",
    )

    text_x = 1.02
    y0 = 0.98
    dy = 0.026
    ax.text(
        text_x, y0, r"Bands at $k=0$ (sorted by $E$):",
        transform=ax.transAxes, fontsize=7, fontweight="bold",
        va="top", family="monospace",
    )
    ax.text(
        text_x, y0 - dy, r"idx   $E$ (meV)   $|w|^2$",
        transform=ax.transAxes, fontsize=6, va="top", family="monospace",
    )
    sorted_by_E = np.argsort(evals[i_zero, :])[::-1]
    for j, band_idx in enumerate(sorted_by_E):
        e = evals[i_zero, band_idx]
        w = weights_zero[band_idx]
        is_chi = (band_idx == n_cells - 1) or (band_idx == band_2nd_weight)
        color = "red" if is_chi else "black"
        line = f"{j:2d}:  {e:+6.2f}   {w:.3f}"
        ax.text(
            text_x, y0 - (j + 2) * dy, line,
            transform=ax.transAxes, fontsize=6, color=color,
            family="monospace", va="top",
        )
    ax.text(
        text_x, y0 - (n_cells + 2) * dy,
        rf"$\chi = E_{{n-1}} - E_{{{band_2nd_weight}}} = {chi:.3f}$ meV",
        transform=ax.transAxes, fontsize=6.5, color="red",
        fontweight="bold", family="monospace", va="top",
    )

    fig.subplots_adjust(left=0.08, right=0.78, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_sweep(out_dir, geo, n_shells, k_vals, K_mag, V_arr, phi_arr, label,
              save_plots=True):
    """Loop over (V, phi) pairs: collect gap, chi, and lambda data.

    If save_plots is False, skip per-point plot generation (summary only).
    """
    n_pts = len(V_arr)
    i_zero = int(np.argmin(np.abs(k_vals)))
    i_minus_2K = int(np.argmin(np.abs(k_vals - (-2.0 * K_mag))))
    data = []
    for i_pt in range(n_pts):
        V = float(V_arr[i_pt])
        phi = float(phi_arr[i_pt])
        phi_deg = phi * 180.0 / np.pi
        evals, evecs = compute_bands(geo, n_shells, k_vals, V, phi)
        (delta, k_star, active, delta_top2, delta_2_3, bw,
         e_top, e_second, e_third,
         chi, weights_zero, band_2nd_weight, e_2nd_weight) = \
            compute_gap(k_vals, evals, evecs, i_zero)
        lam, main_band_idx, side_band_idx, e_main_m2K, e_side_m2K, _ = \
            compute_lambda(evals, evecs, i_minus_2K)
        rho, side_k_idx_rho, side_band_idx_rho, _, e_side_rho = \
            compute_rho(evals, evecs, k_vals, i_minus_2K, main_band_idx)
        data.append({
            "V": V, "phi": phi, "phi_deg": phi_deg,
            "delta": delta, "k_star": k_star, "active": active,
            "delta_top2": delta_top2, "delta_2_3": delta_2_3,
            "bandwidth": bw,
            "e_top": e_top, "e_second": e_second, "e_third": e_third,
            "chi": chi,
            "weights_zero": weights_zero,
            "band_2nd_weight": band_2nd_weight,
            "e_2nd_weight": e_2nd_weight,
            "lambda": lam,
            "main_band_idx_m2K": main_band_idx,
            "side_band_idx_m2K": side_band_idx,
            "e_main_m2K": e_main_m2K,
            "e_side_m2K": e_side_m2K,
            "rho": rho,
            "side_k_idx_rho": side_k_idx_rho,
            "side_band_idx_rho": side_band_idx_rho,
            "e_side_rho": e_side_rho,
        })
        if save_plots:
            out = out_dir / f"band_{label}_{i_pt:02d}_V{V:.2f}_phi{phi_deg:.0f}.png"
            save_band_weight_plot(out, k_vals, K_mag, evals, evecs, V, phi_deg,
                                  delta, k_star, active,
                                  e_top, e_second, e_third,
                                  chi, weights_zero, band_2nd_weight,
                                  e_2nd_weight, i_zero,
                                  lam, main_band_idx, side_band_idx,
                                  e_main_m2K, e_side_m2K, i_minus_2K,
                                  rho, side_k_idx_rho, side_band_idx_rho,
                                  e_side_rho)
    return data


def _plot_twin(ax, Vs, y_V, phi_degs, y_phi, ylabel, color_v, color_p):
    """Helper: plot two curves on twin axes (V sweep on bottom, phi on top)."""
    line_v = ax.plot(Vs, y_V, color=color_v, lw=1.5,
                     label=r"$V$ sweep ($\phi=180^\circ$)")
    ax.set_xlim(Vs[0], Vs[-1])
    ax.set_xlabel("V (meV)")
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="k", lw=0.5, ls=":")

    ax2 = ax.twiny()
    line_p = ax2.plot(phi_degs, y_phi, color=color_p, lw=1.5,
                      label=r"$\phi$ sweep ($V=0.2$)")
    ax2.set_xlim(phi_degs[0], phi_degs[-1])
    ax2.set_xlabel(r"$\phi$ (deg)")

    y_all = np.concatenate([np.atleast_1d(y_V), np.atleast_1d(y_phi)])
    y_min = np.nanmin(y_all)
    y_max = np.nanmax(y_all)
    pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.1
    ax.set_ylim(y_min - pad, y_max + pad)

    handles = line_v + line_p
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="best", fontsize=8)


def make_summary_figure(out_path, data_phi, data_V, Vs, phi_degs):
    """Four-panel summary: Delta, chi, lambda, rho vs V and vs phi (twin axes)."""
    import matplotlib.pyplot as plt

    color_v = "C0"
    color_p = "C3"

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 13.0))

    _plot_twin(axes[0], Vs,
               np.array([d["delta"] for d in data_V]),
               phi_degs, np.array([d["delta"] for d in data_phi]),
               r"$\Delta$ (meV)", color_v, color_p)
    _plot_twin(axes[1], Vs,
               np.array([d["chi"] for d in data_V]),
               phi_degs, np.array([d["chi"] for d in data_phi]),
               r"$\chi$ (meV)", color_v, color_p)
    _plot_twin(axes[2], Vs,
               np.array([d["lambda"] for d in data_V]),
               phi_degs, np.array([d["lambda"] for d in data_phi]),
               r"$\lambda$ (side/main weight at $k=-2K_M$)", color_v, color_p)
    _plot_twin(axes[3], Vs,
               np.array([d["rho"] for d in data_V]),
               phi_degs, np.array([d["rho"] for d in data_phi]),
               r"$\rho$ at same $E$ as main, $k'<-2K_M$", color_v, color_p)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def save_data(out_path, data_phi, data_V):
    """Save both sweep data lists to a single .npz file.

    Each scalar field is saved twice, once per sweep, with a "_V" or "_phi"
    suffix. The "active" tag is saved as a unicode string array. The full
    weights_zero arrays at k=0 are saved as 2D arrays (n_pts, n_cells).
    """
    scalar_fields = [
        "V", "phi_deg", "delta", "k_star",
        "delta_top2", "delta_2_3", "bandwidth",
        "e_top", "e_second", "e_third", "chi",
        "lambda", "rho",
        "band_2nd_weight", "e_2nd_weight",
        "main_band_idx_m2K", "side_band_idx_m2K",
        "e_main_m2K", "e_side_m2K",
        "side_k_idx_rho", "side_band_idx_rho",
        "e_side_rho",
    ]
    save_dict = {}
    for d, suffix in [(data_V, "_V"), (data_phi, "_phi")]:
        for field in scalar_fields:
            save_dict[field + suffix] = np.array(
                [d_i[field] for d_i in d], dtype=float
            )
        save_dict["active" + suffix] = np.array(
            [d_i["active"] for d_i in d], dtype="U10"
        )
        save_dict["weights_zero" + suffix] = np.array(
            [d_i["weights_zero"] for d_i in d]
        )
    np.savez(out_path, **save_dict)
    print(f"Saved data to {out_path}")


def main():
    theta = 0.0
    n_shells = 1
    n_sweep_pts = 61
    n_k = 5000
    save_temp_plots = False

    geo = MoireGeometry(theta)
    G_M = geo.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    K_mag = np.linalg.norm((G1 + G2) / 3)

    k_vals = np.linspace(-5.0 * K_mag, 0.0, n_k)

    out_dir = Path(__file__).with_name("figures") / "temp"
    if save_temp_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    phi_degs = np.linspace(0.0, 120.0, n_sweep_pts)
    phis = phi_degs * np.pi / 180.0
    V_phi_sweep = np.full(n_sweep_pts, 0.4)
    data_phi = run_sweep(out_dir, geo, n_shells, k_vals, K_mag, V_phi_sweep,
                         phis, "phisweep_V0.4", save_plots=save_temp_plots)

    Vs = np.linspace(0.0, 1.0, n_sweep_pts)
    phi_V_sweep = np.full(n_sweep_pts, 180.0 * np.pi / 180.0)
    data_V = run_sweep(out_dir, geo, n_shells, k_vals, K_mag, Vs, phi_V_sweep,
                       "Vsweep_phi180", save_plots=save_temp_plots)

    summary_path = Path(__file__).with_name("figures") / "2D_analysis.pdf"
    make_summary_figure(summary_path, data_phi, data_V, Vs, phi_degs)

    data_path = Path(__file__).with_name("data") / "2D_analysis.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_data(data_path, data_phi, data_V)


if __name__ == "__main__":
    main()
