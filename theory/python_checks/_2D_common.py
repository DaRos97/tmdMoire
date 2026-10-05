"""Shared observables (Δ, χ, λ, ρ) computed from band eigenvalues/eigenvectors.

Extracted from 2D_analysis.py so they can be reused by 2D_figure.py for
annotating the main band panel with the same observables.
"""
import numpy as np


def compute_gap(k_vals, evals, evecs, i_zero):
    """Return (delta, k_star, active, delta_top2, delta_2_3, ...,
    chi, weights_zero, band_2nd_weight, e_2nd_weight).

    Gap definition:
        k*      = argmin_k (E_{n-1}(k) - E_{n-3}(k))
        delta   = max(E_{n-1}(k*) - E_{n-2}(k*),
                      E_{n-2}(k*) - E_{n-3}(k*))
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
    """Weight ratio lambda at k_vals[i_minus_2K].

    Returns (lam, main_band_idx, side_band_idx, e_main, e_side, weights_m2K).
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
    """rho ratio at k' < -2K_M. Returns (rho, side_k_idx, side_band_idx,
    side_weight, e_side_rho).
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