"""Intensity computation for moire band plots.

Computes orbital weights from eigenvectors and spreads them in k and energy
using Gaussian or Lorentzian kernels to produce ARPES-like intensity maps.
"""
import numpy as np


def compute_weights(evecs, n_cells, pow_factor, shade_factor_ws2):
    """Compute central-cell orbital weights from eigenvectors.

    Parameters
    ----------
    evecs : np.ndarray
        Eigenvectors of shape (k_pts, dim, n_bands) where dim = 44*n_cells.
        Should already be sliced to the relevant band window.
    n_cells : int
        Number of mini-BZ cells.
    pow_factor : float
        Exponent applied to absolute eigenvector values.
    shade_factor_ws2 : float
        Weight multiplier for WS2 orbitals (0 = invisible, 1 = same as WSe2).

    Returns
    -------
    np.ndarray
        Central-cell weights of shape (k_pts, n_bands).
    """
    k_pts = evecs.shape[0]
    n_bands = evecs.shape[2]

    ab = np.absolute(evecs) ** pow_factor

    wse2_central = np.sum(ab[:, :22, :], axis=1)
    ws2_central = np.sum(ab[:, 22 * n_cells : 22 * n_cells + 22, :], axis=1)

    return wse2_central + shade_factor_ws2 * ws2_central


def spread_intensity(weights, k_list, evals, e_list, spread_k, spread_e, spread_type):
    """Spread weights in k and energy using Gaussian or Lorentzian kernels.

    Parameters
    ----------
    weights : np.ndarray
        Weights of shape (k_pts, n_bands).
    k_list : np.ndarray
        k-points of shape (k_pts, 2).
    evals : np.ndarray
        Eigenvalues of shape (k_pts, n_bands) in eV.
    e_list : np.ndarray
        Energy grid of shape (n_e,).
    spread_k : float
        Spreading width in k-space (A^-1).
    spread_e : float
        Spreading width in energy (eV).
    spread_type : str
        'Gauss' or 'Lorentz'.

    Returns
    -------
    np.ndarray
        Intensity map of shape (k_pts, n_e).
    """
    k_pts = k_list.shape[0]
    n_bands = weights.shape[1]
    n_e = len(e_list)

    k_dist = np.linalg.norm(
        k_list[:, np.newaxis, :] - k_list[np.newaxis, :, :],
        axis=2
    )

    spread = np.zeros((k_pts, n_e))

    if spread_type == 'Lorentz':
        k2 = spread_k ** 2
        e2 = spread_e ** 2
        for n in range(n_bands):
            e_diff = e_list[None, :] - evals[:, n, None]
            k_kernel = 1.0 / (k_dist ** 2 + k2)
            e_kernel = 1.0 / (e_diff ** 2 + e2)
            spread += weights[:, n, None] * (k_kernel @ e_kernel)
    elif spread_type == 'Gauss':
        for n in range(n_bands):
            e_diff = e_list[None, :] - evals[:, n, None]
            k_kernel = np.exp(-(k_dist / spread_k) ** 2)
            e_kernel = np.exp(-(e_diff / spread_e) ** 2)
            spread += weights[:, n, None] * (k_kernel @ e_kernel)
    else:
        raise ValueError(f"Unknown spread type: {spread_type}")

    return spread
