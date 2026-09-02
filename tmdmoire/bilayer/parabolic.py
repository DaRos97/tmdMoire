"""Zone-folded parabolic band model for moire superlattices.

Computes parabolic dispersions E(k) = offset - |k + G|^2 / (2m*)
zone-folded into the moire mini-Brillouin zone.
"""
import numpy as np
from ..bilayer.geometry import MoireGeometry

HBAR2_2M = 3.81  # eV * A^2, effective mass parameter


def compute_parabolic_bands(k_vals, n_shells, geometry, band_offset=0.0):
    """Compute zone-folded parabolic bands along a k-path.

    One parabolic band per mini-BZ cell: E_c(k) = offset - |k + G_c|^2 / HBAR2_2M

    Parameters
    ----------
    k_vals : np.ndarray
        1D array of k-point magnitudes along the path (A^-1).
    n_shells : int
        Number of moire shells for zone-folding.
    geometry : MoireGeometry
        Moire lattice geometry (reciprocal vectors, lu_table).
    band_offset : float, optional
        Energy offset for all bands (eV). Defaults to 0.

    Returns
    -------
    k_vals_out : np.ndarray
        Input k-values (unchanged).
    evals_out : np.ndarray
        Eigenvalues sorted ascending, shape (n_k, n_cells).
    K_mag : float
        Magnitude of the moire K-point vector.
    """
    G_M = geometry.reciprocal_vectors()
    G1, G2 = G_M[1], G_M[2]
    K_mag = np.linalg.norm((G1 + G2) / 3)

    lu = MoireGeometry.lu_table(n_shells)
    n_cells = len(lu)

    # Precompute G vectors for each cell
    G_cells = np.array([lu[c][0] * G1 + lu[c][1] * G2 for c in range(n_cells)])

    k_list = np.column_stack([k_vals, np.zeros_like(k_vals)])
    n_k = len(k_vals)

    # Build diagonal elements: E[c, k] = offset - |k + G_c|^2 / HBAR2_2M
    evals_out = np.zeros((n_k, n_cells))
    for c in range(n_cells):
        Gc = G_cells[c]
        k_sq = np.sum((k_list + Gc) ** 2, axis=1)
        evals_out[:, c] = band_offset - k_sq / HBAR2_2M

    # Sort eigenvalues at each k-point, track which cell each came from
    sort_idx = np.argsort(evals_out, axis=1)
    evals_sorted = np.take_along_axis(evals_out, sort_idx, axis=1)

    return k_vals, evals_sorted, K_mag, sort_idx
