"""Real-space local density of states (LDOS) for moire supercells.

Computes the k-integrated spectral function at a 1D line of real-space
positions along the moire lattice direction a1 + a2:

    LDOS(r, E) = (1/N_k) * Σ_{k,n} |ψ_{nk}(r)|² * η / [π ((E - E_{nk})² + η²)]

The real-space wavefunction for orbital α at position r is reconstructed
from the supercell eigenvectors as:

    ψ_{nk}^α(r) = Σ_{ic} exp(i (k + G_ic)·r) · c_{α, ic}(k)
"""

import numpy as np
from ..utils.kpoints import R_z


def get_moire_lattice_vectors(G_M):
    """Compute real-space moire lattice vectors from reciprocal vectors.

    Parameters
    ----------
    G_M : list[np.ndarray]
        Moire reciprocal lattice vectors (at least 3 entries: [0, G1, G2, ...]).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Real-space lattice vectors (a1, a2).
    """
    a_M = 2 * np.pi * np.linalg.inv(np.array([G_M[1], G_M[2]]))
    a1 = a_M[:, 0]
    a2 = R_z(np.pi / 3) @ a1
    return a1, a2


def compute_r_list(r_pts, a1, a2, extra=0.0):
    """Compute 1D real-space positions along the direction a1 + a2.

    Parameters
    ----------
    r_pts : int
        Number of real-space grid points.
    a1 : np.ndarray
        First real-space moire lattice vector.
    a2 : np.ndarray
        Second real-space moire lattice vector.
    extra : float
        Fraction of the moire period to extend beyond the full a1+a2 length.
        extra=0 gives r in [0, |a1+a2|]. extra=1/6 gives [0, |a1+a2|*(1+1/6)].

    Returns
    -------
    tuple[np.ndarray, float]
        Positions of shape (r_pts, 2) and the total length rL = |a1+a2|.
    """
    direction = a1 + a2
    rL = np.linalg.norm(direction)
    r_max = rL * (1 + extra)
    r_list = np.zeros((r_pts, 2))
    for i in range(r_pts):
        r_list[i] = direction / rL * r_max * i / r_pts
    return r_list, rL


def compute_k_grid(k_pts, G_M, center=None):
    """Generate a uniform k-point grid covering the mini-Brillouin zone.

    Parameters
    ----------
    k_pts : int
        Number of points per side of the k-grid (total points = k_pts²).
    G_M : list[np.ndarray]
        Moire reciprocal lattice vectors.
    center : np.ndarray, optional
        Offset for the k-grid center. Defaults to [0, 0] (Gamma point).
        Pass the monolayer K-point for K-centered LDOS.

    Returns
    -------
    np.ndarray
        k-points of shape (k_pts², 2).
    """
    if center is None:
        center = np.zeros(2)
    G1, G2 = G_M[1], G_M[2]
    k_list = np.zeros((k_pts, k_pts, 2))
    for ix in range(k_pts):
        for iy in range(k_pts):
            k_list[ix, iy] = center + G1 * ix / k_pts + G2 * iy / k_pts
    return k_list.reshape(-1, 2)


def compute_ldos(evals, evecs, k_flat, G_M, lu, n_cells, r_list, e_list, eta):
    """Compute the real-space local density of states.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues of shape (n_k, dim) in eV, where dim = 44 * n_cells.
    evecs : np.ndarray
        Eigenvectors of shape (n_k, dim, dim), complex.
    k_flat : np.ndarray
        k-points of shape (n_k, 2).
    G_M : list[np.ndarray]
        Moire reciprocal lattice vectors.
    lu : list[tuple]
        Lookup table from MoireGeometry.lu_table().
    n_cells : int
        Number of mini-BZ cells.
    r_list : np.ndarray
        Real-space positions of shape (n_r, 2).
    e_list : np.ndarray
        Energy grid of shape (n_e,).
    eta : float
        Lorentzian broadening width (eV).

    Returns
    -------
    np.ndarray
        LDOS of shape (n_r, n_e).
    """
    n_k = evals.shape[0]
    n_r = len(r_list)
    n_e = len(e_list)
    dim = evals.shape[1]

    Gs = [np.zeros(2)]
    for i in range(1, 7):
        Gs.append(R_z(np.pi / 3 * (i - 1)) @ G_M[1])

    Kbs = np.zeros((n_cells, 2))
    for i in range(n_cells):
        Kbs[i] = Gs[1] * lu[i][0] + Gs[2] * lu[i][1]

    ig = np.arange(n_cells)[np.newaxis, :]
    alpha = np.arange(2 * 22)[:, np.newaxis]
    ind = (alpha % 22) + ig * 22 + n_cells * 22 * (alpha // 22)

    def lorentzian(E, E0):
        return eta / (np.pi * ((E - E0) ** 2 + eta ** 2))

    ldos = np.zeros((n_r, n_e))

    for ik in range(n_k):
        evals_k = evals[ik]
        evecs_k = evecs[ik]
        kGs = Kbs + k_flat[ik]
        phases = np.exp(1j * r_list @ kGs.T)[np.newaxis, :, :]
        for n, En in enumerate(evals_k):
            coeffs = evecs_k[ind, n]
            coeffs_all = coeffs[:, np.newaxis, :]
            psi_alpha = np.sum(phases * coeffs_all, axis=-1)
            psi_r_all = np.sum(np.abs(psi_alpha) ** 2, axis=0)
            lorentz = lorentzian(e_list, En)
            ldos += psi_r_all[:, None] * lorentz[None, :]

    ldos /= n_k
    return ldos
