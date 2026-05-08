"""K-point path generation and rotation utilities."""
import numpy as np


def R_z(t):
    """2D rotation matrix by angle t (radians).

    Parameters
    ----------
    t : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        2×2 rotation matrix.
    """
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def get_k_list(cut, k_pts, tmd="WSe2", endpoint=False, return_norm=False):
    """Generate momentum points along a high-symmetry path.

    Parameters
    ----------
    cut : str
        Path specification, e.g. "Kp-G-K" or "K-M-Kp".
        Supported points: Kp, K, M, G.
    k_pts : int
        Total number of points along the path.
    tmd : str
        Material name for lattice constant lookup.
    endpoint : bool
        Whether to include the final point of the last segment.
    return_norm : bool
        If True, also return cumulative distance along the path.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Momentum points of shape (k_pts, 2), optionally with cumulative
        distances of shape (k_pts,).
    """
    from ..constants import LATTICE_CONSTANTS

    b2 = 4 * np.pi / np.sqrt(3) / LATTICE_CONSTANTS[tmd] * np.array([0, 1])
    b1 = R_z(-np.pi / 3) @ b2
    b6 = R_z(-2 * np.pi / 3) @ b2
    K = (b1 + b6) / 3
    Kp = R_z(np.pi / 3) @ K
    G = np.array([0, 0])
    M = b1 / 2
    dic_Kpts = {"K": K, "Kp": Kp, "G": G, "M": M}
    terms = cut.split("-")
    dks = np.zeros(len(terms) - 1)
    for i in range(len(terms) - 1):
        dks[i] = np.linalg.norm(dic_Kpts[terms[i + 1]] - dic_Kpts[terms[i]])
    tot_k = np.sum(dks)
    ls = np.zeros(len(terms) - 1, dtype=int)
    for i in range(len(terms) - 1):
        ls[i] = int(dks[i] / tot_k * k_pts)
        if i == len(terms) - 2 and endpoint:
            ls[i] += 1
    k_pts = ls.sum() + 1
    res = np.zeros((k_pts, 2))
    for i in range(len(terms) - 1):
        for p in range(ls[i]):
            ind = p if i == 0 else p + ls[:i].sum()
            res[ind] = dic_Kpts[terms[i]] + (dic_Kpts[terms[i + 1]] - dic_Kpts[terms[i]]) / ls[i] * p
        if i == len(terms) - 2:
            res[-1] = dic_Kpts[terms[i + 1]]
    if return_norm:
        norm = np.zeros(k_pts)
        for i in range(1, k_pts):
            norm[i] = norm[i - 1] + np.linalg.norm(res[i] - res[i - 1])
        return res, norm
    return res
