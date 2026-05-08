"""Monolayer Hamiltonian construction.

``MonolayerHamiltonian`` builds the 22×22 single-layer Hamiltonian
H(k) = H_TB(k) + H_SO + offset·I at arbitrary momentum points.
The tight-binding part includes nearest-neighbor (t1–t3), M–X coupling
(t4–t5), and second-nearest-neighbor (t6) hoppings.
"""
import numpy as np
import scipy.linalg as la
from ..constants import (
    J_PLUS, J_MINUS, J_MX_PLUS, J_MX_MINUS, A_1, A_2,
)


class MonolayerHamiltonian:
    """Builds the 22×22 monolayer tight-binding Hamiltonian H(k).

    The basis is 11 orbitals × 2 spins = 22 dimensions:
    - 0–10: spin-up, 11–21: spin-down
    - Orbitals: [d_xz, d_yz, p_z^o, p_x^o, p_y^o, d_z2, d_xy, d_x2-y2, p_z^e, p_x^e, p_y^e]

    Parameters
    ----------
    material : TMDMaterial
        The TMD material providing the lattice constant.

    Examples
    --------
    >>> from tmdmoire import TMDMaterial
    >>> mat = TMDMaterial("WSe2")
    >>> ham = MonolayerHamiltonian(mat)
    >>> t = mat.build_hopping_matrices()
    >>> e = mat.build_onsite_energies()
    >>> HSO = mat.build_soc_hamiltonian()
    >>> k = np.array([0.0, 0.0])  # Gamma point
    >>> H = ham.build(k, t, e, HSO, mat.params[-3])
    >>> H.shape
    (22, 22)
    """

    def __init__(self, material):
        self.material = material

    def build(self, k_points, hopping, epsilon, HSO, offset):
        """Construct the 22×22 Hamiltonian at given momentum points.

        Parameters
        ----------
        k_points : np.ndarray
            Momentum points, shape (2,) for a single point or (N, 2) for batch.
        hopping : list[np.ndarray]
            Six 11×11 hopping matrices from ``TMDMaterial.build_hopping_matrices()``.
        epsilon : np.ndarray
            11 on-site energies from ``TMDMaterial.build_onsite_energies()``.
        HSO : np.ndarray
            22×22 SOC Hamiltonian from ``TMDMaterial.build_soc_hamiltonian()``.
        offset : float
            Global energy shift (params[40]).

        Returns
        -------
        np.ndarray
            Hamiltonian: shape (22, 22) for single k, or (N, 22, 22) for batch.
        """
        a_mono = self.material.lattice_constant
        delta = a_mono * np.array([
            A_1, A_1 + A_2, A_2,
            -(2 * A_1 + A_2) / 3, (A_1 + 2 * A_2) / 3, (A_1 - A_2) / 3,
            -2 * (A_1 + 2 * A_2) / 3, 2 * (2 * A_1 + A_2) / 3, 2 * (A_2 - A_1) / 3,
        ])
        vec = len(k_points.shape) == 2
        H_0 = np.zeros((11, 11), dtype=complex) if not vec else np.zeros((11, 11, k_points.shape[0]), dtype=complex)

        for i in range(11):
            H_0[i, i] += (epsilon[i]
                + 2 * hopping[0][i, i] * np.cos(np.dot(k_points, delta[0]))
                + 2 * hopping[1][i, i] * (np.cos(np.dot(k_points, delta[1])) + np.cos(np.dot(k_points, delta[2])))
            )

        for ind in J_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (2 * hopping[0][i, j] * np.cos(np.dot(k_points, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_points, delta[1])) + np.exp(-1j * np.dot(k_points, delta[2])))
                + hopping[2][i, j] * (np.exp(1j * np.dot(k_points, delta[1])) + np.exp(1j * np.dot(k_points, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        for ind in J_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (-2 * 1j * hopping[0][i, j] * np.sin(np.dot(k_points, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_points, delta[1])) - np.exp(-1j * np.dot(k_points, delta[2])))
                + hopping[2][i, j] * (-np.exp(1j * np.dot(k_points, delta[1])) + np.exp(1j * np.dot(k_points, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        for ind in J_MX_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = hopping[3][i, j] * (np.exp(1j * np.dot(k_points, delta[3])) - np.exp(1j * np.dot(k_points, delta[5])))
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        for ind in J_MX_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (hopping[3][i, j] * (np.exp(1j * np.dot(k_points, delta[3])) + np.exp(1j * np.dot(k_points, delta[5])))
                + hopping[4][i, j] * np.exp(1j * np.dot(k_points, delta[4]))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        H_1 = np.zeros((11, 11), dtype=complex) if not vec else np.zeros((11, 11, k_points.shape[0]), dtype=complex)
        H_1[8, 5] += hopping[5][8, 5] * (np.exp(1j * np.dot(k_points, delta[6])) + np.exp(1j * np.dot(k_points, delta[7])) + np.exp(1j * np.dot(k_points, delta[8])))
        H_1[5, 8] += np.conjugate(H_1[8, 5])
        H_1[10, 5] += hopping[5][10, 5] * (np.exp(1j * np.dot(k_points, delta[6])) - 1/2 * np.exp(1j * np.dot(k_points, delta[7])) - 1/2 * np.exp(1j * np.dot(k_points, delta[8])))
        H_1[5, 10] += np.conjugate(H_1[10, 5])
        H_1[9, 5] += np.sqrt(3) / 2 * hopping[5][10, 5] * (-np.exp(1j * np.dot(k_points, delta[7])) + np.exp(1j * np.dot(k_points, delta[8])))
        H_1[5, 9] += np.conjugate(H_1[9, 5])
        H_1[8, 7] += hopping[5][8, 7] * (np.exp(1j * np.dot(k_points, delta[6])) - 1/2 * np.exp(1j * np.dot(k_points, delta[7])) - 1/2 * np.exp(1j * np.dot(k_points, delta[8])))
        H_1[7, 8] += np.conjugate(H_1[8, 7])
        H_1[8, 6] += np.sqrt(3) / 2 * hopping[5][8, 7] * (-np.exp(1j * np.dot(k_points, delta[7])) + np.exp(1j * np.dot(k_points, delta[8])))
        H_1[6, 8] += np.conjugate(H_1[8, 6])
        H_1[9, 6] += 3/4 * hopping[5][10, 7] * (np.exp(1j * np.dot(k_points, delta[7])) + np.exp(1j * np.dot(k_points, delta[8])))
        H_1[6, 9] += np.conjugate(H_1[9, 6])
        H_1[10, 6] += np.sqrt(3) / 4 * hopping[5][10, 7] * (np.exp(1j * np.dot(k_points, delta[7])) - np.exp(1j * np.dot(k_points, delta[8])))
        H_1[6, 10] += np.conjugate(H_1[10, 6])
        H_1[9, 7] += H_1[10, 6]
        H_1[7, 9] += H_1[6, 10]
        H_1[10, 7] += hopping[5][10, 7] * (np.exp(1j * np.dot(k_points, delta[6])) + 1/4 * np.exp(1j * np.dot(k_points, delta[7])) + 1/4 * np.exp(1j * np.dot(k_points, delta[8])))
        H_1[7, 10] += np.conjugate(H_1[10, 7])

        H_TB = H_0 + H_1
        H = np.zeros((22, 22), dtype=complex) if not vec else np.zeros((22, 22, k_points.shape[0]), dtype=complex)
        H[:11, :11] = H_TB
        H[11:, 11:] = H_TB
        if len(H.shape) == 3:
            H = np.transpose(H, (2, 0, 1))
        H += HSO
        H += np.identity(22) * offset
        return H

    def eigenvalues(self, k_points, hopping, epsilon, HSO, offset):
        """Compute eigenvalues of H(k) at given momentum points."""
        H = self.build(k_points, hopping, epsilon, HSO, offset)
        if len(H.shape) == 3:
            return np.array([la.eigvalsh(H[i]) for i in range(H.shape[0])])
        return la.eigvalsh(H)

    def eigenvectors(self, k_points, hopping, epsilon, HSO, offset):
        """Compute eigenvalues and eigenvectors of H(k)."""
        H = self.build(k_points, hopping, epsilon, HSO, offset)
        if len(H.shape) == 3:
            evals = np.zeros((H.shape[0], 22))
            evecs = np.zeros((H.shape[0], 22, 22), dtype=complex)
            for i in range(H.shape[0]):
                evals[i], evecs[i] = la.eigh(H[i])
            return evals, evecs
        return la.eigh(H)
