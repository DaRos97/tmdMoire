"""Monolayer and moire Hamiltonian construction.

``MonolayerHamiltonian`` builds the 22×22 single-layer Hamiltonian
H(k) = H_TB(k) + H_SO + offset·I at arbitrary momentum points.
The tight-binding part includes nearest-neighbor (t1–t3), M–X coupling
(t4–t5), and second-nearest-neighbor (t6) hoppings.

``MoireHamiltonian`` constructs the full (44·N)×(44·N) supercell
Hamiltonian for a twisted heterobilayer, where N is the number of
mini-Brillouin zones included. It combines:
- Monolayer dispersions shifted to each mini-BZ
- Interlayer coupling (w1p on p_z^e, w1d on d_z2)
- Moire potential coupling between neighboring mini-BZs
"""
import numpy as np
import scipy.linalg as la
from .constants import (
    J_PLUS, J_MINUS, J_MX_PLUS, J_MX_MINUS, A_1, A_2, M_LIST,
)
from .moire_geometry import MoireGeometry


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

        # Diagonal elements
        for i in range(11):
            H_0[i, i] += (epsilon[i]
                + 2 * hopping[0][i, i] * np.cos(np.dot(k_points, delta[0]))
                + 2 * hopping[1][i, i] * (np.cos(np.dot(k_points, delta[1])) + np.cos(np.dot(k_points, delta[2])))
            )

        # Off-diagonal symmetric (+) terms
        for ind in J_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (2 * hopping[0][i, j] * np.cos(np.dot(k_points, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_points, delta[1])) + np.exp(-1j * np.dot(k_points, delta[2])))
                + hopping[2][i, j] * (np.exp(1j * np.dot(k_points, delta[1])) + np.exp(1j * np.dot(k_points, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        # Off-diagonal antisymmetric (-) terms
        for ind in J_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (-2 * 1j * hopping[0][i, j] * np.sin(np.dot(k_points, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_points, delta[1])) - np.exp(-1j * np.dot(k_points, delta[2])))
                + hopping[2][i, j] * (-np.exp(1j * np.dot(k_points, delta[1])) + np.exp(1j * np.dot(k_points, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        # M-X coupling (+) terms
        for ind in J_MX_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = hopping[3][i, j] * (np.exp(1j * np.dot(k_points, delta[3])) - np.exp(1j * np.dot(k_points, delta[5])))
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        # M-X coupling (-) terms
        for ind in J_MX_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (hopping[3][i, j] * (np.exp(1j * np.dot(k_points, delta[3])) + np.exp(1j * np.dot(k_points, delta[5])))
                + hopping[4][i, j] * np.exp(1j * np.dot(k_points, delta[4]))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        # Second-nearest-neighbor (t6) terms
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
        """Compute eigenvalues of H(k) at given momentum points.

        Parameters
        ----------
        k_points : np.ndarray
            Momentum points, shape (2,) or (N, 2).
        hopping, epsilon, HSO, offset
            Same as :meth:`build`.

        Returns
        -------
        np.ndarray
            Eigenvalues: shape (22,) for single k, or (N, 22) for batch.
        """
        H = self.build(k_points, hopping, epsilon, HSO, offset)
        if len(H.shape) == 3:
            return np.array([la.eigvalsh(H[i]) for i in range(H.shape[0])])
        return la.eigvalsh(H)

    def eigenvectors(self, k_points, hopping, epsilon, HSO, offset):
        """Compute eigenvalues and eigenvectors of H(k).

        Parameters
        ----------
        k_points : np.ndarray
            Momentum points, shape (2,) or (N, 2).
        hopping, epsilon, HSO, offset
            Same as :meth:`build`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Eigenvalues and eigenvectors. For batch: ((N, 22), (N, 22, 22)).
        """
        H = self.build(k_points, hopping, epsilon, HSO, offset)
        if len(H.shape) == 3:
            evals = np.zeros((H.shape[0], 22))
            evecs = np.zeros((H.shape[0], 22, 22), dtype=complex)
            for i in range(H.shape[0]):
                evals[i], evecs[i] = la.eigh(H[i])
            return evals, evecs
        return la.eigh(H)


class MoireHamiltonian:
    """Builds the full moire supercell Hamiltonian for a twisted heterobilayer.

    The supercell Hamiltonian has dimension (44·N)×(44·N) where N is the
    number of mini-Brillouin zones (nCells). The basis is:
    - 0 to 22·N−1: WSe2 orbitals in each mini-BZ
    - 22·N to 44·N−1: WS2 orbitals in each mini-BZ

    Parameters
    ----------
    wse2 : TMDMaterial
        WSe2 monolayer material.
    ws2 : TMDMaterial
        WS2 monolayer material.
    geometry : MoireGeometry
        Moire lattice geometry (twist angle, reciprocal vectors).
    """

    def __init__(self, wse2, ws2, geometry: MoireGeometry):
        self.wse2 = wse2
        self.ws2 = ws2
        self.geometry = geometry

    def _build_moire_potential(self, V_G, V_K, psi_G, psi_K):
        """Construct the 22×22 moire potential matrix.

        Applies different potential amplitudes to out-of-plane vs in-plane
        orbitals with complex phases.

        Parameters
        ----------
        V_G : float
            Moire potential amplitude at Gamma (out-of-plane orbitals).
        V_K : float
            Moire potential amplitude at K (in-plane orbitals).
        psi_G : float
            Phase for out-of-plane orbitals.
        psi_K : float
            Phase for in-plane orbitals.

        Returns
        -------
        np.ndarray
            22×22 complex moire potential matrix.
        """
        Id = np.zeros((22, 22), dtype=complex)
        out_of_plane = V_G * np.exp(1j * psi_G)
        in_plane = V_K * np.exp(1j * psi_K)
        list_out = (0, 1, 2, 5, 8)
        list_in = (3, 4, 6, 7, 9, 10)
        for i in list_out:
            Id[i, i] = out_of_plane
            Id[i + 11, i + 11] = out_of_plane
        for i in list_in:
            Id[i, i] = in_plane
            Id[i + 11, i + 11] = in_plane
        return Id

    def build_supercell(self, k_point, n_shells, interlayer_params, pars_V):
        """Construct the full (44·N)×(44·N) supercell Hamiltonian.

        Parameters
        ----------
        k_point : np.ndarray
            Momentum point in the mini-BZ, shape (2,).
        n_shells : int
            Number of mini-BZ shells around the central zone.
        interlayer_params : dict
            Interlayer coupling parameters:
            - stacking: 'P' or 'H'
            - w1p, w2p: p-orbital coupling strengths
            - w1d, w2d: d-orbital coupling strengths
        pars_V : tuple
            Moire potential parameters (V_G, V_K, psi_G, psi_K).

        Returns
        -------
        np.ndarray
            Full supercell Hamiltonian, shape (44·N, 44·N).
        """
        n_cells = MoireGeometry.n_cells(n_shells)
        G_M = self.geometry.reciprocal_vectors()
        moire_ham = self._build_moire_potential(*pars_V)
        lu = MoireGeometry.lu_table(n_shells)

        hopping_wse2 = self.wse2.build_hopping_matrices()
        epsilon_wse2 = self.wse2.build_onsite_energies()
        HSO_wse2 = self.wse2.build_soc_hamiltonian()
        offset_wse2 = self.wse2.params[-3]

        hopping_ws2 = self.ws2.build_hopping_matrices()
        epsilon_ws2 = self.ws2.build_onsite_energies()
        HSO_ws2 = self.ws2.build_soc_hamiltonian()
        offset_ws2 = self.ws2.params[-3]

        args_wse2 = (hopping_wse2, epsilon_wse2, HSO_wse2, self.wse2.lattice_constant, offset_wse2)
        args_ws2 = (hopping_ws2, epsilon_ws2, HSO_ws2, self.ws2.lattice_constant, offset_ws2)

        Ham = np.zeros((n_cells * 44, n_cells * 44), dtype=complex)
        Kns = np.zeros((n_cells, 2))
        for i in range(n_cells):
            Kns[i] = k_point + G_M[1] * lu[i][0] + G_M[2] * lu[i][1]

        orbdList = [5, 16]
        psi_interlayer = 0 if interlayer_params['stacking'] == 'P' else 2 * np.pi / 3

        # Monolayer dispersions and interlayer coupling
        for n in range(n_cells):
            Kn = Kns[n]
            Ham[n * 22:(n + 1) * 22, n * 22:(n + 1) * 22] = self._build_monolayer_ham(Kn, args_wse2)
            Ham[(n_cells + n) * 22:(n_cells + n + 1) * 22, (n_cells + n) * 22:(n_cells + n + 1) * 22] = self._build_monolayer_ham(Kn, args_ws2)

            for iSO in [0, 11]:
                Ham[n * 22 + 8 + iSO, (n_cells + n) * 22 + 8 + iSO] += interlayer_params['w1p']
            for orbd in orbdList:
                Ham[n * 22 + orbd, (n_cells + n) * 22 + orbd] += interlayer_params['w1d']

        Ham[n_cells * 22:, :n_cells * 22] = np.copy(Ham[:n_cells * 22, n_cells * 22:].T.conj())

        # Moire replicas: coupling between neighboring mini-BZs
        for n in range(0, n_shells + 1):
            for s in range(np.sign(n) * (1 + (n - 1) * n * 3), n * (n + 1) * 3 + 1):
                ind_s = lu[s]
                for i in M_LIST:
                    ind_nn = (ind_s[0] + i[0], ind_s[1] + i[1])
                    try:
                        nn = lu.index(ind_nn)
                    except ValueError:
                        continue
                    g = M_LIST.index(i)
                    Vup = moire_ham if g % 2 else moire_ham.conj()
                    Ham[s * 22:(s + 1) * 22, nn * 22:(nn + 1) * 22] += Vup
                    Ham[n_cells * 22 + s * 22:n_cells * 22 + (s + 1) * 22, n_cells * 22 + nn * 22:n_cells * 22 + (nn + 1) * 22] += Vup

        return Ham

    def _build_monolayer_ham(self, k_point, args):
        """Build a single 22×22 monolayer Hamiltonian (internal, non-vectorized).

        Parameters
        ----------
        k_point : np.ndarray
            Single momentum point, shape (2,).
        args : tuple
            (hopping, epsilon, HSO, a_mono, offset).

        Returns
        -------
        np.ndarray
            22×22 Hamiltonian.
        """
        hopping, epsilon, HSO, a_mono, offset = args
        delta = a_mono * np.array([
            A_1, A_1 + A_2, A_2,
            -(2 * A_1 + A_2) / 3, (A_1 + 2 * A_2) / 3, (A_1 - A_2) / 3,
            -2 * (A_1 + 2 * A_2) / 3, 2 * (2 * A_1 + A_2) / 3, 2 * (A_2 - A_1) / 3,
        ])
        H_0 = np.zeros((11, 11), dtype=complex)
        for i in range(11):
            H_0[i, i] += (epsilon[i]
                + 2 * hopping[0][i, i] * np.cos(np.dot(k_point, delta[0]))
                + 2 * hopping[1][i, i] * (np.cos(np.dot(k_point, delta[1])) + np.cos(np.dot(k_point, delta[2])))
            )
        for ind in J_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (2 * hopping[0][i, j] * np.cos(np.dot(k_point, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_point, delta[1])) + np.exp(-1j * np.dot(k_point, delta[2])))
                + hopping[2][i, j] * (np.exp(1j * np.dot(k_point, delta[1])) + np.exp(1j * np.dot(k_point, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)
        for ind in J_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (-2 * 1j * hopping[0][i, j] * np.sin(np.dot(k_point, delta[0]))
                + hopping[1][i, j] * (np.exp(-1j * np.dot(k_point, delta[1])) - np.exp(-1j * np.dot(k_point, delta[2])))
                + hopping[2][i, j] * (-np.exp(1j * np.dot(k_point, delta[1])) + np.exp(1j * np.dot(k_point, delta[2])))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)
        for ind in J_MX_PLUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = hopping[3][i, j] * (np.exp(1j * np.dot(k_point, delta[3])) - np.exp(1j * np.dot(k_point, delta[5])))
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)
        for ind in J_MX_MINUS:
            i, j = ind[0] - 1, ind[1] - 1
            temp = (hopping[3][i, j] * (np.exp(1j * np.dot(k_point, delta[3])) + np.exp(1j * np.dot(k_point, delta[5])))
                + hopping[4][i, j] * np.exp(1j * np.dot(k_point, delta[4]))
            )
            H_0[i, j] += temp
            H_0[j, i] += np.conjugate(temp)

        H_1 = np.zeros((11, 11), dtype=complex)
        H_1[8, 5] += hopping[5][8, 5] * (np.exp(1j * np.dot(k_point, delta[6])) + np.exp(1j * np.dot(k_point, delta[7])) + np.exp(1j * np.dot(k_point, delta[8])))
        H_1[5, 8] += np.conjugate(H_1[8, 5])
        H_1[10, 5] += hopping[5][10, 5] * (np.exp(1j * np.dot(k_point, delta[6])) - 1/2 * np.exp(1j * np.dot(k_point, delta[7])) - 1/2 * np.exp(1j * np.dot(k_point, delta[8])))
        H_1[5, 10] += np.conjugate(H_1[10, 5])
        H_1[9, 5] += np.sqrt(3) / 2 * hopping[5][10, 5] * (-np.exp(1j * np.dot(k_point, delta[7])) + np.exp(1j * np.dot(k_point, delta[8])))
        H_1[5, 9] += np.conjugate(H_1[9, 5])
        H_1[8, 7] += hopping[5][8, 7] * (np.exp(1j * np.dot(k_point, delta[6])) - 1/2 * np.exp(1j * np.dot(k_point, delta[7])) - 1/2 * np.exp(1j * np.dot(k_point, delta[8])))
        H_1[7, 8] += np.conjugate(H_1[8, 7])
        H_1[8, 6] += np.sqrt(3) / 2 * hopping[5][8, 7] * (-np.exp(1j * np.dot(k_point, delta[7])) + np.exp(1j * np.dot(k_point, delta[8])))
        H_1[6, 8] += np.conjugate(H_1[8, 6])
        H_1[9, 6] += 3/4 * hopping[5][10, 7] * (np.exp(1j * np.dot(k_point, delta[7])) + np.exp(1j * np.dot(k_point, delta[8])))
        H_1[6, 9] += np.conjugate(H_1[9, 6])
        H_1[10, 6] += np.sqrt(3) / 4 * hopping[5][10, 7] * (np.exp(1j * np.dot(k_point, delta[7])) - np.exp(1j * np.dot(k_point, delta[8])))
        H_1[6, 10] += np.conjugate(H_1[10, 6])
        H_1[9, 7] += H_1[10, 6]
        H_1[7, 9] += H_1[6, 10]
        H_1[10, 7] += hopping[5][10, 7] * (np.exp(1j * np.dot(k_point, delta[6])) + 1/4 * np.exp(1j * np.dot(k_point, delta[7])) + 1/4 * np.exp(1j * np.dot(k_point, delta[8])))
        H_1[7, 10] += np.conjugate(H_1[10, 7])

        H_TB = H_0 + H_1
        H = np.zeros((22, 22), dtype=complex)
        H[:11, :11] = H_TB
        H[11:, 11:] = H_TB
        H += HSO
        H += np.identity(22) * offset
        return H

    def diagonalize(self, k_points, n_shells, interlayer_params, pars_V):
        """Diagonalize the supercell Hamiltonian at each k-point.

        Parameters
        ----------
        k_points : np.ndarray
            Momentum points, shape (N_k, 2).
        n_shells : int
            Number of mini-BZ shells.
        interlayer_params : dict
            Interlayer coupling parameters.
        pars_V : tuple
            Moire potential parameters (V_G, V_K, psi_G, psi_K).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Eigenvalues (N_k, 44·N) and eigenvectors (N_k, 44·N, 44·N).
        """
        k_pts = k_points.shape[0]
        n_cells = MoireGeometry.n_cells(n_shells)
        evals = np.zeros((k_pts, n_cells * 44))
        evecs = np.zeros((k_pts, n_cells * 44, n_cells * 44), dtype=complex)
        for i in range(k_pts):
            H_tot = self.build_supercell(k_points[i], n_shells, interlayer_params, pars_V)
            evals[i], evecs[i] = scipy.linalg.eigh(H_tot, check_finite=False, overwrite_a=True)
        return evals, evecs
