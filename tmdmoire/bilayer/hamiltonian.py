"""Moire supercell Hamiltonian construction.

``MoireHamiltonian`` constructs the full (44·N)×(44·N) supercell
Hamiltonian for a twisted heterobilayer, where N is the number of
mini-Brillouin zones included. It combines:
- Monolayer dispersions shifted to each mini-BZ
- Interlayer coupling (w1p on p_z^e, w1d on d_z2)
- Moire potential coupling between neighboring mini-BZs
"""
import numpy as np
import scipy.linalg as la
from ..constants import (
    J_PLUS, J_MINUS, J_MX_PLUS, J_MX_MINUS, A_1, A_2, M_LIST,
)
from ..utils.kpoints import R_z
from ..material import TMDMaterial
from .geometry import MoireGeometry


class MoireHamiltonian:
    """Builds the full moire supercell Hamiltonian for a twisted heterobilayer.

    The supercell Hamiltonian has dimension (44·N)×(44·N) where N is the
    number of mini-Brillouin zones (n_cells). The basis is:
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

    def __init__(self, wse2: TMDMaterial, ws2: TMDMaterial, geometry: MoireGeometry):
        self.wse2 = wse2
        self.ws2 = ws2
        self.geometry = geometry

    def _build_moire_potential(self, V_G, V_K, psi_G, psi_K):
        """Construct the 22×22 moire potential matrix."""
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

    def _build_interlayer_coupling(self, interlayer_params, k_):
        """Construct the 22×22 interlayer coupling matrix. We use here the t=w1+w2*sum_{i=1}^6 e^{ike_i}"""
        Ham_int = np.zeros((22, 22), dtype=complex)
        orbd = 5
        orbp = 8
        nn_vectors = self.wse2.lattice_constant * np.sqrt(3) * np.array([R_z(np.pi/3*i) @ np.array([1,0]) for i in range(6)])
        nn_term = np.sum(np.exp(1j * np.array([np.dot(nn_v, k_) for nn_v in nn_vectors])))
        for i_so in [0, 11]:
            Ham_int[orbp + i_so, orbp + i_so] = interlayer_params["w1p"] + interlayer_params["w2p"] * nn_term
            Ham_int[orbd + i_so, orbd + i_so] = interlayer_params["w1d"] + interlayer_params["w2d"] * nn_term
        return Ham_int

    def _build_monolayer_ham(self, k_point, args):
        """Build a single 22×22 monolayer Hamiltonian (internal, non-vectorized)."""
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

    def build_supercell(self, k_point, n_shells, interlayer_params, pars_V,
                        k_idx=None, mono_hams_wse2=None, mono_hams_ws2=None):
        """Construct the full (44·N)×(44·N) supercell Hamiltonian.

        Parameters
        ----------
        k_point : np.ndarray
            Base k-point in the mini-BZ.
        n_shells : int
            Number of moire shells.
        interlayer_params : dict
            Interlayer coupling parameters.
        pars_V : tuple
            Moire potential parameters (V_G, V_K, psi_G, psi_K).
        k_idx : int, optional
            Index of k_point in the cached Hamiltonian lists.
            Only used when n_shells=0 and mono_hams are provided.
        mono_hams_wse2 : list[np.ndarray], optional
            Pre-computed 22×22 WSe2 Hamiltonians at each k-point.
            Only used when n_shells=0.
        mono_hams_ws2 : list[np.ndarray], optional
            Pre-computed 22×22 WS2 Hamiltonians at each k-point.
            Only used when n_shells=0.
        """
        n_cells = MoireGeometry.n_cells(n_shells)
        G_M = self.geometry.reciprocal_vectors()
        moire_ham = self._build_moire_potential(*pars_V)
        lu = MoireGeometry.lu_table(n_shells)

        use_cache = n_shells == 0 and k_idx is not None and mono_hams_wse2 is not None

        if not use_cache:
            hopping_wse2 = self.wse2.build_hopping_matrices()
            epsilon_wse2 = self.wse2.build_onsite_energies()
            HSO_wse2 = self.wse2.build_soc_hamiltonian()
            offset_wse2 = self.wse2.params[-3]
            args_wse2 = (hopping_wse2, epsilon_wse2, HSO_wse2, self.wse2.lattice_constant, offset_wse2)

            hopping_ws2 = self.ws2.build_hopping_matrices()
            epsilon_ws2 = self.ws2.build_onsite_energies()
            HSO_ws2 = self.ws2.build_soc_hamiltonian()
            offset_ws2 = self.ws2.params[-3]
            args_ws2 = (hopping_ws2, epsilon_ws2, HSO_ws2, self.ws2.lattice_constant, offset_ws2)

        Ham = np.zeros((n_cells * 44, n_cells * 44), dtype=complex)
        Kns = np.zeros((n_cells, 2))
        for i in range(n_cells):
            Kns[i] = k_point + G_M[1] * lu[i][0] + G_M[2] * lu[i][1]

        for n in range(n_cells):
            Kn = Kns[n]
            if use_cache:
                Ham[n * 22:(n + 1) * 22, n * 22:(n + 1) * 22] = mono_hams_wse2[k_idx]
                Ham[(n_cells + n) * 22:(n_cells + n + 1) * 22, (n_cells + n) * 22:(n_cells + n + 1) * 22] = mono_hams_ws2[k_idx]
            else:
                Ham[n * 22:(n + 1) * 22, n * 22:(n + 1) * 22] = self._build_monolayer_ham(Kn, args_wse2)
                Ham[(n_cells + n) * 22:(n_cells + n + 1) * 22, (n_cells + n) * 22:(n_cells + n + 1) * 22] = self._build_monolayer_ham(Kn, args_ws2)
            # Interlayer coupling w1
            interlayer_ham = self._build_interlayer_coupling(interlayer_params,Kn)
            Ham[n * 22:(n + 1) * 22, (n_cells + n) * 22:(n_cells + n + 1) * 22] = interlayer_ham

        Ham[n_cells * 22:, :n_cells * 22] = np.copy(Ham[:n_cells * 22, n_cells * 22:].T.conj())

        for n in range(0, n_shells + 1):    # Index of the shell
            for s in range(np.sign(n) * (1 + (n - 1) * n * 3), n * (n + 1) * 3 + 1):    # Index inside the shell
                ind_s = lu[s]
                for i in M_LIST:
                    ind_nn = (ind_s[0] + i[0], ind_s[1] + i[1])
                    try:
                        nn = lu.index(ind_nn)
                    except ValueError:
                        continue
                    g = M_LIST.index(i)
                    Vup = moire_ham if g % 2 == 0 else moire_ham.conj()
                    Ham[s * 22:(s + 1) * 22, nn * 22:(nn + 1) * 22] += Vup
                    Ham[n_cells * 22 + s * 22:n_cells * 22 + (s + 1) * 22, n_cells * 22 + nn * 22:n_cells * 22 + (nn + 1) * 22] += Vup
        return Ham

    def diagonalize(self, k_points, n_shells, interlayer_params, pars_V,
                    mono_hams_wse2=None, mono_hams_ws2=None):
        """Diagonalize the supercell Hamiltonian at each k-point."""
        k_pts = k_points.shape[0]
        n_cells = MoireGeometry.n_cells(n_shells)
        evals = np.zeros((k_pts, n_cells * 44))
        evecs = np.zeros((k_pts, n_cells * 44, n_cells * 44), dtype=complex)
        for i in range(k_pts):
            H_tot = self.build_supercell(k_points[i], n_shells, interlayer_params, pars_V,
                                         k_idx=i,
                                         mono_hams_wse2=mono_hams_wse2,
                                         mono_hams_ws2=mono_hams_ws2)
            evals[i], evecs[i] = la.eigh(H_tot, check_finite=False, overwrite_a=True)
        return evals, evecs
