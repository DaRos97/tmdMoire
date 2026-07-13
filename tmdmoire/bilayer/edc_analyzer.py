"""EDC analysis for bilayer moire superlattice.

Computes energy distribution curves from supercell eigenvalues,
fits Lorentzian profiles, computes band gaps, and local density of states.
"""
import numpy as np

from ..material import TMDMaterial
from .geometry import MoireGeometry
from .hamiltonian import MoireHamiltonian
from ..constants import ENERGY_OFFSETS


def find_peak_seeds_gamma(weight_list, energy_list, full_energy_values, full_weight_values):
    """Find 4 peak seeds for Gamma-point EDC using find_peaks on intensity profile.

    Uses scipy.signal.find_peaks to locate local maxima, then classifies
    by energy region (TVB > -1.5 eV, LVB < -1.5 eV). Side bands are found
    from raw eigenstates.

    Parameters
    ----------
    weight_list : np.ndarray
        Spread intensity profile.
    energy_list : np.ndarray
        Energy grid corresponding to weight_list.
    full_energy_values : np.ndarray
        Raw eigenvalues in the band window.
    full_weight_values : np.ndarray
        Raw orbital weights in the band window.

    Returns
    -------
    list of tuple
        Four (energy, weight) tuples sorted by energy descending.
    """
    from scipy.signal import find_peaks

    peaks_idx, _ = find_peaks(weight_list, height=weight_list.max() * 0.005, distance=int(0.01 / 0.005))
    peaks_found = list(zip(energy_list[peaks_idx], weight_list[peaks_idx]))

    tvb_region = [(e, h) for e, h in peaks_found if e > -1.5]
    tvb_main = max(tvb_region, key=lambda x: x[1]) if tvb_region else (-1.16, 10.0)

    lvb_region = [(e, h) for e, h in peaks_found if e < -1.5]
    lvb_main = max(lvb_region, key=lambda x: x[1]) if lvb_region else (-1.82, 10.0)

    eigen_by_energy = sorted(zip(full_energy_values, full_weight_values), key=lambda x: x[0], reverse=True)

    side_candidates = [e for e in eigen_by_energy if e[0] < tvb_main[0] - 0.01 and e[0] > -1.5]
    tvb_side = max(side_candidates, key=lambda x: x[1]) if side_candidates else (tvb_main[0] - 0.05, tvb_main[1] * 0.3)

    lvb_side_candidates = [e for e in eigen_by_energy if e[0] < lvb_main[0] - 0.01]
    lvb_side = max(lvb_side_candidates, key=lambda x: x[1]) if lvb_side_candidates else (lvb_main[0] - 0.05, lvb_main[1] * 0.3)

    peak_states = sorted([tvb_main, tvb_side, lvb_main, lvb_side], key=lambda x: x[0], reverse=True)
    return peak_states


class EDCAnalyzer:
    def __init__(self, wse2: TMDMaterial, ws2: TMDMaterial, geometry: MoireGeometry, config: dict):
        self.wse2 = wse2
        self.ws2 = ws2
        self.geometry = geometry
        self.config = config

    def compute_gap(self, params: tuple, bz_point: str, plot_bands_gap: bool = False):
        n_cells = self.config["n_cells"]
        k_point = self.config["k_point"]
        interlayer_params = self.config["interlayer_params"]
        pars_V = self.config["pars_V"]

        if bz_point == "G":
            Vg, phiG = params
            pars_V = (Vg, pars_V[1], phiG, pars_V[3])
        elif bz_point == "K":
            Vk, phiK = params
            pars_V = (pars_V[0], Vk, pars_V[2], phiK)

        pts = 51
        k_list = np.zeros((pts, 2))
        k_list[:, 0] = np.linspace(0, 0.12, pts)
        if bz_point == "K":
            from ..constants import LATTICE_CONSTANTS
            k_list[:, 0] += 4 * np.pi / 3 / LATTICE_CONSTANTS["WSe2"]

        moire_ham = MoireHamiltonian(self.wse2, self.ws2, self.geometry)
        evals, evecs = moire_ham.diagonalize(
            k_list, self.config["n_shells"], interlayer_params, pars_V
        )

        n_tvb = 28 * n_cells
        gap = np.min(evals[:, n_tvb - 1] - evals[:, n_tvb - 2])
        return gap

    def compute_ldos(self, evals, evecs, r_list, e_list, k_flat, spreadE):
        n_shells = self.config["n_shells"]
        n_cells = self.config["n_cells"]
        theta = self.config["theta_deg"] / 180 * np.pi

        r_pts = r_list.shape[0]
        e_pts = len(e_list)
        k_pts = k_flat.shape[0]
        LDOS = np.zeros((r_pts, e_pts))

        lu = MoireGeometry.lu_table(n_shells)
        G_M = self.geometry.reciprocal_vectors()
        Kbs = np.zeros((n_cells, 2))
        for i in range(n_cells):
            Kbs[i] = G_M[1] * lu[i][0] + G_M[2] * lu[i][1]

        ig = np.arange(n_cells)[np.newaxis, :]
        alpha = np.arange(44)[:, np.newaxis]
        ind = (alpha % 22) + ig * 22 + n_cells * 22 * (alpha // 22)

        for ik in range(k_pts):
            evals_k = evals[ik]
            evecs_k = evecs[ik]
            k_gs = Kbs + k_flat[ik]
            phases = np.exp(1j * r_list @ k_gs.T)[np.newaxis, :, :]

            for n, En in enumerate(evals_k):
                coeffs = evecs_k[ind, n]
                coeffs_all = coeffs[:, np.newaxis, :]
                psi_alpha = np.sum(phases * coeffs_all, axis=-1)
                psi_r_all = np.sum(np.abs(psi_alpha) ** 2, axis=0)
                lorentz_matrix = spreadE / (np.pi * ((e_list - En) ** 2 + spreadE ** 2))
                LDOS += psi_r_all[:, None] * lorentz_matrix[None, :] / k_pts

        return LDOS
