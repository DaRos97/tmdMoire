import numpy as np
import scipy
from scipy.special import wofz
import lmfit
from .material import TMDMaterial
from .moire_geometry import MoireGeometry
from .hamiltonian import MoireHamiltonian


def _voigt(x, center, amplitude, gamma, sigma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def _two_lorentzian_one_gaussian(x, amp1, cen1, gam1, amp2, cen2, gam2, sig):
    return _voigt(x, cen1, amp1, gam1, sig) + _voigt(x, cen2, amp2, gam2, sig)


class EDCAnalyzer:
    def __init__(self, wse2: TMDMaterial, ws2: TMDMaterial, geometry: MoireGeometry, config: dict):
        self.wse2 = wse2
        self.ws2 = ws2
        self.geometry = geometry
        self.config = config

    def compute_edc(self, params: tuple, bz_point: str, spreadE: float = 0.03,
                    sample: str = "S11", plot_bands: bool = False, plot_fit: bool = False):
        from .constants import EDC_G_POSITIONS, EDC_K_POSITIONS, ENERGY_OFFSETS

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

        moire_ham = MoireHamiltonian(self.wse2, self.ws2, self.geometry)
        evals, evecs = moire_ham.diagonalize(
            k_point, self.config["n_shells"], interlayer_params, pars_V
        )

        evals = evals[0]
        evecs = evecs[0]
        ab = np.absolute(evecs) ** 2
        weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0)

        pTVB, successTVB = self._fit_bands("TVB", evals, weights, n_cells, spreadE, sample, bz_point, plot_fit)
        if not successTVB:
            return np.nan, False

        if bz_point == "K":
            return pTVB, True

        pLVB, successLVB = self._fit_bands("LVB", evals, weights, n_cells, spreadE, sample, bz_point, plot_fit)
        if successLVB:
            return (pTVB[0], pTVB[1], pLVB[0]), True
        return np.nan, False

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
            from .constants import LATTICE_CONSTANTS
            k_list[:, 0] += 4 * np.pi / 3 / LATTICE_CONSTANTS["WSe2"]

        moire_ham = MoireHamiltonian(self.wse2, self.ws2, self.geometry)
        evals, evecs = moire_ham.diagonalize(
            k_list, self.config["n_shells"], interlayer_params, pars_V
        )

        nTVB = 28 * n_cells
        gap = np.min(evals[:, nTVB - 1] - evals[:, nTVB - 2])
        return gap

    def _fit_bands(self, band_type, evals, weights, n_cells, spreadE, sample, bz_point, plot_fit):
        indexB = 28 * n_cells - 1 if band_type == "TVB" else 26 * n_cells - 1
        indexL = indexB - 2 * n_cells + 1 if bz_point == "G" else indexB - n_cells + np.argmax(weights[indexB - n_cells:indexB]) + 1
        nSOC = 2 if bz_point == "G" else 1
        energyB = evals[indexB]
        weightB = weights[indexB]

        fullEnergyValues = evals[indexL:indexB + 1]
        fullWeightValues = weights[indexL:indexB + 1]

        minE = fullEnergyValues[0]
        maxE = fullEnergyValues[-1]
        Delta = maxE - minE
        minE -= Delta / 2
        maxE += Delta / 2
        nE = int((maxE - minE) / 0.005)
        energyList = np.linspace(minE, maxE, nE)
        weightList = np.zeros(len(energyList))

        if np.max(fullWeightValues[:-nSOC]) > weightB:
            return (0, 0), False

        for i in range(len(fullEnergyValues)):
            weightList += spreadE / np.pi * fullWeightValues[i] / ((energyList - fullEnergyValues[i]) ** 2 + spreadE ** 2)

        model = lmfit.Model(_two_lorentzian_one_gaussian)
        cen1 = energyList[np.argmax(weightList)]
        cen2 = cen1 - 0.05 if bz_point == "G" else cen1 - 0.15
        params_fit = model.make_params(
            amp1=1.57, cen1=cen1, gam1=0.03,
            amp2=0.41, cen2=cen2, gam2=0.03,
            sig=0.07,
        )
        params_fit["sig"].set(min=1e-6, max=50)
        params_fit["gam1"].set(min=1e-6, max=50)
        params_fit["gam2"].set(min=1e-6, max=50)
        params_fit["amp1"].set(min=0)
        params_fit["amp2"].set(min=0)

        result = model.fit(weightList, params_fit, x=energyList)
        amp1 = result.best_values["amp1"]
        amp2 = result.best_values["amp2"]
        cen1 = result.best_values["cen1"]
        cen2 = result.best_values["cen2"]

        if result.success and amp1 > 1e-3 and amp2 > 1e-3 and result.redchi < 1e-2 and amp1 > amp2:
            return (cen1, cen2), True
        return (0, 0), False

    def compute_ldos(self, evals, evecs, r_list, e_list, k_flat, spreadE):
        n_shells = self.config["n_shells"]
        n_cells = self.config["n_cells"]
        theta = self.config["theta_deg"] / 180 * np.pi

        rPts = r_list.shape[0]
        ePts = len(e_list)
        kPts = k_flat.shape[0]
        LDOS = np.zeros((rPts, ePts))

        lu = MoireGeometry.lu_table(n_shells)
        G_M = self.geometry.reciprocal_vectors()
        Kbs = np.zeros((n_cells, 2))
        for i in range(n_cells):
            Kbs[i] = G_M[1] * lu[i][0] + G_M[2] * lu[i][1]

        ig = np.arange(n_cells)[np.newaxis, :]
        alpha = np.arange(44)[:, np.newaxis]
        ind = (alpha % 22) + ig * 22 + n_cells * 22 * (alpha // 22)

        for ik in range(kPts):
            evals_k = evals[ik]
            evecs_k = evecs[ik]
            kGs = Kbs + k_flat[ik]
            phases = np.exp(1j * r_list @ kGs.T)[np.newaxis, :, :]

            for n, En in enumerate(evals_k):
                coeffs = evecs_k[ind, n]
                coeffs_all = coeffs[:, np.newaxis, :]
                psi_alpha = np.sum(phases * coeffs_all, axis=-1)
                psi_r_all = np.sum(np.abs(psi_alpha) ** 2, axis=0)
                lorentz_matrix = spreadE / (np.pi * ((e_list - En) ** 2 + spreadE ** 2))
                LDOS += psi_r_all[:, None] * lorentz_matrix[None, :] / kPts

        return LDOS
