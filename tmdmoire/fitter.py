"""Parameter fitting: chi-squared objective and Nelder-Mead minimization.

The ``ParameterFitter`` class encapsulates the full monolayer fitting
procedure. It computes a weighted chi-squared objective function that
combines band dispersion matching with physical constraints (orbital
character, parameter distance from DFT, band gap, etc.), then minimizes
it using scipy's Nelder-Mead algorithm.

Chi-squared terms:
    - Band distance: Σ(TB band − ARPES band)² (always included)
    - K₁: Parameter distance from DFT values
    - K₂: Orbital band content at M point
    - K₃: Orbital occupation at Γ and K vs DFT
    - K₄: Conduction band minimum position at K
    - K₅: Band gap at K vs DFT gap
    - K₆: Weight multiplier for high-symmetry points
"""
import numpy as np
import scipy.linalg as la
from .constants import (
    ORBITAL_CHARACTER, TVB2, TVB4, IND_ILC, ze_i, z2_i, xe_i, ye_i, x2_i, xy_i,
)
from .material import TMDMaterial
from .arpes_data import ARPESData


class ParameterFitter:
    """Fits tight-binding parameters to ARPES data via Nelder-Mead minimization.

    Parameters
    ----------
    material : TMDMaterial
        The TMD material to fit (WSe2 or WS2).
    arpes_data : ARPESData
        Experimental ARPES data for the material.
    config : dict
        Fitting configuration with keys:
        - ``Ks``: tuple of 6 constraint weights (K₁–K₆)
        - ``boundType``: "relative" or "absolute"
        - ``Bs``: tuple of bound parameters

    Attributes
    ----------
    min_chi2 : float
        Best chi-squared value found so far.
    evaluation_step : int
        Number of chi-squared evaluations performed.

    Examples
    --------
    >>> material = TMDMaterial("WSe2")
    >>> arpes = ARPESData("WSe2", master_folder="/path/", pts=91)
    >>> config = {
    ...     "Ks": (1e-5, 0.5, 1.0, 1.0, 0.5, 5.0),
    ...     "boundType": "absolute",
    ...     "Bs": (5, 2, 4, 1, 0),
    ... }
    >>> fitter = ParameterFitter(material, arpes, config)
    >>> result = fitter.run(material.dft_params, max_eval=int(1e4))
    """

    def __init__(self, material: TMDMaterial, arpes_data: ARPESData, config: dict):
        self.material = material
        self.arpes_data = arpes_data
        self.config = config
        self.min_chi2 = 1e5
        self.evaluation_step = 0

    def chi2(self, params_tb: np.ndarray, HSO: np.ndarray, SOC_pars: np.ndarray,
             max_eval: int, return_energy: bool = False) -> float:
        """Compute chi-squared for a given set of TB parameters (excluding SOC).

        Parameters
        ----------
        params_tb : np.ndarray
            Tight-binding parameters (41 values, excluding SOC).
        HSO : np.ndarray
            Pre-computed 22×22 SOC Hamiltonian.
        SOC_pars : np.ndarray
            SOC parameters [L_W, L_S].
        max_eval : int
            Maximum number of evaluations before raising an error.
        return_energy : bool
            If True, return band energies instead of chi-squared.

        Returns
        -------
        float or np.ndarray
            Chi-squared value, or band energies if ``return_energy=True``.
        """
        K1, K2, K3, K4, K5, K6 = self.config["Ks"]
        full_params = np.append(params_tb, SOC_pars)

        hopping = self.material.build_hopping_matrices(full_params)
        epsilon = self.material.build_onsite_energies(full_params)
        offset = full_params[-3]
        args_H = (hopping, epsilon, HSO, self.material.lattice_constant, offset)

        k_pts = self.arpes_data.fit_data[:, 1:3]
        all_H = self._build_hamiltonian(k_pts, args_H)
        nbands = 6
        tb_en = np.zeros((nbands, k_pts.shape[0]))
        cond_en = np.zeros(k_pts.shape[0])
        for i in range(k_pts.shape[0]):
            energies = la.eigvalsh(all_H[i])
            tb_en[:, i] = energies[14 - nbands:14][::-1]
            cond_en[i] = energies[14]

        if return_energy:
            return tb_en

        # Band distance term
        chi2_band_distance = 0
        special_indices = [0, np.argmax(self.arpes_data.fit_data[:, 3]),
                           np.argmin(self.arpes_data.fit_data[:, 4]),
                           self.arpes_data.fit_data.shape[0] - 1]
        weights = np.ones(self.arpes_data.fit_data.shape[0])
        weights[special_indices] = K6
        for ib in range(nbands):
            chi2_band_distance += np.sum(
                np.absolute(
                    ((tb_en[ib] - self.arpes_data.fit_data[:, 3 + ib]) * weights)
                    [~np.isnan(self.arpes_data.fit_data[:, 3 + ib])]
                ) ** 2
            ) / self.arpes_data.fit_data[~np.isnan(self.arpes_data.fit_data[:, 3 + ib])].shape[0]

        # K1: parameter distance from DFT
        K1_par_dis = self.material.parameter_distance(full_params)

        # K2: orbital band content at M
        k_pts_bc = np.array([self.arpes_data.M, np.zeros(2), self.arpes_data.K])
        Ham_bc = self._build_hamiltonian(k_pts_bc, args_H)
        evals_M, evecs_M = np.linalg.eigh(Ham_bc[0])
        bandsM = TVB4 if self.material.name == "WSe2" else TVB2
        K2_M = np.sum(np.absolute(evecs_M[IND_ILC, :][:, bandsM]) ** 2)
        if self.material.name == "WS2":
            K2_M *= 2

        # K3: orbital occupation at Gamma and K
        evals_G, evecs_G = np.linalg.eigh(Ham_bc[1])
        occ_ze, occ_z2 = ORBITAL_CHARACTER[self.material.name]["G"]
        G_ze_tvb1 = np.absolute(evecs_G[ze_i, 13]) ** 2 + np.absolute(evecs_G[ze_i + 11, 13]) ** 2
        G_ze_tvb2 = np.absolute(evecs_G[ze_i, 12]) ** 2 + np.absolute(evecs_G[ze_i + 11, 12]) ** 2
        G_z2_tvb1 = np.absolute(evecs_G[z2_i, 13]) ** 2 + np.absolute(evecs_G[z2_i + 11, 13]) ** 2
        G_z2_tvb2 = np.absolute(evecs_G[z2_i, 12]) ** 2 + np.absolute(evecs_G[z2_i + 11, 12]) ** 2

        evals_K, evecs_K = np.linalg.eigh(Ham_bc[2])
        occ_p1_tvb1, occ_p1_tvb2, occ_d2_tvb1, occ_d2_tvb2 = ORBITAL_CHARACTER[self.material.name]["K"]
        K_p1_tvb1 = (np.absolute(-1 / np.sqrt(2) * (evecs_K[xe_i, 13] - 1j * evecs_K[ye_i, 13])) ** 2
                     + np.absolute(-1 / np.sqrt(2) * (evecs_K[xe_i + 11, 13] - 1j * evecs_K[ye_i + 11, 13])) ** 2)
        K_p1_tvb2 = (np.absolute(-1 / np.sqrt(2) * (evecs_K[xe_i, 12] - 1j * evecs_K[ye_i, 12])) ** 2
                     + np.absolute(-1 / np.sqrt(2) * (evecs_K[xe_i + 11, 12] - 1j * evecs_K[ye_i + 11, 12])) ** 2)
        K_d2_tvb1 = (np.absolute(1 / np.sqrt(2) * (evecs_K[x2_i, 13] - 1j * evecs_K[xy_i, 13])) ** 2
                     + np.absolute(1 / np.sqrt(2) * (evecs_K[x2_i + 11, 13] - 1j * evecs_K[xy_i + 11, 13])) ** 2)
        K_d2_tvb2 = (np.absolute(1 / np.sqrt(2) * (evecs_K[x2_i, 12] - 1j * evecs_K[xy_i, 12])) ** 2
                     + np.absolute(1 / np.sqrt(2) * (evecs_K[x2_i + 11, 12] - 1j * evecs_K[xy_i + 11, 12])) ** 2)

        K3_DFT = (abs(occ_ze - G_ze_tvb1) + abs(occ_ze - G_ze_tvb2)
                  + abs(occ_z2 - G_z2_tvb1) + abs(occ_z2 - G_z2_tvb2)
                  + abs(occ_p1_tvb1 - K_p1_tvb1) + abs(occ_p1_tvb2 - K_p1_tvb2)
                  + abs(occ_d2_tvb1 - K_d2_tvb1) + abs(occ_d2_tvb2 - K_d2_tvb2))

        # K4: conduction band minimum at K
        if abs(self.arpes_data.fit_data[np.argmin(cond_en), 0] - self.arpes_data.K[0]) < 1e-3:
            K4_band_min = 0
        else:
            K4_band_min = 1

        # K5: band gap at K vs DFT
        DFT_params = self.material.dft_params
        args_H_DFT = (self.material.build_hopping_matrices(DFT_params),
                      self.material.build_onsite_energies(DFT_params),
                      self.material.build_soc_hamiltonian(DFT_params),
                      self.material.lattice_constant, DFT_params[-3])
        Ham_DFT = self._build_hamiltonian(np.array([self.arpes_data.K]), args_H_DFT)
        evals_DFT = np.linalg.eigvalsh(Ham_DFT[0])
        gap_DFT = evals_DFT[14] - evals_DFT[13]
        gap_p = evals_K[14] - evals_K[13]
        K5_gap = abs(gap_DFT - gap_p)

        result = chi2_band_distance + (K1 * K1_par_dis + K2 * K2_M
                                       + K3 * K3_DFT + K4 * K4_band_min + K5 * K5_gap)

        self.evaluation_step += 1
        if result < self.min_chi2:
            self.min_chi2 = result

        if self.evaluation_step > max_eval:
            raise RuntimeError(f"Reached max number of evaluations ({max_eval})")

        return result

    def chi2_full(self, params_full: np.ndarray, max_eval: int, return_energy: bool = False) -> float:
        """Wrapper that includes SOC parameters in the fit.

        Parameters
        ----------
        params_full : np.ndarray
            Full 43-parameter array (including SOC).
        max_eval : int
            Maximum number of evaluations.
        return_energy : bool
            If True, return band energies instead of chi-squared.

        Returns
        -------
        float or np.ndarray
            Chi-squared value or band energies.
        """
        SOC_pars = params_full[-2:]
        HSO = self.material.build_soc_hamiltonian(SOC_pars)
        return self.chi2(params_full[:-2], HSO, SOC_pars, max_eval, return_energy)

    def get_bounds(self) -> list[tuple]:
        """Generate parameter bounds based on the configured bound type.

        Returns
        -------
        list[tuple]
            List of (lower, upper) bounds for each parameter.
        """
        bt = self.config["boundType"]
        Bs = self.config["Bs"]
        if bt == "relative":
            return self.material.get_bounds_relative(*Bs)
        elif bt == "absolute":
            return self.material.get_bounds_absolute(*Bs)
        raise ValueError(f"Unknown bound type: {bt}")

    def run(self, initial_params: np.ndarray, max_eval: int = 5e6) -> dict:
        """Run the Nelder-Mead minimization.

        Parameters
        ----------
        initial_params : np.ndarray
            Starting parameter values (typically DFT params).
        max_eval : int
            Maximum number of chi-squared evaluations.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``x``: optimized parameter array
            - ``fun``: final chi-squared value
        """
        from scipy.optimize import minimize

        if self.config["Bs"][-1] == 0:
            # SOC bounds set to 0: fit only TB params, HSO fixed from DFT
            HSO = self.material.build_soc_hamiltonian()
            args_chi2 = (HSO, self.material.dft_params[-2:], max_eval, False)
            bounds = self.get_bounds()[:-2]
            func = lambda x: self.chi2(x, *args_chi2)
        else:
            # Fit all parameters including SOC
            args_chi2 = (max_eval, False)
            bounds = self.get_bounds()
            func = lambda x: self.chi2_full(x, *args_chi2)

        result = minimize(func, args=(), x0=initial_params, bounds=bounds,
                          method="Nelder-Mead",
                          options={"disp": True, "adaptive": True, "fatol": 1e-4, "maxiter": 1e6})
        return {"x": result.x, "fun": result.fun}

    def _build_hamiltonian(self, k_points, args_H):
        """Build the monolayer Hamiltonian at given k-points (internal)."""
        from .hamiltonian import MonolayerHamiltonian
        ham = MonolayerHamiltonian(self.material)
        return ham.build(k_points, *args_H)
