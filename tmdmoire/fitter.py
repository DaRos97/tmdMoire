"""Parameter fitting: chi-squared objective and dual annealing minimization.

The ``ParameterFitter`` class encapsulates the full monolayer fitting
procedure. It computes a weighted chi-squared objective function that
combines band dispersion matching with physical constraints (orbital
character, parameter distance from DFT, band gap, etc.), then minimizes
it using scipy's dual annealing algorithm chained with Nelder-Mead for
local refinement.

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
from pathlib import Path
from .constants import (
    ORBITAL_CHARACTER, TVB2, TVB4, IND_ILC, ze_i, z2_i, xe_i, ye_i, x2_i, xy_i,
)
from .material import TMDMaterial
from .arpes_data import ARPESData


class ParameterFitter:
    """Fits tight-binding parameters to ARPES data via dual annealing + Nelder-Mead.

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
    _gap_DFT : float
        Precomputed DFT band gap at K (constant throughout fitting).

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
    >>> result = fitter.run(maxiter=int(1e4), seed=42)
    """

    def __init__(self, material: TMDMaterial, arpes_data: ARPESData, config: dict):
        self.material = material
        self.arpes_data = arpes_data
        self.config = config
        self._gap_DFT = self._compute_DFT_gap()

    def chi2(self, params_tb: np.ndarray, HSO: np.ndarray, SOC_pars: np.ndarray,
             return_energy: bool = False) -> float:
        """Compute chi-squared for a given set of TB parameters (excluding SOC).

        Parameters
        ----------
        params_tb : np.ndarray
            Tight-binding parameters (41 values, excluding SOC).
        HSO : np.ndarray
            Pre-computed 22×22 SOC Hamiltonian.
        SOC_pars : np.ndarray
            SOC parameters [L_W, L_S].
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
        args_H = (hopping, epsilon, HSO, offset)

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

        # Band distance term: global normalization over all valid points
        chi2_band_distance = 0.0
        total_valid = 0
        special_indices = [0, np.argmax(self.arpes_data.fit_data[:, 3]),
                           np.argmin(self.arpes_data.fit_data[:, 4]),
                           self.arpes_data.fit_data.shape[0] - 1]
        weights = np.ones(self.arpes_data.fit_data.shape[0])
        weights[special_indices] = K6
        for ib in range(nbands):
            valid = ~np.isnan(self.arpes_data.fit_data[:, 3 + ib])
            chi2_band_distance += np.sum(
                np.absolute(
                    (tb_en[ib] - self.arpes_data.fit_data[:, 3 + ib]) * weights
                )[valid] ** 2
            )
            total_valid += valid.sum()
        chi2_band_distance /= total_valid

        # K1: parameter distance from DFT
        K1_par_dis = self.material.parameter_distance(full_params)

        # K2: orbital band content at M (mean per orbital per band)
        k_pts_bc = np.array([self.arpes_data.M, np.zeros(2), self.arpes_data.K])
        Ham_bc = self._build_hamiltonian(k_pts_bc, args_H)
        evals_M, evecs_M = np.linalg.eigh(Ham_bc[0])
        bandsM = TVB4 if self.material.name == "WSe2" else TVB2
        K2_M = np.sum(np.absolute(evecs_M[IND_ILC, :][:, bandsM]) ** 2) / (len(IND_ILC) * len(bandsM))

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
                  + abs(occ_d2_tvb1 - K_d2_tvb1) + abs(occ_d2_tvb2 - K_d2_tvb2)) / 8

        # K4: conduction band minimum at K
        # Continuous penalty: squared relative distance of CBM from K point
        cbm_idx = np.argmin(cond_en)
        cbm_k = self.arpes_data.fit_data[cbm_idx, 0]
        k_mod = np.linalg.norm(self.arpes_data.K)
        K4_band_min = ((cbm_k - k_mod) / k_mod) ** 2

        # K5: band gap at K vs DFT (relative error)
        gap_DFT = self._gap_DFT
        gap_p = evals_K[14] - evals_K[13]
        K5_gap = abs(gap_DFT - gap_p) / gap_DFT

        result = chi2_band_distance + (K1 * K1_par_dis + K2 * K2_M
                                       + K3 * K3_DFT + K4 * K4_band_min + K5 * K5_gap)

        return result

    def chi2_full(self, params_full: np.ndarray, return_energy: bool = False) -> float:
        """Wrapper that includes SOC parameters in the fit.

        Parameters
        ----------
        params_full : np.ndarray
            Full 43-parameter array (including SOC).
        return_energy : bool
            If True, return band energies instead of chi-squared.

        Returns
        -------
        float or np.ndarray
            Chi-squared value or band energies.
        """
        SOC_pars = params_full[-2:]
        HSO = self.material.build_soc_hamiltonian(SOC_pars)
        return self.chi2(params_full[:-2], HSO, SOC_pars, return_energy)

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

    def run(self, initial_params: np.ndarray | None = None, maxiter: int = 3000, seed: int = 42) -> dict:
        """Run global optimization via dual annealing + Nelder-Mead refinement.

        Dual annealing explores the full parameter space (reproducible with seed),
        then Nelder-Mead refines the best solution locally.

        Parameters
        ----------
        initial_params : np.ndarray, optional
            Starting point for the annealing trajectory. Defaults to DFT params.
        maxiter : int
            Maximum dual annealing iterations.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``x``: optimized parameter array
            - ``fun``: final chi-squared value
            - ``nfev``: total number of function evaluations
            - ``method``: optimization method used
        """
        from scipy.optimize import dual_annealing

        if initial_params is None:
            initial_params = self.material.dft_params

        if self.config["Bs"][-1] == 0:
            HSO = self.material.build_soc_hamiltonian()
            args_chi2 = (HSO, self.material.dft_params[-2:], False)
            bounds = self.get_bounds()[:-2]
            initial_params = initial_params[:-2]
            func = lambda x: self.chi2(x, *args_chi2)
        else:
            args_chi2 = (False,)
            bounds = self.get_bounds()
            func = lambda x: self.chi2_full(x, *args_chi2)

        result = dual_annealing(
            func,
            bounds=bounds,
            x0=initial_params,
            seed=seed,
            maxiter=maxiter,
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "options": {"adaptive": True, "fatol": 1e-3, "maxiter": 50},
            },
        )
        return {"x": result.x, "fun": result.fun, "nfev": result.nfev, "method": "dual_annealing", "seed": seed}

    def compute_bands(self, params: np.ndarray | None = None) -> np.ndarray:
        """Compute TB band energies at ARPES k-points.

        Parameters
        ----------
        params : np.ndarray, optional
            43-parameter array. Defaults to DFT params.

        Returns
        -------
        np.ndarray
            Shape ``(6, n_kpts)`` — top 6 valence band energies.
        """
        if params is None:
            params = self.material.dft_params

        SOC_pars = params[-2:]
        HSO = self.material.build_soc_hamiltonian(SOC_pars)
        return self.chi2(params[:-2], HSO, SOC_pars, return_energy=True)

    def save(self, result: dict, output_dir: str = "Data") -> Path:
        """Save fitting result to an npz file.

        Parameters
        ----------
        result : dict
            Dictionary from ``run()`` with keys: x, fun, nfev, method.
            Must also contain ``idx`` for the filename.
        output_dir : str
            Directory to save the file.

        Returns
        -------
        Path
            Path to the saved file.
        """
        config = self.config
        Ks = config["Ks"]
        params = result["x"]

        constraints = self._compute_constraint_breakdown(params)
        tb_en = self.compute_bands(params)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fn = out_dir / f"fit_{self.material.name}_idx{result['idx']}.npz"
        np.savez(fn,
                 params=params,
                 chi2=result["fun"],
                 nfev=result["nfev"],
                 Ks=Ks,
                 Bs=config["Bs"],
                 boundType=config["boundType"],
                 seed=result.get("seed", 42),
                 material=self.material.name,
                 chi2_band=constraints["chi2_band"],
                 K1_val=constraints["K1"],
                 K2_val=constraints["K2"],
                 K3_val=constraints["K3"],
                 K4_val=constraints["K4"],
                 K5_val=constraints["K5"],
                 tb_en=tb_en,
                 k_path=self.arpes_data.fit_data[:, 0],
        )
        return fn

    def _compute_constraint_breakdown(self, params: np.ndarray) -> dict:
        """Compute individual constraint values at given parameters.

        Returns a dict with keys: chi2_band, K1, K2, K3, K4, K5.
        These are the raw (unweighted) constraint terms.
        """
        K1, K2, K3, K4, K5, K6 = self.config["Ks"]

        if self.config["Bs"][-1] == 0:
            full_params = np.append(params, self.material.dft_params[-2:])
        else:
            full_params = params

        hopping = self.material.build_hopping_matrices(full_params)
        epsilon = self.material.build_onsite_energies(full_params)
        offset = full_params[-3]
        HSO = self.material.build_soc_hamiltonian(full_params[-2:])
        args_H = (hopping, epsilon, HSO, offset)

        k_pts = self.arpes_data.fit_data[:, 1:3]
        all_H = self._build_hamiltonian(k_pts, args_H)
        nbands = 6
        tb_en = np.zeros((nbands, k_pts.shape[0]))
        cond_en = np.zeros(k_pts.shape[0])
        for i in range(k_pts.shape[0]):
            energies = la.eigvalsh(all_H[i])
            tb_en[:, i] = energies[14 - nbands:14][::-1]
            cond_en[i] = energies[14]

        # Band distance (unweighted by K6 for the stored value)
        chi2_band_distance = 0.0
        total_valid = 0
        for ib in range(nbands):
            valid = ~np.isnan(self.arpes_data.fit_data[:, 3 + ib])
            chi2_band_distance += np.sum(
                np.absolute(tb_en[ib] - self.arpes_data.fit_data[:, 3 + ib])[valid] ** 2
            )
            total_valid += valid.sum()
        chi2_band_distance /= total_valid

        # K1
        K1_par_dis = self.material.parameter_distance(full_params)

        # K2
        k_pts_bc = np.array([self.arpes_data.M, np.zeros(2), self.arpes_data.K])
        Ham_bc = self._build_hamiltonian(k_pts_bc, args_H)
        evals_M, evecs_M = np.linalg.eigh(Ham_bc[0])
        bandsM = TVB4 if self.material.name == "WSe2" else TVB2
        K2_M = np.sum(np.absolute(evecs_M[IND_ILC, :][:, bandsM]) ** 2) / (len(IND_ILC) * len(bandsM))

        # K3
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
                  + abs(occ_d2_tvb1 - K_d2_tvb1) + abs(occ_d2_tvb2 - K_d2_tvb2)) / 8

        # K4
        cbm_idx = np.argmin(cond_en)
        cbm_k = self.arpes_data.fit_data[cbm_idx, 0]
        k_mod = np.linalg.norm(self.arpes_data.K)
        K4_band_min = ((cbm_k - k_mod) / k_mod) ** 2

        # K5
        gap_p = evals_K[14] - evals_K[13]
        K5_gap = abs(self._gap_DFT - gap_p) / self._gap_DFT

        return {
            "chi2_band": chi2_band_distance,
            "K1": K1_par_dis,
            "K2": K2_M,
            "K3": K3_DFT,
            "K4": K4_band_min,
            "K5": K5_gap,
        }

    def _compute_DFT_gap(self) -> float:
        """Precompute the DFT band gap at K (constant throughout fitting)."""
        DFT = self.material.dft_params
        args = (self.material.build_hopping_matrices(DFT),
                self.material.build_onsite_energies(DFT),
                self.material.build_soc_hamiltonian(DFT),
                DFT[-3])
        Ham = self._build_hamiltonian(np.array([self.arpes_data.K]), args)
        ev = np.linalg.eigvalsh(Ham[0])
        return ev[14] - ev[13]

    def _build_hamiltonian(self, k_points, args_H):
        """Build the monolayer Hamiltonian at given k-points (internal)."""
        from .hamiltonian import MonolayerHamiltonian
        ham = MonolayerHamiltonian(self.material)
        return ham.build(k_points, *args_H)
