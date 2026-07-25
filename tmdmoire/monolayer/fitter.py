"""Parameter fitting: chi-squared objective and Nelder-Mead minimization.

The ``ParameterFitter`` class encapsulates the full monolayer fitting
procedure. It computes a weighted chi-squared objective function that
combines band dispersion matching with physical constraints (orbital
character, parameter distance from DFT, band gap, etc.), then minimizes
it using scipy's Nelder-Mead algorithm starting from DFT parameters.

All constraint computation is centralized in
:meth:`_compute_constraint_breakdown`, which serves as the single source
of truth. Both :meth:`chi2` (objective function) and :meth:`save`
(result storage) delegate to it.

Chi-squared terms:
    - Band distance: Σ(TB band − ARPES band)² (always included)
    - K₁: Parameter distance from DFT values
    - K₂: Orbital band content at M point
    - K₃: Orbital occupation at Γ and K vs DFT
    - K₄: Conduction band minimum position at K
    - K₅: Band gap at K vs DFT gap
    - K₆: Weight multiplier for high-symmetry points

Stored ``band_K6`` is the K6-weighted mean-squared band distance

(the version that appears in the objective function). A separate
``band_dist`` field (pure band distance, no K6 weights)

is stored for cross-comparison across grid points with different K6.
"""
import numpy as np
import scipy.linalg as la
from pathlib import Path
from ..constants import (
    ORBITAL_CHARACTER, TVB2, TVB4, IND_ILC, ze_i, z2_i, xe_i, ye_i, x2_i, xy_i,
)
from ..material import TMDMaterial
from .data import MonolayerData
from .hamiltonian import MonolayerHamiltonian


class ParameterFitter:
    """Fits tight-binding parameters to ARPES data via Nelder-Mead minimization.

    Parameters
    ----------
    material : TMDMaterial
        The TMD material to fit (WSe2 or WS2).
    data : MonolayerData
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
    >>> data = MonolayerData("WSe2", master_folder="/path/", pts=91)
    >>> config = {
    ...     "Ks": (1e-5, 0.5, 1.0, 1.0, 0.5, 5.0),
    ...     "boundType": "absolute",
    ...     "Bs": (5, 2, 4, 1, 0),
    ... }
    >>> fitter = ParameterFitter(material, data, config)
    >>> result = fitter.run(seed=42)
    """

    def __init__(self, material: TMDMaterial, data: MonolayerData, config: dict, idx: int = 0):
        self.material = material
        self.data = data
        self.config = config
        self._gap_DFT = self._compute_DFT_gap()
        self._idx = idx
        self._min_chi2 = np.inf
        self._eval_step = 0
        self._output_dir = None

    def chi2(self, params_tb: np.ndarray, HSO: np.ndarray, SOC_pars: np.ndarray,
             return_energy: bool = False) -> float:
        """Compute chi-squared for a given set of TB parameters (excluding SOC)."""
        full_params = np.append(params_tb, SOC_pars)

        if return_energy:
            hopping = self.material.build_hopping_matrices(full_params)
            epsilon = self.material.build_onsite_energies(full_params)
            offset = full_params[-3]
            args_H = (hopping, epsilon, HSO, offset)

            k_pts = self.data.fit_data[:, 1:3]
            all_H = self._build_hamiltonian(k_pts, args_H)
            nbands = 6
            tb_en = np.zeros((nbands, k_pts.shape[0]))
            for i in range(k_pts.shape[0]):
                energies = la.eigvalsh(all_H[i])
                tb_en[:, i] = energies[14 - nbands:14][::-1]
            return tb_en

        breakdown = self._compute_constraint_breakdown(full_params)
        K1, K2, K3, K4, K5, K6 = self.config["Ks"]
        result = (breakdown["chi2_band_weighted"]
                  + K1 * breakdown["K1"] + K2 * breakdown["K2"]
                  + K3 * breakdown["K3"] + K4 * breakdown["K4"]
                  + K5 * breakdown["K5"])

        self._eval_step += 1
        if result < self._min_chi2:
            self._min_chi2 = result
            self._best_params = params_tb.copy()
            if self._output_dir is not None:
                chi2_elements = np.array([
                    breakdown["chi2_band_weighted"],
                    breakdown["K1"], breakdown["K2"],
                    breakdown["K3"], breakdown["K4"], breakdown["K5"],
                ])
                temp_fn = Path(self._output_dir) / f"temp_best_{self._idx}.npz"
                Path(self._output_dir).mkdir(parents=True, exist_ok=True)
                np.savez(temp_fn, elements=chi2_elements, pars=full_params)

        return result

    def chi2_full(self, params_full: np.ndarray, return_energy: bool = False) -> float:
        """Wrapper that includes SOC parameters in the fit."""
        SOC_pars = params_full[-2:]
        HSO = self.material.build_soc_hamiltonian(SOC_pars)
        return self.chi2(params_full[:-2], HSO, SOC_pars, return_energy)

    def get_bounds(self) -> list[tuple]:
        """Generate parameter bounds based on the configured bound type."""
        bt = self.config["boundType"]
        Bs = self.config["Bs"]
        if bt == "relative":
            return self.material.get_bounds_relative(*Bs)
        elif bt == "absolute":
            return self.material.get_bounds_absolute(*Bs)
        raise ValueError(f"Unknown bound type: {bt}")

    def run(self, seed: int = 42,
            output_dir: str = "Data") -> dict:
        """Run Nelder-Mead minimization starting from DFT parameters."""
        from scipy.optimize import minimize

        self._output_dir = output_dir
        self._min_chi2 = np.inf
        self._eval_step = 0

        nm_maxiter = self.config.get("optimizer", {}).get("nm_maxiter", 500)
        nm_fatol = self.config.get("optimizer", {}).get("nm_fatol", 1e-4)

        if self.config["Bs"][-1] == 0:
            HSO = self.material.build_soc_hamiltonian()
            args_chi2 = (HSO, self.material.dft_params[-2:])
            bounds = self.get_bounds()[:-2]
            x0 = self.material.dft_params[:-2]
        else:
            args_chi2 = ()
            bounds = self.get_bounds()
            x0 = self.material.dft_params

        func = lambda x: self.chi2(x, *args_chi2)

        result = minimize(
            func,
            x0=x0,
            bounds=bounds,
            method="Nelder-Mead",
            options={
                "adaptive": True,
                "fatol": nm_fatol,
                "maxiter": nm_maxiter,
                "disp": True,
            },
        )

        return {"x": result.x, "fun": result.fun, "nfev": result.nfev,
                "method": "Nelder-Mead", "seed": seed}

    def compute_bands(self, params: np.ndarray | None = None) -> np.ndarray:
        """Compute TB band energies at ARPES k-points."""
        if params is None:
            params = self.material.dft_params

        if self.config["Bs"][-1] == 0 and params.shape[0] == 41:
            full_params = np.append(params, self.material.dft_params[-2:])
        else:
            full_params = params

        SOC_pars = full_params[-2:]
        HSO = self.material.build_soc_hamiltonian(SOC_pars)
        return self.chi2(full_params[:-2], HSO, SOC_pars, return_energy=True)

    def save(self, result: dict, output_dir: str = "Data") -> Path:
        """Save fitting result to an npz file."""
        config = self.config
        Ks = config["Ks"]
        params = result["x"]

        constraints = self._compute_constraint_breakdown(params)
        tb_en = self.compute_bands(params)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fn = out_dir / f"fit_idx{result['idx']}.npz"
        np.savez(fn,
                 params=params,
                 chi2=result["fun"],
                 nfev=result["nfev"],
                 Ks=Ks,
                 Bs=config["Bs"],
                 boundType=config["boundType"],
                 seed=result.get("seed", 42),
                 material=self.material.name,
                  band_K6=constraints["chi2_band_weighted"],
                  band_dist=constraints["chi2_band"],
                 K1_val=constraints["K1"],
                 K2_val=constraints["K2"],
                 K3_val=constraints["K3"],
                 K4_val=constraints["K4"],
                 K5_val=constraints["K5"],
                 tb_en=tb_en,
                 k_path=self.data.fit_data[:, 0],
        )
        return fn

    def _compute_constraint_breakdown(self, params: np.ndarray) -> dict:
        """Compute all constraint terms at given parameters."""
        K1, K2, K3, K4, K5, K6 = self.config["Ks"]

        full_params = params if params.shape[0] == 43 else np.append(params, self.material.dft_params[-2:])

        hopping = self.material.build_hopping_matrices(full_params)
        epsilon = self.material.build_onsite_energies(full_params)
        offset = full_params[-3]
        HSO = self.material.build_soc_hamiltonian(full_params[-2:])
        args_H = (hopping, epsilon, HSO, offset)

        k_pts = self.data.fit_data[:, 1:3]
        all_H = self._build_hamiltonian(k_pts, args_H)
        nbands = 6
        tb_en = np.zeros((nbands, k_pts.shape[0]))
        cond_en = np.zeros(k_pts.shape[0])
        for i in range(k_pts.shape[0]):
            energies = la.eigvalsh(all_H[i])
            tb_en[:, i] = energies[14 - nbands:14][::-1]
            cond_en[i] = energies[14]

        chi2_band_unweighted = 0.0
        special_indices = [0, np.argmax(self.data.fit_data[:, 3]),
                           np.argmin(self.data.fit_data[:, 4]),
                           self.data.fit_data.shape[0] - 1]
        weights = np.ones(self.data.fit_data.shape[0])
        weights[special_indices] = K6
        chi2_band_weighted = 0.0
        for ib in range(nbands):
            valid = ~np.isnan(self.data.fit_data[:, 3 + ib])
            tb_diff = tb_en[ib] - self.data.fit_data[:, 3 + ib]
            chi2_band_unweighted += np.sum(np.absolute(tb_diff[valid]) ** 2
                                           ) / valid.sum()
            chi2_band_weighted += np.sum(
                np.absolute((tb_diff * weights)[valid]) ** 2
            ) / valid.sum()

        K1_par_dis = self.material.parameter_distance(full_params)

        k_pts_bc = np.array([self.data.M, np.zeros(2), self.data.K])
        Ham_bc = self._build_hamiltonian(k_pts_bc, args_H)
        evals_M, evecs_M = la.eigh(Ham_bc[0])
        bands_m = TVB4 if self.material.name == "WSe2" else TVB2
        K2_M = np.sum(np.absolute(evecs_M[IND_ILC, :][:, bands_m]) ** 2)
        if self.material.name == "WS2":
            K2_M *= 2

        evals_G, evecs_G = la.eigh(Ham_bc[1])
        occ_ze, occ_z2 = ORBITAL_CHARACTER[self.material.name]["G"]
        G_ze_tvb1 = np.absolute(evecs_G[ze_i, 13]) ** 2 + np.absolute(evecs_G[ze_i + 11, 13]) ** 2
        G_ze_tvb2 = np.absolute(evecs_G[ze_i, 12]) ** 2 + np.absolute(evecs_G[ze_i + 11, 12]) ** 2
        G_z2_tvb1 = np.absolute(evecs_G[z2_i, 13]) ** 2 + np.absolute(evecs_G[z2_i + 11, 13]) ** 2
        G_z2_tvb2 = np.absolute(evecs_G[z2_i, 12]) ** 2 + np.absolute(evecs_G[z2_i + 11, 12]) ** 2

        evals_K, evecs_K = la.eigh(Ham_bc[2])
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

        cbm_idx = np.argmin(cond_en)
        cbm_k = self.data.fit_data[cbm_idx, 0]
        if abs(cbm_k - np.linalg.norm(self.data.K)) < 1e-3:
            K4_band_min = 0
        else:
            K4_band_min = 1

        gap_p = evals_K[14] - evals_K[13]
        K5_gap = abs(self._gap_DFT - gap_p)

        return {
            "chi2_band": chi2_band_unweighted,
            "chi2_band_weighted": chi2_band_weighted,
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
        Ham = self._build_hamiltonian(np.array([self.data.K]), args)
        ev = la.eigvalsh(Ham[0])
        return ev[14] - ev[13]

    def _build_hamiltonian(self, k_points, args_H):
        """Build the monolayer Hamiltonian at given k-points (internal)."""
        ham = MonolayerHamiltonian(self.material)
        return ham.build(k_points, *args_H)
