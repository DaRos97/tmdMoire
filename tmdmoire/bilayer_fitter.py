"""Bilayer interlayer coupling fitter.

Fits 4 interlayer hopping parameters (w1p, w1d, w2p, w2d) to reproduce
the 3 top valence bands from bilayer ARPES data along the Γ–K path.

Uses scipy.optimize.minimize (Nelder-Mead) from multiple starting points.
The stacking type (P or AP) controls the interlayer phase and is fixed
during fitting. Parameter bounds are enforced at [-10, 10] eV via clipping
with a penalty for out-of-bounds excursions.
"""
import numpy as np
import scipy.optimize as opt
from .material import TMDMaterial
from .bilayer_data import BilayerData
from .hamiltonian import MoireHamiltonian
from .moire_geometry import MoireGeometry
from .constants import ENERGY_OFFSETS


class BilayerFitter:
    """Fits interlayer coupling parameters to bilayer ARPES data.

    Parameters
    ----------
    wse2 : TMDMaterial
        WSe2 monolayer with fitted parameters loaded.
    ws2 : TMDMaterial
        WS2 monolayer with fitted parameters loaded.
    master_folder : str
        Path to repository root (for loading BilayerData).
    stacking : str
        'P' (parallel) or 'AP' (anti-parallel).
    n_kpts : int
        Number of equidistant k-points between Γ and K.
        Used for both ARPES data interpolation and Hamiltonian construction.
    """

    BOUNDS = (-10.0, 10.0)
    """Hard bounds for all interlayer parameters in eV."""

    def __init__(self, wse2: TMDMaterial, ws2: TMDMaterial,
                 master_folder: str, stacking: str = "P",
                 n_kpts: int = 51):
        self.wse2 = wse2
        self.ws2 = ws2
        self.master_folder = master_folder
        self.stacking = stacking
        self.n_kpts = n_kpts
        self.n_bands = 3

        a = wse2.lattice_constant
        self.K_vec = np.array([4 * np.pi / 3 / a, 0])
        self.k_list = np.array([self.K_vec / (self.n_kpts - 1) * i
                                for i in range(self.n_kpts)])
        self.comp_k = np.linalg.norm(self.k_list, axis=1)

        self.bilayer_data = BilayerData(master_folder, pts=n_kpts)
        self.exp_data = self.bilayer_data.fit_data
        self.exp_k = self.exp_data[:, 0]
        self.exp_energies = self.exp_data[:, 1:]

    def _build_hamiltonian(self, interlayer_params: dict) -> tuple:
        """Build 44×44 bilayer Hamiltonian at all k-points.

        Parameters
        ----------
        interlayer_params : dict
            Keys: stacking, w1p, w1d, w2p, w2d.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Eigenvalues (n_kpts, 44) and eigenvectors (n_kpts, 44, 44).
        """
        geometry = MoireGeometry(0.0)
        ham = MoireHamiltonian(self.wse2, self.ws2, geometry)
        pars_V = (0.0, 0.0, 0.0, 0.0)
        return ham.diagonalize(self.k_list, n_shells=0,
                               interlayer_params=interlayer_params,
                               pars_V=pars_V)

    def chi2(self, params: np.ndarray) -> float:
        """Chi-squared objective: sum of squared energy differences.

        Parameters are clipped to [-10, 10] eV. A penalty is added
        for any parameter outside bounds to keep the optimizer within range.

        Parameters
        ----------
        params : np.ndarray
            [w1p, w1d, w2p, w2d] in eV.

        Returns
        -------
        float
            Mean squared error + penalty for out-of-bounds.
        """
        lo, hi = self.BOUNDS
        penalty = 0.0
        clipped = np.clip(params, lo, hi)
        for p in params:
            if p < lo or p > hi:
                penalty += 1e4 * (min(abs(p - lo), abs(p - hi))) ** 2

        w1p, w1d, w2p, w2d = clipped
        interlayer_params = {
            "stacking": self.stacking,
            "w1p": w1p, "w1d": w1d,
            "w2p": w2p, "w2d": w2d,
        }

        evals, _ = self._build_hamiltonian(interlayer_params)

        # Apply S11 sample energy offset (-0.47 eV) to match experimental reference frame
        energy_offset = ENERGY_OFFSETS["S11"]
        evals_shifted = evals + energy_offset

        # Top 3 valence bands (one from each spin-degenerate pair):
        # Band 1 (top, ~-0.9 eV at Γ): index 27
        # Band 2 (middle, ~-1.3 eV at Γ): index 25
        # Band 3 (lower, ~-1.8 eV at Γ): index 23
        # Order matches experimental: Band 1 (top) first, Band 3 (lowest) last
        band_indices = [27, 25, 23]
        computed = evals_shifted[:, band_indices]

        chi2 = 0.0
        n_points = 0
        for ib in range(self.n_bands):
            exp_e = self.exp_energies[:, ib]
            valid = ~np.isnan(exp_e)
            if valid.sum() == 0:
                continue
            comp_interp = np.interp(self.exp_k[valid], self.comp_k, computed[:, ib])
            chi2 += np.sum((comp_interp - exp_e[valid]) ** 2)
            n_points += valid.sum()

        mse = chi2 / n_points if n_points > 0 else 1e10
        return mse + penalty

    def run(self, n_starts: int = 10, seed: int = 42) -> dict:
        """Run optimization from multiple starting points.

        Parameters
        ----------
        n_starts : int
            Number of random starting points.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: x (best params), fun (best chi2), nfev, success, all_results,
            k_list, evals (for plotting).
        """
        rng = np.random.default_rng(seed)
        lo, hi = self.BOUNDS
        all_results = []

        for i in range(n_starts):
            x0 = rng.uniform(lo, hi, size=4)
            result = opt.minimize(
                self.chi2, x0, method="Nelder-Mead",
                options={"maxiter": 2000, "fatol": 1e-6},
            )
            result.x = np.clip(result.x, lo, hi)
            all_results.append(result)

        best = min(all_results, key=lambda r: r.fun)

        best_params = np.clip(best.x, lo, hi)
        interlayer_params = {
            "stacking": self.stacking,
            "w1p": best_params[0], "w1d": best_params[1],
            "w2p": best_params[2], "w2d": best_params[3],
        }
        evals, _ = self._build_hamiltonian(interlayer_params)
        evals = evals + ENERGY_OFFSETS["S11"]

        no_coupling_params = {
            "stacking": self.stacking,
            "w1p": 0.0, "w1d": 0.0,
            "w2p": 0.0, "w2d": 0.0,
        }
        evals_no_coupling, _ = self._build_hamiltonian(no_coupling_params)
        evals_no_coupling = evals_no_coupling + ENERGY_OFFSETS["S11"]

        return {
            "x": best_params,
            "fun": best.fun,
            "nfev": best.nfev,
            "success": best.success,
            "all_results": all_results,
            "k_list": self.k_list,
            "evals": evals,
            "evals_no_coupling": evals_no_coupling,
        }

    def save(self, result: dict, output_dir: str) -> str:
        """Save fitted interlayer parameters.

        Parameters
        ----------
        result : dict
            Output from ``run()``.
        output_dir : str
            Directory to save files.

        Returns
        -------
        str
            Path to saved .npy file.
        """
        from pathlib import Path
        import json
        import datetime

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        params = result["x"]
        np.save(out_dir / "interlayer_params.npy", params)

        metadata = {
            "stacking": self.stacking,
            "n_kpts": self.n_kpts,
            "bounds": list(self.BOUNDS),
            "w1p": float(params[0]),
            "w1d": float(params[1]),
            "w2p": float(params[2]),
            "w2d": float(params[3]),
            "chi2": float(result["fun"]),
            "nfev": int(result["nfev"]),
            "success": bool(result["success"]),
            "n_starts": len(result["all_results"]),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        meta_fn = out_dir / "interlayer_params_metadata.json"
        with open(meta_fn, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(out_dir / "interlayer_params.npy")
