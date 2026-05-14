"""Bilayer interlayer coupling fitter.

Fits 4 interlayer hopping parameters (w1p, w1d, w2p, w2d) to reproduce
the 3 top valence bands from bilayer ARPES data along the Γ–K path.
Uses scipy.optimize.minimize (Nelder-Mead) from multiple starting points.
"""
import numpy as np
import scipy.optimize as opt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..material import TMDMaterial
from .data import BilayerData
from .hamiltonian import MoireHamiltonian
from .geometry import MoireGeometry
from ..constants import ENERGY_OFFSETS


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
    n_kpts : int
        Number of equidistant k-points between Γ and K.
    """

    BOUNDS = [
        (-5.0, 5.0),     # w1p
        (-5.0, 5.0),     # w1d
        (-5.0, 5.0),     # w2p
        (-5.0, 5.0),     # w2d
    ]

    def __init__(self, wse2: TMDMaterial, ws2: TMDMaterial,
                 master_folder: str, n_kpts: int = 51,
                 gamma_weight: float = 5.0, gamma_sigma: float = 0.15):
        self.wse2 = wse2
        self.ws2 = ws2
        self.n_kpts = n_kpts
        self.n_bands = 3
        self._debug_iter = 0
        self._debug_dir = None
        self._debug_max = None
        self._debug_no_coupling_evals = None
        self.gamma_weight = gamma_weight
        self.gamma_sigma = gamma_sigma

        a = wse2.lattice_constant
        self.K_vec = np.array([4 * np.pi / 3 / a, 0])
        self.k_list = np.array([self.K_vec / (n_kpts - 1) * i
                                for i in range(n_kpts)])
        self.comp_k = np.linalg.norm(self.k_list, axis=1)

        self.bilayer_data = BilayerData(master_folder, pts=n_kpts)
        self.exp_data = self.bilayer_data.fit_data
        self.exp_k = self.exp_data[:, 0]
        self.exp_energies = self.exp_data[:, 1:]

        self.mono_hams_wse2 = [wse2.build_hamiltonian(k) for k in self.k_list]
        self.mono_hams_ws2 = [ws2.build_hamiltonian(k) for k in self.k_list]

    def _check_bounds(self, params):
        penalty = 0.0
        for i in range(len(params)):
            lo, hi = self.BOUNDS[i]
            v = params[i]
            if v < lo or v > hi:
                penalty += 1e4 * (min(abs(v - lo), abs(v - hi))) ** 2
        return penalty

    def _clip_params(self, params):
        clipped = np.zeros_like(params)
        for i in range(len(params)):
            lo, hi = self.BOUNDS[i]
            clipped[i] = np.clip(params[i], lo, hi)
        return clipped

    def _build_hamiltonian(self, w1p, w1d, w2p, w2d):
        interlayer_params = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
        geometry = MoireGeometry(0.0)
        ham = MoireHamiltonian(self.wse2, self.ws2, geometry)
        pars_V = (0.0, 0.0, 0.0, 0.0)
        return ham.diagonalize(self.k_list, n_shells=0,
                               interlayer_params=interlayer_params,
                               pars_V=pars_V,
                               mono_hams_wse2=self.mono_hams_wse2,
                               mono_hams_ws2=self.mono_hams_ws2)

    def chi2(self, params):
        penalty = self._check_bounds(params)
        w1p, w1d, w2p, w2d = self._clip_params(params)

        evals, _ = self._build_hamiltonian(w1p, w1d, w2p, w2d)
        evals_shifted = evals + ENERGY_OFFSETS["S11"]

        band_indices = [27, 26, 25]
        computed = evals_shifted[:, band_indices]

        chi2 = 0.0
        n_points = 0
        for ib in range(self.n_bands):
            exp_e = self.exp_energies[:, ib]
            valid = ~np.isnan(exp_e)
            if valid.sum() == 0:
                continue
            comp_interp = np.interp(self.exp_k[valid], self.comp_k, computed[:, ib])
            residuals = comp_interp - exp_e[valid]
            k_vals = self.exp_k[valid]
            weights = 1.0 + self.gamma_weight * np.exp(-k_vals ** 2 / (2 * self.gamma_sigma ** 2))
            chi2 += np.sum(weights * residuals ** 2)
            n_points += valid.sum()

        mse = chi2 / n_points if n_points > 0 else 1e10
        return mse + penalty

    def _debug_plot(self, params, chi2_val=None):
        if self._debug_dir is None:
            return
        w1p, w1d, w2p, w2d = self._clip_params(params)
        if chi2_val is None:
            chi2_val = self.chi2(params)

        evals, _ = self._build_hamiltonian(w1p, w1d, w2p, w2d)
        evals = evals + ENERGY_OFFSETS["S11"]

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        band_indices = [27, 26, 25, 24]
        computed = evals[:, band_indices]
        no_coup = self._debug_no_coupling_evals[:, band_indices]

        colors_exp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        colors_fit = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
        band_labels = ["Band 1", "Band 2", "Band 3", "Band 4"]

        exp_data = self.bilayer_data.fit_data
        exp_k = exp_data[:, 0]
        for ib in range(self.n_bands):
            exp_e = exp_data[:, ib + 1]
            valid = ~np.isnan(exp_e)
            ax.scatter(exp_k[valid], exp_e[valid], s=20, c=colors_exp[ib],
                       alpha=0.7, zorder=3, label=f"{band_labels[ib]} (ARPES)")
            ax.plot(self.comp_k, computed[:, ib], color=colors_fit[ib],
                    lw=2.5, zorder=2, label=f"{band_labels[ib]} (current)")
            ax.plot(self.comp_k, no_coup[:, ib], color="gray",
                    lw=1.5, ls="--", zorder=1,
                    label=f"{band_labels[ib]} (no coupling)" if ib == 0 else "")

        ax.plot(self.comp_k, computed[:, 3], color=colors_fit[3],
                lw=2.5, zorder=2, label="Band 4 (current)")
        ax.plot(self.comp_k, no_coup[:, 3], color="gray",
                lw=1.5, ls="--", zorder=1, label="Band 4 (no coupling)")

        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=12)
        ax.set_ylabel("Energy (eV)", fontsize=12)
        ax.set_title(f"Iter {self._debug_iter}: w1p={w1p:+.4f} w1d={w1d:+.4f}  χ²={chi2_val:.6f}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")

        fn = self._debug_dir / f"iter_{self._debug_iter:04d}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._debug_iter += 1

    def _make_callback(self, debug_every=1):
        def callback(intermediate_result):
            if self._debug_iter % debug_every != 0:
                self._debug_iter += 1
                return
            params = intermediate_result.x
            chi2_val = self.chi2(params)
            self._debug_plot(params, chi2_val)
            if self._debug_max is not None and self._debug_iter >= self._debug_max:
                raise StopIteration("Debug limit reached")
        return callback

    def run(self, n_starts=10, seed=42, debug_dir=None, debug_max_iters=None,
            debug_every=1, x0=None):
        rng = np.random.default_rng(seed)
        all_results = []

        self._debug_iter = 0
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_max = debug_max_iters
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            evals_nc, _ = self._build_hamiltonian(0.0, 0.0, 0.0, 0.0)
            self._debug_no_coupling_evals = evals_nc + ENERGY_OFFSETS["S11"]

        callback = self._make_callback(debug_every=debug_every) if self._debug_dir else None

        for i in range(n_starts):
            if x0 is not None and i == 0:
                start = np.array(x0, dtype=float)
            else:
                start = np.array([rng.uniform(lo, hi) for lo, hi in self.BOUNDS])
            try:
                result = opt.minimize(
                    self.chi2, start, method="Nelder-Mead",
                    options={"maxiter": 2000, "fatol": 1e-6},
                    callback=callback,
                )
            except StopIteration:
                break
            result.x = self._clip_params(result.x)
            all_results.append(result)
            if self._debug_max is not None and self._debug_iter >= self._debug_max:
                break

        best = min(all_results, key=lambda r: r.fun)
        best_params = self._clip_params(best.x)

        evals, _ = self._build_hamiltonian(*best_params)
        evals = evals + ENERGY_OFFSETS["S11"]
        evals_nc, _ = self._build_hamiltonian(0.0, 0.0, 0.0, 0.0)
        evals_nc = evals_nc + ENERGY_OFFSETS["S11"]

        return {
            "x": best_params,
            "fun": best.fun,
            "nfev": best.nfev,
            "success": best.success,
            "all_results": all_results,
            "k_list": self.k_list,
            "evals": evals,
            "evals_no_coupling": evals_nc,
        }

    def save(self, result, output_dir):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json, datetime

        params = ["w1p", "w1d", "w2p", "w2d"]
        full_params = result["x"]
        np.save(out_dir / "interlayer_params.npy", full_params)

        metadata = {
            "n_kpts": self.n_kpts,
            **{p: float(full_params[i]) for i, p in enumerate(params)},
            "chi2": float(result["fun"]),
            "nfev": int(result["nfev"]),
            "success": bool(result["success"]),
            "n_starts": len(result["all_results"]),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(out_dir / "interlayer_params_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return str(out_dir / "interlayer_params.npy")
