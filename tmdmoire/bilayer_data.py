"""Bilayer ARPES data loading for WSe2/WS2 heterobilayer.

The ``BilayerData`` class reads experimental band dispersion data from
tab-delimited text files in the ``Inputs/bilayer_fitting/`` directory
for the K'→Γ→K high-symmetry path.

Data pipeline
-------------
1. **Load raw** — read ``Inputs/bilayer_fitting/WSe2WS2_Band{N}.txt`` files.
   Each file contains two tab-separated columns: momentum (Å¹) and energy (eV).
   Momentum 0 corresponds to the Gamma point; negative values are on the K' side,
   positive values on the K side.
2. **Symmetrize** — average equivalent K'→Γ and Γ→K segments by folding
   negative momenta to positive |k| and averaging energies where both sides
   have valid data.
3. **Interpolate** — map symmetrized data onto a uniform |k| grid. Each band
   is only defined within its own raw data limits (Band 3 has shorter range
   than Bands 1–2).

Unlike monolayer ARPES data, bilayer data:
- Contains 3 bands (not 6) with different momentum coverage
- Has scattered NAN values (not just at path boundaries)
"""
import numpy as np
from pathlib import Path


class BilayerData:
    """Loads and processes bilayer ARPES band dispersion data for WSe2/WS2.

    Reads raw momentum-energy data from ``Inputs/bilayer_fitting/WSe2WS2_Band{N}.txt``,
    symmetrizes K'→Γ and Γ→K segments, and interpolates onto a uniform |k| grid.

    Parameters
    ----------
    master_folder : str
        Path to the repository root (must contain ``Inputs/bilayer_fitting/``).
    pts : int
        Number of interpolation points along the |k| axis (0 to max_k).

    Attributes
    ----------
    raw_data : list[np.ndarray]
        Raw data per band, each a (N, 2) array of [momentum, energy].
    sym_data : list[np.ndarray]
        Symmetrized data per band, each a (M, 2) array of [|k|, energy].
    fit_data : np.ndarray
        Interpolated data of shape (pts, n_bands + 1):
        [|k|, E_band1, E_band2, E_band3]. NaN outside each band's range.
    n_bands : int
        Number of bands (3).
    """

    def __init__(self, master_folder: str, pts: int = 200):
        self.master_folder = master_folder
        self.pts = pts
        self.n_bands = 3
        self.raw_data = self._load_raw()
        self.sym_data = self._load_or_symmetrize()
        self.fit_data = self._interpolate(pts)

    @property
    def momentum_range(self) -> tuple[float, float]:
        """Full momentum range covered by the raw data (signed)."""
        k_min = min(b[:, 0].min() for b in self.raw_data)
        k_max = max(b[:, 0].max() for b in self.raw_data)
        return k_min, k_max

    @property
    def max_abs_k(self) -> float:
        """Maximum |k| across all symmetrized bands."""
        return max(s[:, 0].max() for s in self.sym_data)

    def _load_raw(self) -> list[np.ndarray]:
        """Read raw bilayer ARPES data from tab-delimited text files.

        Each file contains two tab-separated columns: momentum and energy.
        Missing energy values are encoded as "NAN" or empty lines and
        stored as ``np.nan``.

        Returns
        -------
        list[np.ndarray]
            List of (N, 2) arrays, one per band, columns [momentum, energy].
        """
        raw = []
        for ib in range(1, self.n_bands + 1):
            fn = Path(self.master_folder) / "Inputs" / "bilayer_fitting" / f"WSe2WS2_Band{ib}.txt"
            with open(fn) as f:
                lines = f.readlines()
            temp = []
            for line in lines:
                parts = line.split("\t")
                k = float(parts[0])
                e_str = parts[1].strip()
                if e_str == "" or e_str == "NAN":
                    temp.append([k, np.nan])
                else:
                    temp.append([k, float(e_str)])
            raw.append(np.array(temp))
        return raw

    def _symmetrize(self) -> list[np.ndarray]:
        """Symmetrize K'→Γ and Γ→K segments by averaging.

        For each band:
        1. Split data at k=0 into negative (K' side) and positive (K side).
        2. Mirror the negative side to positive |k|.
        3. Average energies where both sides have valid (non-NaN) data.
        4. Keep one-sided values where only one side has data.
        5. Discard points where both sides are NaN.

        Returns
        -------
        list[np.ndarray]
            Symmetrized data per band, each a (M, 2) array of [|k|, energy].
        """
        sym = []
        for ib in range(self.n_bands):
            rd = self.raw_data[ib]

            neg_mask = rd[:, 0] < 0
            pos_mask = rd[:, 0] > 0

            neg = rd[neg_mask].copy()
            pos = rd[pos_mask].copy()

            neg[:, 0] = np.abs(neg[:, 0])
            neg = neg[::-1]

            nk_neg = neg.shape[0]
            nk_pos = pos.shape[0]
            nk = min(nk_neg, nk_pos)

            k_neg = neg[:nk, 0]
            k_pos = pos[:nk, 0]
            e_neg = neg[:nk, 1]
            e_pos = pos[:nk, 1]

            mask_neg = ~np.isnan(e_neg)
            mask_pos = ~np.isnan(e_pos)
            mask_both = mask_neg & mask_pos
            mask_only_neg = mask_neg & ~mask_pos
            mask_only_pos = ~mask_neg & mask_pos

            k_out = k_neg.copy()
            e_out = np.full(nk, np.nan)
            e_out[mask_both] = (e_neg[mask_both] + e_pos[mask_both]) / 2
            e_out[mask_only_neg] = e_neg[mask_only_neg]
            e_out[mask_only_pos] = e_pos[mask_only_pos]

            valid = ~np.isnan(e_out)
            result = np.column_stack([k_out[valid], e_out[valid]])

            extra_neg = neg[nk:]
            extra_pos = pos[nk:]
            if len(extra_neg) > 0:
                v = ~np.isnan(extra_neg[:, 1])
                result = np.vstack([result, extra_neg[v]])
            if len(extra_pos) > 0:
                v = ~np.isnan(extra_pos[:, 1])
                result = np.vstack([result, extra_pos[v]])

            result = result[result[:, 0].argsort()]
            sym.append(result)
        return sym

    def _load_or_symmetrize(self) -> list[np.ndarray]:
        """Load symmetrized data from cache or compute and save it.

        Checks ``Data/sym_bilayer.npz`` for a previously computed
        symmetrized dataset. If found and the file is newer than all
        raw input files, it is loaded directly. Otherwise, symmetrization
        is performed and the result is saved.

        Returns
        -------
        list[np.ndarray]
            Symmetrized data with same structure as ``raw_data``.
        """
        data_dir = Path("Data")
        cache_fn = data_dir / "sym_bilayer.npz"

        if cache_fn.exists():
            cache_mtime = cache_fn.stat().st_mtime
            all_raw_newer = False
            for ib in range(1, self.n_bands + 1):
                raw_fn = Path(self.master_folder) / "Inputs" / "bilayer_fitting" / f"WSe2WS2_Band{ib}.txt"
                if raw_fn.stat().st_mtime > cache_mtime:
                    all_raw_newer = True
                    break
            if not all_raw_newer:
                return _load_bilayer_sym_cache(cache_fn, self.n_bands)

        sym = self._symmetrize()
        data_dir.mkdir(parents=True, exist_ok=True)
        _save_bilayer_sym_cache(cache_fn, sym, self.n_bands)
        return sym

    def _interpolate(self, pts: int) -> np.ndarray:
        """Interpolate symmetrized data onto a uniform |k| grid.

        The output array has ``pts`` rows and ``n_bands + 1`` columns:
        [|k|, E_band1, E_band2, E_band3].

        Each band is only defined within its own symmetrized |k| range.
        Band 3 has a shorter range (~0.76 Å⁻¹) than Bands 1–2 (~1.26 Å⁻¹),
        so values beyond Band 3's range are NaN.

        Parameters
        ----------
        pts : int
            Number of points along the |k| axis.

        Returns
        -------
        np.ndarray
            Interpolated data of shape (pts, n_bands + 1).
        """
        max_k = self.max_abs_k
        k_grid = np.linspace(0, max_k, pts)

        result = np.zeros((pts, self.n_bands + 1))
        result[:, 0] = k_grid

        for ib in range(self.n_bands):
            sd = self.sym_data[ib]
            valid = ~np.isnan(sd[:, 1])
            if valid.sum() < 2:
                result[:, ib + 1] = np.nan
                continue
            k_band = sd[valid, 0]
            e_band = sd[valid, 1]
            k_max_band = k_band.max()
            within_range = k_grid <= k_max_band
            result[within_range, ib + 1] = np.interp(
                k_grid[within_range], k_band, e_band
            )
            result[~within_range, ib + 1] = np.nan

        return result


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _save_bilayer_sym_cache(fn: Path, sym: list, n_bands: int):
    """Save symmetrized bilayer data to an npz file."""
    save_dict = {}
    for ib in range(n_bands):
        arr = sym[ib]
        save_dict[f"band{ib}_data"] = arr.ravel()
        save_dict[f"band{ib}_shape"] = arr.shape
    np.savez(fn, **save_dict)


def _load_bilayer_sym_cache(fn: Path, n_bands: int) -> list[np.ndarray]:
    """Load symmetrized bilayer data from an npz cache file."""
    data = np.load(fn, allow_pickle=True)
    sym = []
    for ib in range(n_bands):
        arr = data[f"band{ib}_data"]
        shape = tuple(data[f"band{ib}_shape"])
        sym.append(arr.reshape(shape))
    return sym
