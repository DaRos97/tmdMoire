"""ARPES data loading, symmetrization, and interpolation.

The ``ARPESData`` class reads experimental band dispersion data from
tab-delimited text files in the ``Inputs/`` directory, symmetrizes
the K→Γ and Γ→K segments, and interpolates onto a uniform grid for
use in the tight-binding fitting procedure.

Data pipeline
-------------
1. **Load raw** — read ``Inputs/{path}_{TMD}_band{N}.txt`` files as-is.
   No transformation is applied; the loader preserves the original
   momentum sign and ordering from the experiment.
2. **Symmetrize** — average equivalent K→Γ and Γ→K segments, handle
   special cases for sparse bands (see :meth:`_symmetrize`).
3. **Interpolate** — map symmetrized data onto ``pts`` equidistant
   points along the combined Γ–K–M path.

The symmetrized data is cached in ``Data/sym_{TMD}.npz`` to avoid
re-processing on subsequent runs.

Supported high-symmetry paths:
    - KpGK: K′ → Γ → K (6 bands for WSe2/WS2)
    - KMKp: K → M → K′ (4 bands for WSe2/WS2)
"""
import numpy as np
import scipy.linalg as la
from pathlib import Path
import json
from .constants import LATTICE_CONSTANTS, MONOLAYER_OFFSETS


class ARPESData:
    """Loads and processes ARPES band dispersion data for a TMD monolayer.

    Reads raw momentum-energy data from ``Inputs/{path}_{TMD}_band{N}.txt``,
    symmetrizes equivalent path segments, and interpolates onto a uniform
    grid of ``pts`` points for fitting.

    Parameters
    ----------
    tmd : str
        Material name, "WSe2" or "WS2".
    master_folder : str
        Path to the repository root (must contain ``Inputs/`` subdirectory).
    pts : int
        Number of interpolation points along the combined path.
        Should be of the form 3n + 1 for even segment division.

    Attributes
    ----------
    raw_data : dict
        Raw data keyed by path name, each a list of (k, E) arrays per band.
    sym_data : dict
        Symmetrized data with K→Γ and Γ→K segments averaged.
    fit_data : np.ndarray
        Final fitting array of shape (pts, 9):
        [|k|, kx, ky, E_band1, ..., E_band6].

    Examples
    --------
    >>> arpes = ARPESData("WSe2", master_folder="/path/to/repo/", pts=91)
    >>> arpes.fit_data.shape
    (91, 9)
    """

    def __init__(self, tmd: str, master_folder: str, pts: int = 61):
        if tmd not in ("WSe2", "WS2"):
            raise ValueError(f"Unknown TMD: {tmd}")
        self.tmd = tmd
        self.master_folder = master_folder
        self.nbands = self._load_manifest()
        self.paths = list(self.nbands.keys())
        self.pts = pts
        a = LATTICE_CONSTANTS[self.tmd]
        self.M = np.array([np.pi, np.pi / np.sqrt(3)]) / a
        self.K = np.array([4 * np.pi / 3, 0]) / a
        self.Kp = np.array([2 * np.pi / 3, 2 * np.pi / np.sqrt(3)]) / a
        self.raw_data = self._load_raw()
        self.sym_data = self._load_or_symmetrize()
        self.fit_data = self._interpolate(pts)

    @property
    def gamma_point(self) -> np.ndarray:
        """Gamma point in reciprocal space."""
        return np.zeros(2)

    @property
    def k_point(self) -> np.ndarray:
        """K point in reciprocal space."""
        return self.K.copy()

    @property
    def m_point(self) -> np.ndarray:
        """M point in reciprocal space."""
        return self.M.copy()

    def _load_manifest(self) -> dict:
        """Read band counts from Inputs/manifest.json.

        Returns
        -------
        dict
            Mapping of path name to number of bands, e.g.
            ``{"KpGK": 6, "KMKp": 4}``.
        """
        manifest_path = Path(self.master_folder) / "Inputs" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            return dict(manifest[self.tmd])
        # Fallback to hardcoded defaults if manifest is missing
        return {"KpGK": 6, "KMKp": 4}

    def _load_raw(self) -> dict:
        """Read raw ARPES data from tab-delimited text files without transformation.

        Each file contains two tab-separated columns: momentum and energy.
        Missing energy values are encoded as "NAN" or empty lines and
        stored as ``np.nan``. The raw data is returned exactly as stored
        in the files — no sign changes, no reordering.

        Returns
        -------
        dict
            Nested dict: ``{path: [band_array, ...]}`` where each
            ``band_array`` has shape ``(N_points, 2)`` with columns
            ``[momentum, energy]``.
        """
        raw = {}
        for path in self.paths:
            raw[path] = []
            for ib in range(self.nbands[path]):
                fn = Path(self.master_folder) / "Inputs" / f"{path}_{self.tmd}_band{ib + 1}.txt"
                with open(fn) as f:
                    lines = f.readlines()
                temp = []
                for line in lines:
                    k_str, e_str = line.split("\t")
                    k = float(k_str)
                    if e_str.strip() == "" or e_str.strip() == "NAN":
                        temp.append([k, np.nan])
                    else:
                        temp.append([k, float(e_str)])
                raw[path].append(np.array(temp))
        return raw

    def _load_or_symmetrize(self) -> dict:
        """Load symmetrized data from cache or compute and save it.

        Checks ``Data/sym_{TMD}.npz`` for a previously computed
        symmetrized dataset. If found and the file is newer than all
        raw input files, it is loaded directly. Otherwise, symmetrization
        is performed and the result is saved.

        Returns
        -------
        dict
            Symmetrized data with same structure as ``raw_data``.
        """
        data_dir = Path("Data")
        cache_fn = data_dir / f"sym_{self.tmd}.npz"

        if cache_fn.exists():
            cache_mtime = cache_fn.stat().st_mtime
            all_raw_newer = False
            for path in self.paths:
                for ib in range(self.nbands[path]):
                    raw_fn = Path(self.master_folder) / "Inputs" / f"{path}_{self.tmd}_band{ib + 1}.txt"
                    if raw_fn.stat().st_mtime > cache_mtime:
                        all_raw_newer = True
                        break
            if not all_raw_newer:
                return _load_sym_cache(cache_fn, self.paths, self.nbands)

        sym = self._symmetrize()
        data_dir.mkdir(parents=True, exist_ok=True)
        _save_sym_cache(cache_fn, sym, self.paths, self.nbands)
        return sym

    def _symmetrize(self) -> dict:
        """Symmetrize K→Γ and Γ→K segments by averaging.

        For each band on each path, the raw data spans both sides of
        the high-symmetry point (Γ or M). This method:

        1. Splits the data at the symmetry point (midpoint of the array).
        2. Reverses the left-side segment to align with the right side.
        3. Averages energies where both sides have valid (non-NaN) data.
        4. Keeps the one-sided value where only one side has data.
        5. Discards points where both sides are NaN.

        Special cases:
            - **KpGK bands 3–6**: These bands have sparse data with only
              negative momenta. The raw data is used directly after
              taking ``|k|`` and reversing the order so momentum increases
              from Γ outward.
            - **WS2 KMKp bands 3–4**: Only the left segment (K→M) is
              used because the right segment (M→K′) has poorer fit quality.
              The left side is mirrored to positive momentum.

        Returns
        -------
        dict
            Symmetrized data with same structure as ``raw_data``.
        """
        sym = {}
        for path in self.paths:
            sym[path] = []
            for ib in range(self.nbands[path]):
                rd = self.raw_data[path][ib]

                # KpGK bands 3+: sparse data with only negative momenta
                if ib > 1 and path == "KpGK":
                    data = rd.copy()
                    data[:, 0] = np.abs(data[:, 0])
                    data = data[::-1]
                    sym[path].append(data)
                    continue

                # WS2 KMKp bands 3+: use only left segment (K→M)
                if ib > 1 and path == "KMKp" and self.tmd == "WS2":
                    left = rd[: rd.shape[0] // 2].copy()
                    left[:, 0] = np.abs(left[:, 0])
                    left = left[::-1]
                    sym[path].append(left)
                    continue

                # General case: average left and right segments
                nk = rd.shape[0]
                nkl = nk // 2
                nkr = nk // 2 if nk % 2 == 0 else nk // 2 + 1

                temp = np.zeros((nkl, 2))
                temp[:, 0] = rd[nkr:, 0]
                rd_m = rd[:nkl, 1][::-1]
                rd_p = rd[nkr:, 1]

                mask_m = ~np.isnan(rd_m)
                mask_p = ~np.isnan(rd_p)
                mask_tot = mask_m & mask_p
                temp[mask_tot, 1] = (rd_m[mask_tot] + rd_p[mask_tot]) / 2

                mask_tm = mask_m & ~mask_p
                temp[mask_tm, 1] = rd_m[mask_tm]

                mask_tp = ~mask_m & mask_p
                temp[mask_tp, 1] = rd_p[mask_tp]

                temp = np.delete(temp, ~mask_m & ~mask_p, axis=0)

                if nk % 2 == 1:
                    temp = np.insert(temp, 0, rd[nk // 2], axis=0)

                sym[path].append(temp)
        return sym

    def _interpolate(self, pts: int) -> np.ndarray:
        """Interpolate symmetrized data onto a uniform grid.

        The output array has ``pts`` rows and 9 columns:
        ``[|k|, kx, ky, E_1, E_2, E_3, E_4, E_5, E_6]``.

        Bands 3–4 are NaN except near M or Γ.
        Bands 5–6 are NaN except near Γ.

        An energy offset is applied to the KMKp segment to align it
        with the KpGK segment. The offset is material-specific:
        WSe2: −0.052 eV, WS2: +0.010 eV.

        Parameters
        ----------
        pts : int
            Number of points along the combined path.

        Returns
        -------
        np.ndarray
            Interpolated data of shape ``(pts, 9)``.
        """
        modKM = la.norm(self.M - self.K)
        modK = la.norm(self.K)
        data = np.zeros((pts, 9))

        if "KpGK" in self.paths:
            ptsGK = pts // 3 * 2 if "KMKp" in self.paths else pts
            data[:ptsGK, 0] = np.linspace(0, modK, ptsGK, endpoint=False)
            data[:ptsGK, 1] = np.linspace(0, modK, ptsGK, endpoint=False)
            data[:ptsGK, 2] = np.zeros(ptsGK)
            for ib in range(self.nbands["KpGK"]):
                sd = self.sym_data["KpGK"][ib]
                ind = np.searchsorted(data[:ptsGK, 0], np.max(sd[:, 0]), side="left")
                ikmax = min(ind, ptsGK)
                data[:ikmax, 3 + ib] = np.interp(
                    data[:ikmax, 0], sd[~np.isnan(sd[:, 1]), 0], sd[~np.isnan(sd[:, 1]), 1]
                )
                data[ikmax:ptsGK, 3 + ib] = np.nan

        if "KMKp" in self.paths:
            if "KpGK" in self.paths:
                ptsKM = pts - ptsGK
            else:
                ptsGK = 0
                ptsKM = pts
            data[ptsGK:, 0] = np.linspace(modK, modK + modKM, ptsKM, endpoint=True)
            data[ptsGK:, 1] = np.linspace(self.K[0], self.M[0], ptsKM, endpoint=True)
            data[ptsGK:, 2] = np.linspace(self.K[1], self.M[1], ptsKM, endpoint=True)
            for ib in range(self.nbands["KMKp"]):
                sd = self.sym_data["KMKp"][ib]
                indmax = np.searchsorted(data[ptsGK:, 0], modK + modKM - np.min(sd[:, 0]), side="left") + 1
                ikmax = min(indmax, ptsKM)
                indmin = np.searchsorted(data[ptsGK:, 0], modK + modKM - np.max(sd[:, 0]), side="right")
                ikmin = min(indmin, ptsKM)
                data[ptsGK + ikmin:ptsGK + ikmax, 3 + ib] = np.interp(
                    data[ptsGK + ikmin:ptsGK + ikmax, 0],
                    modK + modKM - sd[~np.isnan(sd[:, 1]), 0][::-1],
                    sd[~np.isnan(sd[:, 1]), 1][::-1],
                )
                data[ptsGK:ptsGK + ikmin, 3 + ib] = np.nan
                data[ptsGK + ikmax:, 3 + ib] = np.nan
            data[ptsGK:, 7] = np.nan
            data[ptsGK:, 8] = np.nan
            mask = data[ptsGK:, 4] < data[ptsGK:, 5]
            data[ptsGK:, 4][mask], data[ptsGK:, 5][mask] = data[ptsGK:, 5][mask], data[ptsGK:, 4][mask]
            offset = MONOLAYER_OFFSETS[self.tmd]
            data[ptsGK:, 3] += offset
            data[ptsGK:, 4] += offset
            data[ptsGK:, 5] += offset
            data[ptsGK:, 6] += offset

        return data


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _save_sym_cache(fn: Path, sym: dict, paths: list, nbands: dict):
    """Save symmetrized data to an npz file.

    Each band is stored as a 1D flattened array with a companion shape
    array. Keys follow the pattern ``{path}_band{ib}_data`` and
    ``{path}_band{ib}_shape``.
    """
    save_dict = {}
    for path in paths:
        for ib in range(nbands[path]):
            arr = sym[path][ib]
            save_dict[f"{path}_band{ib}_data"] = arr.ravel()
            save_dict[f"{path}_band{ib}_shape"] = arr.shape
    np.savez(fn, **save_dict)


def _load_sym_cache(fn: Path, paths: list, nbands: dict) -> dict:
    """Load symmetrized data from an npz cache file.

    Parameters
    ----------
    fn : Path
        Path to the .npz cache file.
    paths : list
        List of path names (e.g. ["KpGK", "KMKp"]).
    nbands : dict
        Mapping of path name to number of bands.

    Returns
    -------
    dict
        Symmetrized data with same structure as ``raw_data``.
    """
    data = np.load(fn, allow_pickle=True)
    sym = {}
    for path in paths:
        sym[path] = []
        for ib in range(nbands[path]):
            arr = data[f"{path}_band{ib}_data"]
            shape = tuple(data[f"{path}_band{ib}_shape"])
            sym[path].append(arr.reshape(shape))
    return sym
