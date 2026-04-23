"""ARPES data loading, symmetrization, and interpolation.

The ``ARPESData`` class reads experimental band dispersion data from
tab-delimited text files in the ``Inputs/`` directory, symmetrizes
the K→Γ and Γ→K segments, and interpolates onto a uniform grid for
use in the tight-binding fitting procedure.

Supported high-symmetry paths:
    - KpGK: K′ → Γ → K (6 bands for WSe2/WS2)
    - KMKp: K → M → K′ (4 bands for WSe2/WS2)
"""
import numpy as np
import scipy.linalg as la
from pathlib import Path
import copy
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
        if tmd not in LATTICE_CONSTANTS:
            raise ValueError(f"Unknown TMD: {tmd}")
        self.tmd = tmd
        self.master_folder = master_folder
        self.paths = ["KpGK", "KMKp"]
        self.nbands = {"KpGK": 6, "KMKp": 4}
        self.pts = pts
        a = LATTICE_CONSTANTS[self.tmd]
        self.M = np.array([np.pi, np.pi / np.sqrt(3)]) / a
        self.K = np.array([4 * np.pi / 3, 0]) / a
        self.Kp = np.array([2 * np.pi / 3, 2 * np.pi / np.sqrt(3)]) / a
        self.raw_data = self._load_raw()
        self.sym_data = self._symmetrize()
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

    def _load_raw(self) -> dict:
        """Read raw ARPES data from tab-delimited text files.

        Returns
        -------
        dict
            Nested dict: {path: [band_array, ...]} where each band_array
            has shape (N_points, 2) with columns [momentum, energy].
        """
        raw = {}
        for path in self.paths:
            raw[path] = []
            for ib in range(self.nbands[path]):
                fn = Path(self.master_folder + "Inputs/" + path + "_" + self.tmd + "_band%d" % (ib + 1) + ".txt")
                with open(fn, "r") as f:
                    lines = f.readlines()
                temp = []
                for line in lines:
                    k, e = line.split("\t")
                    if e == "NAN\n" or e == "\n":
                        temp.append([float(k), np.nan])
                    else:
                        temp.append([float(k), float(e)])
                        if path == "KpGK" and ib > 1:
                            temp[-1][0] = abs(temp[-1][0])
                temp = np.array(temp)
                if path == "KpGK" and ib > 1:
                    temp = temp[::-1]
                raw[path].append(temp)
        return raw

    def _symmetrize(self) -> dict:
        """Symmetrize K→Γ and Γ→K segments by averaging.

        For bands 3+ on KpGK (which have only positive momenta), the
        raw data is used directly. For WS2 on KMKp, only the left
        segment is used (better fit quality).

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
                if ib > 1 and path == "KpGK":
                    sym[path].append(self.raw_data[path][ib])
                    continue
                if ib > 1 and path == "KMKp" and self.tmd == "WS2":
                    rd = copy.copy(self.raw_data[path][ib])
                    left_side = copy.deepcopy(rd[: rd.shape[0] // 2, :][::-1, :])
                    left_side[:, 0] = np.absolute(left_side[:, 0])
                    sym[path].append(left_side)
                    continue
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
        [|k|, kx, ky, E_1, E_2, E_3, E_4, E_5, E_6].

        Bands 3–4 are NaN except near M or Γ.
        Bands 5–6 are NaN except near Γ.

        Parameters
        ----------
        pts : int
            Number of points along the combined path.

        Returns
        -------
        np.ndarray
            Interpolated data of shape (pts, 9).
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
