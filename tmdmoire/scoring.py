"""Scoring and ranking of grid fitting results.

The ``GridScorer`` class loads all ``.npz`` result files from a grid search,
applies multi-stage filtering and ranking, and returns a sorted table of
the best parameter sets.

Scoring stages:
    1. Hard filter: reject results where K4 > threshold (CBM not at K)
    2. Primary rank: sort by chi2_band_unweighted (lowest first)
    3. Tiebreak: for similar chi2_band_unweighted, prefer lower K3 + K2 + K5

Note: ``chi2_band`` in the .npz file is the K6-weighted band distance
(matching the objective function). ``chi2_band_unweighted`` is the pure
band distance without K6 weighting, used for fair cross-comparison
between grid points with different K6 values.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path


class GridScorer:
    """Loads and ranks fitting results from a parameter grid search.

    Parameters
    ----------
    material : str
        Material name (e.g. "WSe2" or "WS2").
    data_dir : str
        Directory containing fit_*.npz files.
    """

    def __init__(self, material: str, data_dir: str = "Data"):
        self.material = material
        self.data_dir = Path(data_dir)

    def load_results(self) -> pd.DataFrame:
        """Load all fit_{material}_idx*.npz files into a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per result, with columns for chi2, individual constraint
            values, K weights, and the parameter array.
        """
        rows = []
        pattern = "fit_idx*.npz"
        for fn in sorted(self.data_dir.glob(pattern)):
            d = np.load(fn, allow_pickle=True)
            match = re.search(r"idx(\d+)", fn.stem)
            if match is None:
                continue
            idx = int(match.group(1))
            Ks = d["Ks"]
            rows.append({
                "idx": idx,
                "chi2": float(d["chi2"]),
                "chi2_band": float(d["chi2_band"]),
                "chi2_band_unweighted": float(d["chi2_band_unweighted"]) if "chi2_band_unweighted" in d else float(d["chi2_band"]),
                "K1_val": float(d["K1_val"]),
                "K2_val": float(d["K2_val"]),
                "K3_val": float(d["K3_val"]),
                "K4_val": float(d["K4_val"]),
                "K5_val": float(d["K5_val"]),
                "nfev": int(d["nfev"]),
                "K1_w": float(Ks[0]),
                "K2_w": float(Ks[1]),
                "K3_w": float(Ks[2]),
                "K4_w": float(Ks[3]),
                "K5_w": float(Ks[4]),
                "K6_w": float(Ks[5]),
                "Bs": d["Bs"],
                "params": d["params"],
                "tb_en": d["tb_en"],
                "k_path": d["k_path"],
            })
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame | None = None,
              k4_threshold: float = 0.05,
              top_n: int = 50) -> pd.DataFrame:
        """Apply multi-stage scoring and return ranked results.

        Stage 1: Hard filter — reject if K4 > k4_threshold (CBM not at K).
        Stage 2: Primary rank — sort by chi2_band_unweighted ascending.
        Stage 3: Tiebreak — for similar chi2_band_unweighted, prefer lower
                 K3_val + K2_val + K5_val.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded results. If None, loads from disk.
        k4_threshold : float
            Maximum acceptable K4 value (squared relative CBM offset).
        top_n : int
            Maximum number of results to return.

        Returns
        -------
        pd.DataFrame
            Scored and ranked results with a ``rank`` column.
        """
        if df is None:
            df = self.load_results()

        if df.empty:
            return df

        passed = df[df["K4_val"] < k4_threshold].copy()
        passed["composite_secondary"] = (
            passed["K3_val"] + passed["K2_val"] + passed["K5_val"]
        )
        passed = passed.sort_values(
            ["chi2_band_unweighted", "composite_secondary"],
            ascending=[True, True],
        ).reset_index(drop=True)
        passed["rank"] = passed.index + 1
        return passed.head(top_n)

    def summary(self, df: pd.DataFrame | None = None,
                k4_threshold: float = 0.05,
                top_n: int = 10) -> str:
        """Generate a human-readable summary of top results.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded results. If None, loads from disk.
        k4_threshold : float
            K4 hard filter threshold.
        top_n : int
            Number of top results to display.

        Returns
        -------
        str
            Formatted summary table.
        """
        ranked = self.score(df, k4_threshold=k4_threshold, top_n=top_n)
        if ranked.empty:
            return "No results pass the K4 filter."

        lines = [
            f"Top {min(top_n, len(ranked))} results for {self.material}",
            f"  (K4 threshold: {k4_threshold}, total loaded: {len(self.load_results())})",
            "",
            f"{'Rank':>4} {'Idx':>5} {'chi2_bw':>10} {'chi2_band':>10} {'K4_val':>8} {'K3_val':>8} {'K2_val':>8} {'K5_val':>8} {'nfev':>8}",
            "-" * 80,
        ]
        for _, row in ranked.iterrows():
            lines.append(
                f"{row['rank']:>4} {row['idx']:>5} {row['chi2_band_unweighted']:>10.6f} "
                f"{row['chi2_band']:>10.6f} {row['K4_val']:>8.6f} {row['K3_val']:>8.6f} "
                f"{row['K2_val']:>8.6f} {row['K5_val']:>8.6f} {row['nfev']:>8}"
            )
        return "\n".join(lines)

    def get_best_params(self, df: pd.DataFrame | None = None,
                        k4_threshold: float = 0.05) -> np.ndarray | None:
        """Return the parameter array of the best-scoring result.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-loaded results. If None, loads from disk.
        k4_threshold : float
            K4 hard filter threshold.

        Returns
        -------
        np.ndarray or None
            Best parameter array, or None if no results pass the filter.
        """
        ranked = self.score(df, k4_threshold=k4_threshold, top_n=1)
        if ranked.empty:
            return None
        return ranked.iloc[0]["params"]
