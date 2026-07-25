"""Scoring and ranking of grid fitting results.

The ``GridScorer`` class loads all ``.npz`` result files from a grid search,
applies filtering and ranking matching the v3.0 procedure:

    1. Filter: K-value range masks (K2 between -2^-8 and 10, etc.)
    2. For WSe2: bounds-saturation filter (exclude results where any
       parameter group saturated its bounds)
    3. Rank by ``band_K6`` (K6-weighted band distance, matching v3.0's
       primary ranking).
    4. Secondary rank: ``band_K6 + K2_val`` (band distance + M orbital content).
    5. Pick ``ind_chosen`` = 1 for WSe2, 0 for WS2.
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
        """Load all fit_idx*.npz files into a DataFrame."""
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
                "band_K6": float(d["band_K6"]) if "band_K6" in d else float(d["chi2_band"]),
                "band_dist": float(d["band_dist"]) if "band_dist" in d else float(d["chi2_band_unweighted"]) if "chi2_band_unweighted" in d else float(d["chi2_band"]),
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

    def _apply_K_range_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply K-value range mask matching v3.0's load_h5 filtering."""
        mask = (
            (df["K1_w"] > -1e-7)
            & (df["K2_w"] > -(2**(-8)))
            & (df["K2_w"] < 10)
            & (df["K3_w"] > -0.012)
            & (df["K6_w"] > -1)
        )
        return df[mask].copy()

    def _apply_bounds_saturation_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """For WSe2: exclude results where any parameter group saturated its bounds."""
        if self.material != "WSe2":
            return df

        tol = 1e-2
        param_group_bounds = [(0, 7, 0), (7, 28, 1), (28, 36, 2), (36, 40, 3)]

        def _not_saturated(row):
            params = row["params"]
            Bs = row["Bs"]
            for start, end, b_idx in param_group_bounds:
                if np.any(np.abs(params[start:end]) >= Bs[b_idx] - tol):
                    return False
            return True

        mask = df.apply(_not_saturated, axis=1)
        return df[mask].copy()

    def score(self, df: pd.DataFrame | None = None, top_n: int = 50) -> pd.DataFrame:
        """Apply v3.0-style scoring and return ranked results.

        Ranks by band_K6 (primary) and band_K6 + K2_val (secondary).
        Uses ind_chosen = 1 for WSe2, 0 for WS2.
        """
        if df is None:
            df = self.load_results()

        if df.empty:
            return df

        df = self._apply_K_range_mask(df)
        if self.material == "WSe2":
            df = self._apply_bounds_saturation_mask(df)
        if df.empty:
            return df

        df = df.copy()
        df["band_plus_K2"] = df["band_K6"] + df["K2_val"]

        ind_chosen = 1 if self.material == "WSe2" else 0

        ranked_by_chi2 = df.sort_values("band_K6", ascending=True).reset_index(drop=True)
        ranked_by_comb = df.sort_values("band_plus_K2", ascending=True).reset_index(drop=True)

        selected_chi2 = ranked_by_chi2.iloc[min(ind_chosen, len(ranked_by_chi2) - 1)]
        selected_comb = ranked_by_comb.iloc[min(ind_chosen, len(ranked_by_comb) - 1)]

        results = []
        for rank_type, row in [("chi2", selected_chi2), ("chi2+K2_M", selected_comb)]:
            results.append({
                "rank_type": rank_type,
                "idx": row["idx"],
                "band_K6": row["band_K6"],
                "band_dist": row["band_dist"],
                "band_plus_K2": row["band_plus_K2"],
                "K2_val": row["K2_val"],
                "K3_val": row["K3_val"],
                "K4_val": row["K4_val"],
                "K5_val": row["K5_val"],
                "chi2": row["chi2"],
                "nfev": row["nfev"],
                "K1_w": row["K1_w"],
                "K2_w": row["K2_w"],
                "K3_w": row["K3_w"],
                "K4_w": row["K4_w"],
                "K5_w": row["K5_w"],
                "K6_w": row["K6_w"],
                "params": row["params"],
                "tb_en": row["tb_en"],
                "k_path": row["k_path"],
                "Bs": row["Bs"],
            })
        return pd.DataFrame(results)

    def summary(self, df: pd.DataFrame | None = None, top_n: int = 10) -> str:
        """Generate a human-readable summary of top results."""
        ranked = self.score(df, top_n=top_n)
        if ranked.empty:
            return "No results after filtering."

        lines = [
            f"Top results for {self.material} (v3.0 scoring)",
            f"  ind_chosen = {1 if self.material == 'WSe2' else 0}",
            f"  total loaded before filtering: {len(self.load_results())}",
            "",
            f"{'Type':>10} {'Idx':>5} {'band_K6':>10} {'band_dist':>10} {'band+K2':>10} {'chi2':>10} {'K2_val':>8} {'K3_val':>8} {'K4_val':>8} {'K5_val':>8}",
            "-" * 100,
        ]
        for _, row in ranked.iterrows():
            lines.append(
                f"{row['rank_type']:>10} {row['idx']:>5} {row['band_K6']:>10.6f} "
                f"{row['band_dist']:>10.6f} {row['band_plus_K2']:>10.6f} {row['chi2']:>10.6f} "
                f"{row['K2_val']:>8.6f} {row['K3_val']:>8.6f} {row['K4_val']:>8.6f} {row['K5_val']:>8.6f}"
            )
        return "\n".join(lines)

    def get_best_params(self, df: pd.DataFrame | None = None) -> np.ndarray | None:
        """Return the parameter array of the best-scoring result (chi2 ranking)."""
        ranked = self.score(df, top_n=2)
        if ranked.empty:
            return None
        return ranked[ranked["rank_type"] == "chi2"].iloc[0]["params"]
