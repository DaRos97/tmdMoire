"""Run monolayer fitting across a grid of constraint weights.

Executes the full fitting pipeline for each combination of constraint
weights (K1-K6) defined in ``Inputs/monolayer_fitting/fit_config.json``. A snapshot of
the config is copied to ``Data/run_<id>/fit_config.json`` for
reproducibility. Results are saved as
``Data/run_<id>/fit_{TMD}_idx{N}.npz`` files.

Usage
-----
::

    python scripts/run_monolayer_grid.py WSe2                          # All combinations
    python scripts/run_monolayer_grid.py WSe2 --start 0 --end 100      # Chunk
    python scripts/run_monolayer_grid.py WSe2 --run-id 001             # Named run
    python scripts/run_monolayer_grid.py WSe2 --score                   # Score existing results
    python scripts/run_monolayer_grid.py WSe2 --score --run-id 001      # Score specific run
    python scripts/run_monolayer_grid.py WSe2 --score --top 20          # Top 20
    python scripts/run_monolayer_grid.py WSe2 --score --k4-threshold 0.1
    python scripts/run_monolayer_grid.py WSe2 --score --plot              # Score + generate plots
    python scripts/run_monolayer_grid.py WSe2 --score --plot --top 5      # Plots for top 5
    python scripts/run_monolayer_grid.py WSe2 --score --export             # Score + export best params for bilayer

Arguments
---------
- ``WSe2`` or ``WS2``: Target material.
- ``--start``, ``--end``: Run only a subset of indices (for HPC chunking).
- ``--run-id``: Subdirectory name under Data/ for results (default: "default").
  Results are saved to ``Data/<material>_run_<id>/``.
- ``--score``: Skip fitting, just score and display existing results.
- ``--top N``: Show top N results when scoring (default: 10).
- ``--k4-threshold``: K4 hard filter threshold for scoring (default: 0.05).
- ``--export``: Save best params and metadata to ``Inputs/bilayer_fitting/`` for bilayer fitting.

Examples
--------
Run all combinations for WSe2::

    python scripts/run_monolayer_grid.py WSe2

Run indices 0-399 (submit as HPC job 1)::

    python scripts/run_monolayer_grid.py WSe2 --start 0 --end 400

Run with a named subdirectory::

    python scripts/run_monolayer_grid.py WSe2 --run-id 001

Score existing results::

    python scripts/run_monolayer_grid.py WSe2 --score

Score a specific run::

    python scripts/run_monolayer_grid.py WSe2 --score --run-id 001
"""
import sys
import os
import json
import argparse
import itertools
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tmdmoire import (
    TMDMaterial, MonolayerData, ParameterFitter, GridScorer,
    get_repo_root, prepare_run_dir,
)


def load_fit_config(run_dir: str) -> dict:
    """Load fit configuration from the run directory."""
    config_path = os.path.join(run_dir, "fit_config.json")
    with open(config_path) as f:
        return json.load(f)


def build_grid(config: dict) -> list[dict]:
    """Build the full list of parameter configurations from the grid spec."""
    grid = config["grid"]
    keys = ["K1", "K2", "K3", "K4", "K5", "K6"]
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))

    configs = []
    for idx, combo in enumerate(combos):
        configs.append({
            "idx": idx,
            "Ks": combo,
            "pts": config.get("pts", 91),
            "seed": config.get("seed", 42),
            "boundType": config["bounds"]["boundType"],
            "Bs": tuple(config["bounds"]["Bs"]),
            "optimizer": config.get("optimizer", {}),
            "use_dft_x0": config.get("use_dft_x0", True),
        })
    return configs


def run_chunk(material_name: str, master_folder: str,
              start: int, end: int, seed: int = 42,
              run_dir: str = "Data") -> None:
    """Run fitting for a chunk of grid indices."""
    config = load_fit_config(run_dir)
    all_configs = build_grid(config)
    total = len(all_configs)

    if start >= total:
        print(f"Start index {start} exceeds total grid size {total}")
        return
    end = min(end, total)

    material = TMDMaterial(material_name)
    pts = config.get("pts", 91)
    data = MonolayerData(material_name, master_folder, pts=pts)

    print(f"Running indices {start} to {end - 1} / {total}")
    print(f"Material: {material_name}, pts: {pts}, seed: {seed}")
    print(f"Output directory: {run_dir}")
    print()

    t_start = time.time()
    for i in range(start, end):
        cfg = all_configs[i]
        fitter = ParameterFitter(material, data, cfg, idx=i)

        t_fit = time.time()
        result = fitter.run(seed=seed)
        result["idx"] = i
        result["seed"] = seed

        fn = fitter.save(result, output_dir=run_dir)
        t_fit = time.time() - t_fit

        print(f"[{i:4d}/{total}] chi2={result['fun']:.6f}  "
              f"nfev={result['nfev']}  "
              f"time={t_fit:.1f}s  "
              f"saved: {fn.name}")

    t_total = time.time() - t_start
    print(f"\nDone. {end - start} fits in {t_total:.1f}s "
          f"({t_total / (end - start):.1f}s avg)")


def do_score(material_name: str, top_n: int, k4_threshold: float,
             run_dir: str = "Data", master_folder: str = "", plot: bool = False,
             export: bool = False) -> None:
    """Load and score existing results."""
    scorer = GridScorer(material_name, data_dir=run_dir)
    print(scorer.summary(k4_threshold=k4_threshold, top_n=top_n))

    if plot:
        ranked = scorer.score(k4_threshold=k4_threshold, top_n=top_n)
        if not ranked.empty:
            print(f"\nGenerating plots for top {len(ranked)} results...")
            from tmdmoire.plotting.monolayer import plot_top_results
            plot_top_results(ranked, material_name, master_folder, run_dir)

    if export:
        ranked = scorer.score(k4_threshold=k4_threshold, top_n=1)
        if ranked.empty:
            print("\nNo results pass the K4 filter — cannot export.")
            return
        row = ranked.iloc[0]
        out_dir = Path(master_folder) / "Inputs" / "bilayer_fitting"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = row["params"]
        np.save(out_dir / f"tb_{material_name}.npy", params)

        import datetime
        metadata = {
            "material": material_name,
            "idx": int(row["idx"]),
            "rank": int(row["rank"]),
            "chi2": float(row["chi2"]),
            "band_dist": float(row["band_dist"]),
            "band_K6": float(row["band_K6"]),
            "K1_val": float(row["K1_val"]),
            "K2_val": float(row["K2_val"]),
            "K3_val": float(row["K3_val"]),
            "K4_val": float(row["K4_val"]),
            "K5_val": float(row["K5_val"]),
            "K1_w": float(row["K1_w"]),
            "K2_w": float(row["K2_w"]),
            "K3_w": float(row["K3_w"]),
            "K4_w": float(row["K4_w"]),
            "K5_w": float(row["K5_w"]),
            "K6_w": float(row["K6_w"]),
            "nfev": int(row["nfev"]),
            "k4_threshold": k4_threshold,
            "run_id": run_dir,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        import json
        meta_fn = out_dir / f"tb_{material_name}_metadata.json"
        with open(meta_fn, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nExported best params to {out_dir / f'tb_{material_name}.npy'}")
        print(f"Exported metadata to {meta_fn}")


def main():
    parser = argparse.ArgumentParser(
        description="Run monolayer fitting grid search or score results."
    )
    parser.add_argument("material", choices=["WSe2", "WS2"],
                        help="Target material.")
    parser.add_argument("--start", type=int, default=0,
                        help="First grid index (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="Last grid index (exclusive). Default: run all.")
    parser.add_argument("--run-id", type=str, default="default",
                        help="Run identifier. Results saved to Data/run_<id>/ (default: 'default').")
    parser.add_argument("--score", action="store_true",
                        help="Skip fitting, just score existing results.")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to show when scoring.")
    parser.add_argument("--k4-threshold", type=float, default=0.05,
                        help="K4 hard filter threshold for scoring.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fitting.")
    parser.add_argument("--plot", action="store_true",
                        help="When scoring, also generate plots for top results.")
    parser.add_argument("--export", action="store_true",
                        help="When scoring, export best params and metadata to Inputs/bilayer_fitting/.")

    args = parser.parse_args()

    master_folder = get_repo_root()

    run_dir = os.path.join("Data", f"{args.material}_run_{args.run_id}")

    if args.score:
        do_score(args.material, args.top, args.k4_threshold,
                 run_dir=run_dir, master_folder=master_folder, plot=args.plot,
                 export=args.export)
        return

    run_dir = prepare_run_dir(args.run_id, args.material)

    config = load_fit_config(run_dir)
    all_configs = build_grid(config)
    total = len(all_configs)

    if args.end is None:
        args.end = total

    print(f"Grid size: {total} combinations")
    print(f"Running chunk: [{args.start}, {args.end})")
    print(f"Run ID: {args.run_id} -> {run_dir}")
    print()

    run_chunk(args.material, master_folder, args.start, args.end,
              seed=args.seed, run_dir=run_dir)


if __name__ == "__main__":
    main()
