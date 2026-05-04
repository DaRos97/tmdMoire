"""Run monolayer fitting across a grid of constraint weights.

Executes the full fitting pipeline for each combination of constraint
weights (K1-K6) defined in ``Inputs/grid_config.json``. A snapshot of
the config is copied to ``Data/run_<id>/grid_config.json`` for
reproducibility. Results are saved as
``Data/run_<id>/fit_{TMD}_idx{N}.npz`` files.

Usage
-----
::

    python scripts/run_grid.py WSe2                          # All combinations
    python scripts/run_grid.py WSe2 --start 0 --end 100      # Chunk
    python scripts/run_grid.py WSe2 --run-id 001             # Named run
    python scripts/run_grid.py WSe2 --score                   # Score existing results
    python scripts/run_grid.py WSe2 --score --run-id 001      # Score specific run
    python scripts/run_grid.py WSe2 --score --top 20          # Top 20
    python scripts/run_grid.py WSe2 --score --k4-threshold 0.1

Arguments
---------
- ``WSe2`` or ``WS2``: Target material.
- ``--start``, ``--end``: Run only a subset of indices (for HPC chunking).
- ``--run-id``: Subdirectory name under Data/ for results (default: "default").
- ``--score``: Skip fitting, just score and display existing results.
- ``--top N``: Show top N results when scoring (default: 10).
- ``--k4-threshold``: K4 hard filter threshold for scoring (default: 0.05).

Examples
--------
Run all 3600 combinations for WSe2::

    python scripts/run_grid.py WSe2

Run indices 0-399 (submit as HPC job 1)::

    python scripts/run_grid.py WSe2 --start 0 --end 400

Run with a named subdirectory::

    python scripts/run_grid.py WSe2 --run-id 001

Score existing results::

    python scripts/run_grid.py WSe2 --score

Score a specific run::

    python scripts/run_grid.py WSe2 --score --run-id 001
"""
import sys
import os
import json
import shutil
import argparse
import itertools
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tmdmoire import (
    TMDMaterial, ARPESData, ParameterFitter, GridScorer,
    detect_machine, get_master_folder,
)

SOURCE_CONFIG = "Inputs/grid_config.json"


def prepare_run_dir(run_id: str) -> str:
    """Create the run output directory and copy grid_config.json into it.

    Reads the source config from ``Inputs/grid_config.json`` and writes
    a snapshot to ``Data/run_<id>/grid_config.json``.

    Parameters
    ----------
    run_id : str
        Run identifier.

    Returns
    -------
    str
        Path to the run directory.
    """
    run_dir = os.path.join("Data", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    dst = os.path.join(run_dir, "grid_config.json")
    if not os.path.exists(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    elif os.path.getmtime(SOURCE_CONFIG) > os.path.getmtime(dst):
        shutil.copy2(SOURCE_CONFIG, dst)

    return run_dir


def load_grid_config(run_dir: str) -> dict:
    """Load grid configuration from the run directory.

    Parameters
    ----------
    run_dir : str
        Path to the run directory containing grid_config.json.

    Returns
    -------
    dict
        Configuration with keys: grid, bounds, pts, maxiter, seed.
    """
    config_path = os.path.join(run_dir, "grid_config.json")
    with open(config_path) as f:
        return json.load(f)


def build_grid(config: dict) -> list[dict]:
    """Build the full list of parameter configurations from the grid spec.

    Parameters
    ----------
    config : dict
        Grid configuration from ``load_grid_config()``.

    Returns
    -------
    list[dict]
        List of configuration dicts, each with keys: idx, Ks, pts,
        boundType, Bs.
    """
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
            "boundType": config["bounds"]["boundType"],
            "Bs": tuple(config["bounds"]["Bs"]),
        })
    return configs


def run_chunk(material_name: str, master_folder: str,
              start: int, end: int, seed: int = 42,
              maxiter: int = 3000, run_dir: str = "Data") -> None:
    """Run fitting for a chunk of grid indices.

    Parameters
    ----------
    material_name : str
        "WSe2" or "WS2".
    master_folder : str
        Root directory of the project.
    start : int
        First index (inclusive).
    end : int
        Last index (exclusive).
    seed : int
        Random seed for reproducibility.
    maxiter : int
        Maximum dual annealing iterations per fit.
    run_dir : str
        Directory to save .npz results and read grid_config.json from.
    """
    config = load_grid_config(run_dir)
    all_configs = build_grid(config)
    total = len(all_configs)

    if start >= total:
        print(f"Start index {start} exceeds total grid size {total}")
        return
    end = min(end, total)

    material = TMDMaterial(material_name)
    pts = config.get("pts", 91)
    arpes_data = ARPESData(material_name, master_folder, pts=pts)

    print(f"Running indices {start} to {end - 1} / {total}")
    print(f"Material: {material_name}, pts: {pts}, maxiter: {maxiter}, seed: {seed}")
    print(f"Output directory: {run_dir}")
    print()

    t_start = time.time()
    for i in range(start, end):
        cfg = all_configs[i]
        fitter = ParameterFitter(material, arpes_data, cfg)

        t_fit = time.time()
        result = fitter.run(maxiter=maxiter, seed=seed)
        result["idx"] = i
        result["seed"] = seed

        fn = fitter.save(result, output_dir=run_dir)
        t_fit = time.time() - t_fit

        print(f"[{i:4d}/{total}] chi2={result['fun']:.6f}  "
              f"chi2_band={result.get('chi2_band', 'N/A')}  "
              f"nfev={result['nfev']}  "
              f"time={t_fit:.1f}s  "
              f"saved: {fn.name}")

    t_total = time.time() - t_start
    print(f"\nDone. {end - start} fits in {t_total:.1f}s "
          f"({t_total / (end - start):.1f}s avg)")


def do_score(material_name: str, top_n: int, k4_threshold: float,
             run_dir: str = "Data") -> None:
    """Load and score existing results.

    Parameters
    ----------
    material_name : str
        "WSe2" or "WS2".
    top_n : int
        Number of top results to display.
    k4_threshold : float
        K4 hard filter threshold.
    run_dir : str
        Directory containing fit_*.npz files and grid_config.json.
    """
    scorer = GridScorer(material_name, data_dir=run_dir)
    print(scorer.summary(k4_threshold=k4_threshold, top_n=top_n))


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
    parser.add_argument("--maxiter", type=int, default=3000,
                        help="Max dual annealing iterations per fit.")

    args = parser.parse_args()

    machine = detect_machine(os.getcwd())
    master_folder = get_master_folder(os.getcwd())

    run_dir = os.path.join("Data", f"run_{args.run_id}")

    if args.score:
        do_score(args.material, args.top, args.k4_threshold, run_dir=run_dir)
        return

    run_dir = prepare_run_dir(args.run_id)

    config = load_grid_config(run_dir)
    all_configs = build_grid(config)
    total = len(all_configs)

    if args.end is None:
        args.end = total

    print(f"Grid size: {total} combinations")
    print(f"Running chunk: [{args.start}, {args.end})")
    print(f"Run ID: {args.run_id} -> {run_dir}")
    print()

    run_chunk(args.material, master_folder, args.start, args.end,
              seed=args.seed, maxiter=args.maxiter, run_dir=run_dir)


if __name__ == "__main__":
    main()
