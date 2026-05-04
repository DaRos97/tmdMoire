"""Fit monolayer tight-binding parameters to ARPES data.

This script fits 43 tight-binding parameters for a single TMD monolayer
(WSe2 or WS2) to reproduce ARPES-measured band dispersions along the
high-symmetry paths K'-Gamma-K and K-M-K'.

The fitting uses dual annealing (global search) followed by Nelder-Mead
(local refinement) with a weighted chi-squared objective that combines
band dispersion matching with physical constraints (orbital character,
parameter distance from DFT, band gap, etc.).

Usage
-----
::

    python scripts/fit_monolayer.py <WSe2|WS2> <index>
    python scripts/fit_monolayer.py <WSe2|WS2> <index> --run-id 001

Arguments
---------
- ``WSe2`` or ``WS2``: Target material.
- ``index``: Integer selecting a combination of constraint weights (K1-K6)
  from the grid defined in ``Inputs/grid_config.json``.
- ``--run-id``: Run identifier. Results saved to Data/run_<id>/ (default: 'default').

Examples
--------
Fit WSe2 with constraint weight set 0::

    python scripts/fit_monolayer.py WSe2 0

Fit WS2 with constraint weight set 5 into a named run::

    python scripts/fit_monolayer.py WS2 5 --run-id 001

Output
------
Optimized parameters are saved to ``Data/run_<id>/fit_{TMD}_idx{N}.npz``.
"""
import sys
import os
import json
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itertools
from tmdmoire import (
    TMDMaterial, ARPESData, ParameterFitter,
    detect_machine, get_master_folder,
)

SOURCE_CONFIG = "Inputs/grid_config.json"

machine = detect_machine(os.getcwd())
master_folder = get_master_folder(os.getcwd())
disp = machine == "loc"

parser = argparse.ArgumentParser(description="Fit monolayer TB parameters.")
parser.add_argument("material", choices=["WSe2", "WS2"], help="Target material.")
parser.add_argument("index", type=int, help="Grid index.")
parser.add_argument("--run-id", type=str, default="default",
                    help="Run identifier for output subdirectory.")
args = parser.parse_args()

tmd_name = args.material
argc = args.index
if machine == "maf":
    argc -= 1


def prepare_run_dir(run_id: str, material: str) -> str:
    """Create the run output directory and copy grid_config.json into it."""
    run_dir = os.path.join("Data", f"{material}_run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    dst = os.path.join(run_dir, "grid_config.json")
    if not os.path.exists(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    elif os.path.getmtime(SOURCE_CONFIG) > os.path.getmtime(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    return run_dir


def get_args(tmd: str, ind: int, run_dir: str) -> dict:
    """Select constraint weights and bound parameters for a given index.

    Reads the grid definition from the run directory's grid_config.json.

    Parameters
    ----------
    tmd : str
        Material name (unused, kept for API compatibility).
    ind : int
        Index into the parameter grid.
    run_dir : str
        Directory containing grid_config.json.

    Returns
    -------
    dict
        Configuration with keys: ``idx``, ``pts``, ``Ks``, ``boundType``, ``Bs``, ``optimizer``.
    """
    config_path = os.path.join(run_dir, "grid_config.json")
    with open(config_path) as f:
        config = json.load(f)

    grid = config["grid"]
    keys = ["K1", "K2", "K3", "K4", "K5", "K6"]
    values = [grid[k] for k in keys]
    listPar = list(itertools.product(*values))

    if ind >= len(listPar):
        raise IndexError(
            f"Index {ind} out of range (grid has {len(listPar)} combinations)"
        )

    combo = listPar[ind]
    return {
        "idx": ind,
        "pts": config.get("pts", 91),
        "Ks": tuple(combo),
        "boundType": config["bounds"]["boundType"],
        "Bs": tuple(config["bounds"]["Bs"]),
        "optimizer": config.get("optimizer", {}),
    }


run_dir = prepare_run_dir(args.run_id, tmd_name)

args_minimization = get_args(tmd_name, argc, run_dir)
pts = args_minimization["pts"]

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(f" TMD: {tmd_name}")
    for i in range(6):
        print(f" K_{i+1}: {args_minimization['Ks'][i]:.6f}")
    opt = args_minimization.get("optimizer", {})
    print(f" Optimizer: da_maxiter={opt.get('da_maxiter', 100)}, "
          f"nm_maxiter={opt.get('nm_maxiter', 50)}, "
          f"nm_fatol={opt.get('nm_fatol', 1e-3)}")
    print(f" Using {pts} points of interpolated data.")
    print("-" * 15)

material = TMDMaterial(tmd_name)
arpes_data = ARPESData(tmd_name, master_folder, pts=pts)
fitter = ParameterFitter(material, arpes_data, args_minimization)

result = fitter.run(seed=42)
result["idx"] = argc
result["seed"] = 42

fn = fitter.save(result, output_dir=run_dir)

print(f"Final chi2: {result['fun']}")
print(f"Evaluations: {result['nfev']}")
print(f"Saved to: {fn}")
