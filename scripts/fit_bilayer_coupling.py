"""Fit bilayer interlayer coupling parameters to ARPES data.

Fits 4 interlayer hopping parameters (w1p, w1d, w2p, w2d) to reproduce
the 3 top valence bands from WSe2/WS2 bilayer ARPES data along the Γ–K path.

Usage
-----
::

    python scripts/fit_bilayer_coupling.py
    python scripts/fit_bilayer_coupling.py --stacking AP
    python scripts/fit_bilayer_coupling.py --n-starts 20 --seed 123

Arguments
---------
- ``--stacking``: 'P' (parallel, default) or 'AP' (anti-parallel).
- ``--n-starts``: Number of random starting points (default: 10).
- ``--seed``: Random seed (default: 42).
- ``--n-kpts``: Number of equidistant k-points between Γ and K
  (used for both ARPES interpolation and Hamiltonian, default: 51).

Output
------
Fitted parameters saved to ``Inputs/bilayer_fitting/interlayer_params.npy``.
Metadata saved to ``Inputs/bilayer_fitting/interlayer_params_metadata.json``.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tmdmoire import (
    TMDMaterial, BilayerFitter,
    get_repo_root,
)
from tmdmoire.plotting.bilayer import plot_bilayer_fit

master_folder = get_repo_root()

parser = argparse.ArgumentParser(description="Fit bilayer interlayer coupling.")
parser.add_argument("--stacking", type=str, default="P", choices=["P", "AP"],
                    help="Stacking type: P (parallel) or AP (anti-parallel).")
parser.add_argument("--n-starts", type=int, default=10,
                    help="Number of random starting points.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--n-kpts", type=int, default=51,
                    help="Number of equidistant k-points between Γ and K.")
parser.add_argument("--verbose", action="store_true",
                    help="Print fit configuration before starting.")
args = parser.parse_args()

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/bilayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/bilayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

if args.verbose:
    print("------------BILAYER INTERLAYER COUPLING FIT------------")
    print(f" Stacking: {args.stacking}")
    print(f" k-points: {args.n_kpts} (Γ to K, equidistant)")
    print(f" Bounds: [-10, 10] eV")
    print(f" Starts: {args.n_starts}, seed: {args.seed}")
    print("-" * 50)

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

fitter = BilayerFitter(wse2, ws2, master_folder,
                       stacking=args.stacking,
                       n_kpts=args.n_kpts)

result = fitter.run(n_starts=args.n_starts, seed=args.seed)

output_dir = os.path.join(master_folder, "Inputs", "bilayer_fitting")
fn = fitter.save(result, output_dir)

print(f"\nBest chi2: {result['fun']:.6f}")
print(f"Evaluations: {result['nfev']}")
print(f"Success: {result['success']}")
print(f"\nFitted parameters:")
print(f"  w1p = {result['x'][0]:+.4f} eV")
print(f"  w1d = {result['x'][1]:+.4f} eV")
print(f"  w2p = {result['x'][2]:+.4f} eV")
print(f"  w2d = {result['x'][3]:+.4f} eV")
print(f"\nSaved to: {fn}")

plot_bilayer_fit(fitter.bilayer_data, result["k_list"],
                 result["evals"], result["evals_no_coupling"])
