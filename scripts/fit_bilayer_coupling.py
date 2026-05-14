"""Fit bilayer interlayer coupling parameters to ARPES data.

Fits 4 interlayer hopping parameters (w1p, w1d, w2p, w2d) to reproduce
the 3 top valence bands from bilayer ARPES data along the Γ–K path.

Usage
-----
    python scripts/fit_bilayer_coupling.py
    python scripts/fit_bilayer_coupling.py --n-starts 20 --seed 123
    python scripts/fit_bilayer_coupling.py --debug --debug-max 10

Arguments
---------
- ``--n-kpts``: Number of k-points between Γ and K (default: 51).
- ``--n-starts``: Number of random starting points (default: 10).
- ``--seed``: Random seed (default: 42).
- ``--debug``: Save iteration plots to Data/debugging/.
- ``--debug-max``: Stop after N iterations.
- ``--verbose``: Print configuration.

Output
------
Fitted parameters saved to Inputs/bilayer_fitting/interlayer_params.npy.
Metadata saved to Inputs/bilayer_fitting/interlayer_params_metadata.json.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tmdmoire import TMDMaterial, BilayerFitter, get_repo_root
from tmdmoire.constants import ENERGY_OFFSETS
from tmdmoire.plotting.bilayer import plot_bilayer_fit

master_folder = get_repo_root()

parser = argparse.ArgumentParser(description="Fit bilayer interlayer coupling.")
parser.add_argument("--n-kpts", type=int, default=51,
                    help="Number of k-points between Gamma and K.")
parser.add_argument("--n-starts", type=int, default=10,
                    help="Number of random starting points.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--debug", action="store_true",
                    help="Save iteration plots to Data/debugging/.")
parser.add_argument("--debug-max", type=int, default=None,
                    help="Stop after N iterations.")
parser.add_argument("--debug-every", type=int, default=1,
                    help="Save debug plot every N accepted steps.")
parser.add_argument("--verbose", action="store_true",
                    help="Print configuration.")
args = parser.parse_args()

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/bilayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/bilayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

if args.verbose:
    print("------------BILAYER INTERLAYER COUPLING FIT------------")
    print(f" k-points: {args.n_kpts}")
    print(f" Bounds: all params [-5, 5]")
    print(f" Starts: {args.n_starts}, seed: {args.seed}")
    print("-" * 50)

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

fitter = BilayerFitter(wse2, ws2, master_folder, n_kpts=args.n_kpts)

# Pre-fit diagnostic plot
evals_nc, _ = fitter._build_hamiltonian(0.0, 0.0, 0.0, 0.0)
evals_nc = evals_nc + ENERGY_OFFSETS["S11"]
plot_bilayer_fit(fitter.bilayer_data, fitter.k_list, evals_nc, evals_nc)

debug_dir = os.path.join(master_folder, "Data", "debugging") if args.debug else None
result = fitter.run(n_starts=args.n_starts, seed=args.seed,
                    debug_dir=debug_dir, debug_max_iters=args.debug_max,
                    debug_every=args.debug_every)

output_dir = os.path.join(master_folder, "Inputs", "bilayer_fitting")
fn = fitter.save(result, output_dir)

print(f"\nBest chi2: {result['fun']:.6f}")
print(f"Evaluations: {result['nfev']}")
print(f"Success: {result['success']}")
print(f"\nFitted parameters:")
for i, name in enumerate(["w1p", "w1d", "w2p", "w2d"]):
    print(f"  {name} = {result['x'][i]:+.4f} eV")
print(f"\nSaved to: {fn}")

# Post-fit plot
plot_bilayer_fit(fitter.bilayer_data, result["k_list"],
                 result["evals"], result["evals_no_coupling"])
