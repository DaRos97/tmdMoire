"""Fit monolayer tight-binding parameters to ARPES data.

This script fits 43 tight-binding parameters for a single TMD monolayer
(WSe2 or WS2) to reproduce ARPES-measured band dispersions along the
high-symmetry paths K′–Γ–K and K–M–K′.

The fitting uses dual annealing (global search) followed by Nelder-Mead
(local refinement) with a weighted chi-squared objective that combines
band dispersion matching with physical constraints (orbital character,
parameter distance from DFT, band gap, etc.).

Usage
-----
::

    python scripts/fit_monolayer.py <WSe2|WS2> <index>

Arguments
---------
- ``WSe2`` or ``WS2``: Target material.
- ``index``: Integer selecting a combination of constraint weights (K₁–K₆)
  from a parameter grid. There are 2×10×10×2×2×2 = 1600 combinations.

Examples
--------
Fit WSe2 with constraint weight set 0::

    python scripts/fit_monolayer.py WSe2 0

Fit WS2 with constraint weight set 5::

    python scripts/fit_monolayer.py WS2 5

Output
------
Optimized parameters are printed to stdout. Intermediate results during
minimization are saved as ``.npz`` files in the ``Data/`` directory.

Notes
-----
- On the Mafalda cluster (``machine=='maf'``), the index is shifted by −1.
- Setting SOC bounds to 0 fixes the SOC Hamiltonian to DFT values and
  fits only the 41 tight-binding parameters.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tmdmoire import (
    TMDMaterial, ARPESData, ParameterFitter,
    detect_machine, get_master_folder, DFT_INITIAL_PARAMS,
)

machine = detect_machine(os.getcwd())
master_folder = get_master_folder(os.getcwd())
disp = machine == "loc"
maxiter = 3000

if len(sys.argv) != 3:
    print("Usage: python scripts/fit_monolayer.py <WSe2|WS2> <index>")
    sys.exit(1)

tmd_name = sys.argv[1]
if tmd_name not in ["WSe2", "WS2"]:
    raise ValueError(f"Unknown TMD: {tmd_name}")

argc = int(sys.argv[2])
if machine == "maf":
    argc -= 1


def get_args(tmd, ind):
    """Select constraint weights and bound parameters for a given index.

    Generates a grid of constraint weight combinations (K₁–K₆) and
    returns the configuration for the specified index.

    Parameters
    ----------
    tmd : str
        Material name (unused, kept for API compatibility).
    ind : int
        Index into the parameter grid.

    Returns
    -------
    dict
        Configuration with keys: ``pts``, ``Ks``, ``boundType``, ``Bs``.
    """
    import itertools
    lK1 = [0, 1e-5]
    lK2 = np.logspace(-7, 1, 10, base=2)
    lK3 = np.logspace(-7, 1, 10, base=2)
    lK4 = [0.5, 1]
    lK5 = [0.1, 0.5]
    lK6 = [1, 5]
    boundType = "absolute"
    Bs = (5, 2, 4, 1, 0)
    pts = 91
    listPar = list(itertools.product(*[lK1, lK2, lK3, lK4, lK5, lK6]))
    print(f"Index {ind} / {len(listPar)}")
    listPar = listPar[ind]
    return {
        "pts": pts,
        "Ks": tuple(listPar),
        "boundType": boundType,
        "Bs": Bs,
    }


args_minimization = get_args(tmd_name, argc)
pts = args_minimization["pts"]

if disp:
    print("------------CHOSEN PARAMETERS------------")
    print(f" TMD: {tmd_name}")
    for i in range(6):
        print(f" K_{i+1}: {args_minimization['Ks'][i]:.6f}")
    print(f" Using {pts} points of interpolated data.")
    print("-" * 15)

material = TMDMaterial(tmd_name)
arpes_data = ARPESData(tmd_name, master_folder, pts=pts)
fitter = ParameterFitter(material, arpes_data, args_minimization)

result = fitter.run(maxiter=maxiter, seed=42)

print(f"Final chi2: {result['fun']}")
print(f"Final parameters: {result['x']}")
print(f"Evaluations: {result['nfev']}")
