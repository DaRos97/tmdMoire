"""Compute constraint breakdown for exported TB parameters."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.fitter import ParameterFitter
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

for tmd in ["WSe2", "WS2"]:
    # Find the TB file
    bilayer_dir = os.path.join(master_folder, "Inputs", "bilayer_fitting")
    matches = [f for f in os.listdir(bilayer_dir) if f.startswith(f"tb_{tmd}") and f.endswith(".npy")]
    if not matches:
        print(f"No tb_{tmd} file found")
        continue
    tb_path = os.path.join(bilayer_dir, matches[0])
    params = np.load(tb_path)

    material = TMDMaterial(tmd)
    data = MonolayerData(tmd, master_folder, pts=91)

    # Dummy config — K values don't affect the raw breakdown, only the weighted chi2
    config = {
        "Ks": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "boundType": "absolute",
        "Bs": (5, 2, 4, 1, 0),
    }
    fitter = ParameterFitter(material, data, config)

    # Need to pass params without SOC if SOC bounds are 0
    if params.shape[0] == 43:
        # SOC is included — use full params
        tb_params = params[:-2]
    else:
        tb_params = params

    breakdown = fitter._compute_constraint_breakdown(tb_params)

    print(f"\n{'='*50}")
    print(f"  {tmd}  ({matches[0]})")
    print(f"{'='*50}")
    print(f"  band_dist (unweighted):  {breakdown['chi2_band']:.6f}")
    print(f"  band_K6 (weighted):      {breakdown['chi2_band_weighted']:.6f}")
    print(f"  K1 (param distance):     {breakdown['K1']:.6f}")
    print(f"  K2 (orbital content M):  {breakdown['K2']:.6f}")
    print(f"  K3 (orbital occ G/K):    {breakdown['K3']:.6f}")
    print(f"  K4 (CBM at K):           {breakdown['K4']:.6f}")
    print(f"  K5 (band gap at K):      {breakdown['K5']:.6f}")
