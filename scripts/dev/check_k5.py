"""Check K5 (band gap at K) computation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.linalg as la
from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.fitter import ParameterFitter
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

for tmd in ["WSe2", "WS2"]:
    material = TMDMaterial(tmd)
    data = MonolayerData(tmd, master_folder, pts=91)

    config = {
        "Ks": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "boundType": "absolute",
        "Bs": (5, 2, 4, 1, 0),
    }
    fitter = ParameterFitter(material, data, config)

    print(f"\n{'='*50}")
    print(f"  {tmd}")
    print(f"{'='*50}")

    # DFT gap
    dft = material.dft_params
    hopping = material.build_hopping_matrices(dft)
    epsilon = material.build_onsite_energies(dft)
    offset = dft[-3]
    hso = material.build_soc_hamiltonian(dft[-2:])
    args_h = (hopping, epsilon, hso, offset)

    from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian
    ham = MonolayerHamiltonian(material)
    H_K = ham.build(np.array([data.K]), *args_h)
    evals_K = la.eigvalsh(H_K[0])

    print(f"  DFT bands at K:")
    print(f"    TVB1 (12): {evals_K[12]:.4f} eV")
    print(f"    TVB2 (13): {evals_K[13]:.4f} eV")
    print(f"    CBM  (14): {evals_K[14]:.4f} eV")
    print(f"    DFT gap:   {evals_K[14] - evals_K[13]:.4f} eV")
    print(f"    _gap_DFT:  {fitter._gap_DFT:.4f} eV")

    # Now check with the exported fitted params
    bilayer_dir = os.path.join(master_folder, "Inputs", "bilayer_fitting")
    matches = [f for f in os.listdir(bilayer_dir) if f.startswith(f"tb_{tmd}") and f.endswith(".npy")]
    if matches:
        fit_pars = np.load(os.path.join(bilayer_dir, matches[0]))
        hopping = material.build_hopping_matrices(fit_pars)
        epsilon = material.build_onsite_energies(fit_pars)
        offset = fit_pars[-3]
        hso = material.build_soc_hamiltonian(fit_pars[-2:])
        args_h = (hopping, epsilon, hso, offset)

        H_K = ham.build(np.array([data.K]), *args_h)
        evals_K = la.eigvalsh(H_K[0])

        gap_p = evals_K[14] - evals_K[13]
        K5_gap = abs(fitter._gap_DFT - gap_p) / fitter._gap_DFT

        print(f"\n  Fitted bands at K:")
        print(f"    TVB1 (12): {evals_K[12]:.4f} eV")
        print(f"    TVB2 (13): {evals_K[13]:.4f} eV")
        print(f"    CBM  (14): {evals_K[14]:.4f} eV")
        print(f"    Fitted gap: {gap_p:.4f} eV")
        print(f"    K5_gap:     {K5_gap:.6f}")
