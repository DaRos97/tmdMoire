"""Check K5 in detail: offset vs relative band structure."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.linalg as la
from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.monolayer.fitter import ParameterFitter
from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian
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

    bilayer_dir = os.path.join(master_folder, "Inputs", "bilayer_fitting")
    matches = [f for f in os.listdir(bilayer_dir) if f.startswith(f"tb_{tmd}") and f.endswith(".npy")]
    fit_pars = np.load(os.path.join(bilayer_dir, matches[0]))

    print(f"\n{'='*50}")
    print(f"  {tmd}")
    print(f"{'='*50}")

    dft = material.dft_params
    ham = MonolayerHamiltonian(material)

    # DFT bands at K
    H_K_dft = ham.build(np.array([data.K]),
                        material.build_hopping_matrices(dft),
                        material.build_onsite_energies(dft),
                        material.build_soc_hamiltonian(dft[-2:]),
                        dft[-3])
    evals_dft = la.eigvalsh(H_K_dft[0])

    # Fitted bands at K
    H_K_fit = ham.build(np.array([data.K]),
                        material.build_hopping_matrices(fit_pars),
                        material.build_onsite_energies(fit_pars),
                        material.build_soc_hamiltonian(fit_pars[-2:]),
                        fit_pars[-3])
    evals_fit = la.eigvalsh(H_K_fit[0])

    print(f"  Offset: DFT={dft[-3]:.4f}, Fitted={fit_pars[-3]:.4f}, diff={fit_pars[-3]-dft[-3]:.4f}")
    print(f"")
    print(f"  Band energies at K:")
    print(f"  {'Band':>6} {'DFT':>10} {'Fitted':>10} {'Diff':>10}")
    for i in range(11, 16):
        diff = evals_fit[i] - evals_dft[i]
        print(f"  {i:>6} {evals_dft[i]:>10.4f} {evals_fit[i]:>10.4f} {diff:>10.4f}")

    gap_dft = evals_dft[14] - evals_dft[13]
    gap_fit = evals_fit[14] - evals_fit[13]
    print(f"\n  Gap (CBM - TVB2): DFT={gap_dft:.6f}, Fitted={gap_fit:.6f}")
    print(f"  K5 = {abs(gap_dft - gap_fit)/gap_dft:.10f}")

    # Check if all bands shifted by the same amount (pure offset effect)
    shifts = [evals_fit[i] - evals_dft[i] for i in range(11, 16)]
    print(f"\n  Band shifts (fitted - DFT): {[f'{s:.4f}' for s in shifts]}")
    print(f"  Shift range: {max(shifts)-min(shifts):.6f} eV")
