"""Check orbital occupations at K for DFT-derived parameters."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg as la
from tmdmoire.material import TMDMaterial
from tmdmoire.monolayer.data import MonolayerData
from tmdmoire.constants import ORBITAL_CHARACTER, x2_i, xy_i
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

for tmd in ["WSe2", "WS2"]:
    material = TMDMaterial(tmd)
    data = MonolayerData(tmd, master_folder, pts=91)

    dft = material.dft_params
    hopping = material.build_hopping_matrices(dft)
    epsilon = material.build_onsite_energies(dft)
    offset = dft[-3]
    hso = material.build_soc_hamiltonian(dft[-2:])
    args_h = (hopping, epsilon, hso, offset)

    from tmdmoire.monolayer.hamiltonian import MonolayerHamiltonian
    ham = MonolayerHamiltonian(material)
    H_K = ham.build(np.array([data.K]), *args_h)
    evals_K, evecs_K = la.eigh(H_K[0])

    # d_2 = (d_x2-y2 - i*d_xy) / sqrt(2)
    # TVB1 = band 13, TVB2 = band 12
    d2_tvb1 = (np.absolute(1/np.sqrt(2) * (evecs_K[x2_i, 13] - 1j * evecs_K[xy_i, 13])) ** 2
               + np.absolute(1/np.sqrt(2) * (evecs_K[x2_i + 11, 13] - 1j * evecs_K[xy_i + 11, 13])) ** 2)
    d2_tvb2 = (np.absolute(1/np.sqrt(2) * (evecs_K[x2_i, 12] - 1j * evecs_K[xy_i, 12])) ** 2
               + np.absolute(1/np.sqrt(2) * (evecs_K[x2_i + 11, 12] - 1j * evecs_K[xy_i + 11, 12])) ** 2)

    # Also compute p_-1^e for completeness
    from tmdmoire.constants import xe_i, ye_i
    p1_tvb1 = (np.absolute(-1/np.sqrt(2) * (evecs_K[xe_i, 13] - 1j * evecs_K[ye_i, 13])) ** 2
               + np.absolute(-1/np.sqrt(2) * (evecs_K[xe_i + 11, 13] - 1j * evecs_K[ye_i + 11, 13])) ** 2)
    p1_tvb2 = (np.absolute(-1/np.sqrt(2) * (evecs_K[xe_i, 12] - 1j * evecs_K[ye_i, 12])) ** 2
               + np.absolute(-1/np.sqrt(2) * (evecs_K[xe_i + 11, 12] - 1j * evecs_K[ye_i + 11, 12])) ** 2)

    # DFT targets from ORBITAL_CHARACTER
    occ_k = ORBITAL_CHARACTER[tmd]["K"]  # (p_-1 tvb1, p_-1 tvb2, d_2 tvb1, d_2 tvb2)

    print(f"\n{'='*50}")
    print(f"  {tmd}  (DFT-derived)")
    print(f"{'='*50}")
    print(f"  Band energies at K (TVB region):")
    print(f"    TVB1 (band 12): {evals_K[12]:.4f} eV")
    print(f"    TVB2 (band 13): {evals_K[13]:.4f} eV")
    print(f"    CBM  (band 14): {evals_K[14]:.4f} eV")
    print(f"")
    print(f"  Orbital occupations at K (computed from DFT H):")
    print(f"    p_-1^e (TVB1): {p1_tvb1:.4f}")
    print(f"    p_-1^e (TVB2): {p1_tvb2:.4f}")
    print(f"    d_2    (TVB1): {d2_tvb1:.4f}")
    print(f"    d_2    (TVB2): {d2_tvb2:.4f}")
    print(f"")
    print(f"  ORBITAL_CHARACTER targets (from config):")
    print(f"    p_-1^e (TVB1): {occ_k[0]:.4f}")
    print(f"    p_-1^e (TVB2): {occ_k[1]:.4f}")
    print(f"    d_2    (TVB1): {occ_k[2]:.4f}")
    print(f"    d_2    (TVB2): {occ_k[3]:.4f}")
    print(f"")
    print(f"  Differences (computed - target):")
    print(f"    p_-1^e (TVB1): {p1_tvb1 - occ_k[0]:.6f}")
    print(f"    p_-1^e (TVB2): {p1_tvb2 - occ_k[1]:.6f}")
    print(f"    d_2    (TVB1): {d2_tvb1 - occ_k[2]:.6f}")
    print(f"    d_2    (TVB2): {d2_tvb2 - occ_k[3]:.6f}")
