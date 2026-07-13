"""Smoke test: compute a single Gamma EDC for S11, S11_m25, S11_p31.

Verifies that each sample's twist angle produces a valid diagonalization
and 4-Lorentzian EDC fit. Reports moire geometry and fitted peak centers.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import lmfit

from tmdmoire import TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

SAMPLES = ["S11", "S11_m25", "S11_p31"]

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

interlayer_params = np.load(master_folder + "/Inputs/bilayer_fitting/interlayer_params.npy")
w1p, w1d, w2p, w2d = interlayer_params
pars_interlayer = {"stacking": "P", "w1p": w1p, "w2p": w2p, "w1d": w1d, "w2d": w2d}

Vg = 0.015
phiG_deg = 180
phiG = phiG_deg / 180 * np.pi
Vk = 0.0077
phiK_deg = 106
phiK = phiK_deg / 180 * np.pi
spreadE = 0.03
n_shells = 2


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def _four_lorentzian(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
    return (_lorentzian(x, a1, c1, g1) + _lorentzian(x, a2, c2, g2)
            + _lorentzian(x, a3, c3, g3) + _lorentzian(x, a4, c4, g4))


def compute_one(sample):
    theta = TWIST_ANGLES[sample]
    geometry = MoireGeometry(theta)
    n_cells = MoireGeometry.n_cells(n_shells)
    k_list = np.array([np.zeros(2)])
    pars_V = (Vg, Vk, phiG, phiK)

    moire_ham = MoireHamiltonian(wse2, ws2, geometry)
    evals_raw, evecs_raw = moire_ham.diagonalize(k_list, n_shells, pars_interlayer, pars_V)
    evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_raw[0]

    ab = np.absolute(evecs_raw) ** 2
    weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0)

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_l = index_lvb - 2 * n_cells + 1

    full_e = evals_raw[index_l:index_tvb + 1]
    full_w = weights[index_l:index_tvb + 1]

    min_e = full_e[0]
    max_e = full_e[-1]
    delta = max_e - min_e
    min_e -= delta / 2
    max_e += delta / 2
    n_e = int((max_e - min_e) / 0.005)
    energy_list = np.linspace(min_e, max_e, n_e)
    weight_list = np.zeros(len(energy_list))

    for i in range(len(full_e)):
        weight_list += spreadE / np.pi * full_w[i] / (
            (energy_list - full_e[i]) ** 2 + spreadE ** 2
        )

    sorted_idx = np.argsort(full_w)[::-1]
    peak_states = []
    seen = []
    for si in sorted_idx:
        e = full_e[si]
        w = full_w[si]
        if w < 1e-4:
            break
        if not any(abs(e - c) < 0.01 for c in seen):
            peak_states.append((e, w))
            seen.append(e)
        if len(peak_states) == 4:
            break

    if len(peak_states) < 4:
        return None

    peak_states.sort(key=lambda x: x[0], reverse=True)

    model = lmfit.Model(_four_lorentzian)
    params_fit = model.make_params(
        a1=peak_states[0][1], c1=peak_states[0][0], g1=spreadE,
        a2=peak_states[1][1], c2=peak_states[1][0], g2=spreadE,
        a3=peak_states[2][1], c3=peak_states[2][0], g3=spreadE,
        a4=peak_states[3][1], c4=peak_states[3][0], g4=spreadE,
    )
    for p in ["a1", "a2", "a3", "a4"]:
        params_fit[p].set(min=0)
    for p in ["g1", "g2", "g3", "g4"]:
        params_fit[p].set(min=1e-4, max=0.2)
    for i, p in enumerate(["c1", "c2", "c3", "c4"]):
        seed = peak_states[i][0]
        params_fit[p].set(min=seed - 0.05, max=seed + 0.05)

    result = model.fit(weight_list, params_fit, x=energy_list)

    return {
        "success": result.success,
        "redchi": result.redchi,
        "c1": result.best_values["c1"],
        "c2": result.best_values["c2"],
        "c3": result.best_values["c3"],
        "a_moire": geometry.moire_length,
        "n_cells": n_cells,
        "ham_dim": 44 * n_cells,
    }


print(f"{'Sample':<12} {'theta':>6} {'a_moire':>8} {'H dim':>6} {'fit':>6} {'redchi':>8} {'c1 (eV)':>8} {'c2 (eV)':>8} {'c3 (eV)':>8}")
print("-" * 78)

all_ok = True
for s in SAMPLES:
    r = compute_one(s)
    if r is None:
        print(f"{s:<12} {'─':>6} {'─':>8} {'─':>6} {'FAIL':>6} {'─':>8} {'─':>8} {'─':>8} {'─':>8}")
        all_ok = False
        continue
    status = "OK" if r["success"] else "FAIL"
    if not r["success"]:
        all_ok = False
    print(f"{s:<12} {TWIST_ANGLES[s]:>5.1f}° {r['a_moire']:>7.2f}Å {r['ham_dim']:>5}×{r['ham_dim']} "
          f"{status:>5} {r['redchi']:>8.4f} {r['c1']:>8.4f} {r['c2']:>8.4f} {r['c3']:>8.4f}")

print()
if all_ok:
    print("All samples OK")
else:
    print("SOME SAMPLES FAILED")
    sys.exit(1)
