"""Benchmark single EDC computation time."""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from tmdmoire import TMDMaterial, MoireGeometry, EDCAnalyzer, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

Vg = 0.015
phiG_deg = 180
phiG = phiG_deg / 180 * np.pi
spreadE = 0.03
sample = "S11"
n_shells = 2
theta = TWIST_ANGLES[sample]

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

geometry = MoireGeometry(theta)
n_cells = MoireGeometry.n_cells(n_shells)

k_list = np.array([np.zeros(2)])
Vk = 0.0
phiK = 0.0
pars_V = (Vg, Vk, phiG, phiK)

config = {
    "n_shells": n_shells,
    "n_cells": n_cells,
    "k_point": k_list,
    "theta_deg": theta,
    "interlayer_params": pars_interlayer,
    "pars_V": pars_V,
}

analyzer = EDCAnalyzer(wse2, ws2, geometry, config)

# Warmup
analyzer.compute_edc((Vg, phiG), "G", spreadE=spreadE, sample=sample)

N = 5
t0 = time.perf_counter()
for _ in range(N):
    analyzer.compute_edc((Vg, phiG), "G", spreadE=spreadE, sample=sample)
elapsed = time.perf_counter() - t0

print(f"n_cells = {n_cells}, Hamiltonian size = {44 * n_cells}x{44 * n_cells}")
print(f"Average time per point: {elapsed / N * 1000:.1f} ms")
print(f"Wall time for {N} runs: {elapsed:.2f} s")
