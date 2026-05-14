"""Sweep moire potential and interlayer coupling parameters at Gamma.

6D grid: V (moire amplitude), phi (moire phase), w1p, w1d, w2p, w2d.
Computes 3-peak EDC fit at Gamma for each grid point.

Usage: python scripts/sweep_moire_params.py <chunk_index>
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from itertools import product, islice
from tmdmoire import (
    TMDMaterial, MoireGeometry, EDCAnalyzer,
    get_repo_root, get_filename,
    TWIST_ANGLES, LATTICE_CONSTANTS,
)

master_folder = get_repo_root()
n_chunks = 128

BILAYER_DATA_DIR = "/home/users/r/rossid/bilayer_v2.0/Data/"

if len(sys.argv) != 2:
    print("Usage: python3 scripts/sweep_moire_params.py <chunk_index>")
    sys.exit(1)

ind = int(sys.argv[1])
if ind < 0 or ind >= n_chunks:
    raise ValueError(f"Index out of range: {ind}")

sample = "S11"
theta_deviation = 0
n_shells = 2
spreadE = 0.03
theta = TWIST_ANGLES[sample] + theta_deviation
stacking = "P"

interlayer_params_path = master_folder + "/Inputs/bilayer_fitting/interlayer_params.npy"
interlayer_vals = np.load(interlayer_params_path)
w1p_start = interlayer_vals[0]
w1d_start = interlayer_vals[1]
w2p_start = interlayer_vals[2]
w2d_start = interlayer_vals[3]

print(f"Starting interlayer params: w1p={w1p_start:.4f}, w1d={w1d_start:.4f}, w2p={w2p_start:.4f}, w2d={w2d_start:.4f}")

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

listV = np.linspace(0.001, 0.040, 20)
listPhi = np.linspace(160, 180, 11) / 180 * np.pi
listW1p = np.linspace(w1p_start - 0.050, w1p_start + 0.050, 11)
listW1d = np.linspace(w1d_start - 0.050, w1d_start + 0.050, 11)
listW2p = np.linspace(w2p_start - 0.050, w2p_start + 0.050, 11)
listW2d = np.linspace(w2d_start - 0.050, w2d_start + 0.050, 11)

filename = get_filename(
    (listV[0], listV[-1], len(listV),
     int(listPhi[0] / np.pi * 180), int(listPhi[-1] / np.pi * 180), len(listPhi),
     listW1p[0], listW1p[-1], len(listW1p),
     listW1d[0], listW1d[-1], len(listW1d),
     listW2p[0], listW2p[-1], len(listW2p),
     listW2d[0], listW2d[-1], len(listW2d))
)

grid = product(listV, listPhi, listW1p, listW1d, listW2p, listW2d)
total_jobs = len(listV) * len(listPhi) * len(listW1p) * len(listW1d) * len(listW2p) * len(listW2d)

chunk_size = total_jobs // n_chunks
remainder = total_jobs % n_chunks
start = ind * chunk_size + min(ind, remainder)
end = start + chunk_size + (1 if ind < remainder else 0)
chunk_iter = islice(grid, start, end)

print(f"Total jobs: {total_jobs}")
print(f"This chunk ({ind}): {end - start} points (start={start}, end={end})")

columns = ["V", "phi", "w1p", "w1d", "w2p", "w2d", "p1", "p2", "p3"]

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

geometry = MoireGeometry(theta)
n_cells = MoireGeometry.n_cells(n_shells)

k_list = np.array([np.zeros(2)])
pars_V = (listV[0], 0.0, 160 / 180 * np.pi, 0.0)
pars_interlayer = {"stacking": stacking, "w1p": w1p_start, "w2p": w2p_start, "w1d": w1d_start, "w2d": w2d_start}

config = {
    "n_shells": n_shells,
    "n_cells": n_cells,
    "k_point": k_list,
    "theta_deg": theta,
    "interlayer_params": pars_interlayer,
    "pars_V": pars_V,
}

analyzer = EDCAnalyzer(wse2, ws2, geometry, config)

results = []

for pars in chunk_iter:
    V, phi, w1p, w1d, w2p, w2d = pars
    pars_interlayer["w1p"] = w1p
    pars_interlayer["w1d"] = w1d
    pars_interlayer["w2p"] = w2p
    pars_interlayer["w2d"] = w2d
    config["pars_V"] = (V, 0.0, phi, 0.0)
    config["interlayer_params"] = pars_interlayer

    positions, success = analyzer.compute_edc((V, phi), "G", spreadE=spreadE, sample=sample)

    if success:
        results.append((*pars, *positions))
    else:
        results.append((*pars, np.nan, np.nan, np.nan))

df = pd.DataFrame(results, columns=columns)

dirname = get_filename(
    ("edcGSweep", theta_deviation, n_shells, spreadE),
    dirname=BILAYER_DATA_DIR,
    float_precision=4,
) + "_" + filename + "/"

if not Path(dirname).is_dir():
    os.system(f"mkdir -p {dirname}")

output_file = dirname + f"chunk_{ind}_{n_chunks}.h5"
df.to_hdf(output_file, key="results", mode="w", format="table", complevel=5, complib="blosc")

print(f"Finished Gamma sweep chunk {ind}/{n_chunks}")
