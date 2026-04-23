"""EDC analysis at Gamma or K for heterobilayer parameter sweep.

Usage: python scripts/analyze_edc.py <G|K> <index>
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
    detect_machine, get_master_folder, get_home_dn, get_filename,
    TWIST_ANGLES, LATTICE_CONSTANTS,
)

machine = detect_machine(os.getcwd())
master_folder = get_master_folder(os.getcwd())
n_chunks = 128

if len(sys.argv) != 3:
    print("Usage: python3 scripts/analyze_edc.py <G|K> <index>")
    sys.exit(1)

sample = "S11"
bz_point = sys.argv[1]
if bz_point not in ["G", "K"]:
    raise ValueError(f"Unknown BZ point: {bz_point}")

ind = int(sys.argv[2])
if machine == "maf":
    ind -= 1
if ind < 0 or ind >= n_chunks:
    raise ValueError(f"Index out of range: {ind}")

disp = machine == "loc"
compute_gap = bz_point == "K"

theta_deviation = 0
n_shells = 2

if bz_point == "G":
    Vk, phiK = (0.0077, 106 / 180 * np.pi)
    k_list = np.array([np.zeros(2)])
    columns = ["Vg", "phiG", "w1p", "w1d", "p1", "p2", "p3"]
    columns_gap = ["Vg", "phiG", "w1p", "w1d", "gap"]
    args_fn = (Vk, phiK)
elif bz_point == "K":
    Vg, phiG = (0.0165, 175 / 180 * np.pi)
    w1p = -1.556
    w1d = 1.143
    k_list = np.array([[4 * np.pi / 3 / LATTICE_CONSTANTS["WSe2"], 0]])
    columns = ["Vk", "phiK", "p1", "p2"]
    columns_gap = ["Vk", "phiK", "gap"]
    args_fn = (Vg, phiG, w1p, w1d)

spreadE = 0.03
theta = TWIST_ANGLES[sample] + theta_deviation
stacking = "P"
w2p = w2d = 0

monolayer_fns = {
    "WSe2": master_folder + "Inputs/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "Inputs/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

pars_interlayer = {"stacking": stacking, "w1p": w1p, "w2p": w2p, "w1d": w1d, "w2d": w2d}
n_cells = MoireGeometry.n_cells(n_shells)


def get_parameters(chunk_id, bz_point, n_chunks=128):
    if bz_point == "G":
        listVg = np.linspace(0.010, 0.020, 51)
        listPhi = np.linspace(0, 359, 360) / 180 * np.pi
        listW1p = np.linspace(-1.580, -1.530, 11)
        listW1d = np.linspace(1.120, 1.170, 11)
        filename = get_filename(
            (listVg[0], listVg[-1], len(listVg), int(listPhi[0] / np.pi * 180),
             int(listPhi[-1] / np.pi * 180), len(listPhi),
             listW1p[0], listW1p[-1], len(listW1p), listW1d[0], listW1d[-1], len(listW1d))
        )
        grid = product(listVg, listPhi, listW1p, listW1d)
        total_jobs = len(listVg) * len(listPhi) * len(listW1p) * len(listW1d)
    elif bz_point == "K":
        listVk = np.linspace(0.001, 0.040, 196)
        listPhiK = np.linspace(0, 359, 360) / 180 * np.pi
        filename = get_filename(
            (listVk[0], listVk[-1], len(listVk), int(listPhiK[0] / np.pi * 180),
             int(listPhiK[-1] / np.pi * 180), len(listPhiK))
        )
        grid = product(listVk, listPhiK)
        total_jobs = len(listVk) * len(listPhiK)
    chunk_size = total_jobs // n_chunks
    remainder = total_jobs % n_chunks
    start = chunk_id * chunk_size + min(chunk_id, remainder)
    end = start + chunk_size + (1 if chunk_id < remainder else 0)
    chunk_iter = islice(grid, start, end)
    print(f"Total jobs: {total_jobs}")
    print(f"This chunk: {end - start}")
    return chunk_iter, filename


wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

geometry = MoireGeometry(theta)
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

parameters_chunk, list_fn = get_parameters(ind, bz_point, n_chunks=n_chunks)
results = []
results_gap = []

for pars in parameters_chunk:
    if bz_point == "G":
        Vg, phiG, w1p, w1d = pars
        if disp:
            print(f"Vg: {Vg:.3f}\tphiG: {phiG / np.pi * 180:.1f}\tw1p: {w1p:.3f}\tw1d: {w1d:.3f}")
        pars_interlayer["w1p"] = w1p
        pars_interlayer["w1d"] = w1d
        config["pars_V"] = (Vg, Vk, phiG, phiK)
    elif bz_point == "K":
        Vk, phiK = pars
        pars_interlayer["w1p"] = w1p
        pars_interlayer["w1d"] = w1d
        config["pars_V"] = (Vg, Vk, phiG, phiK)

    config["interlayer_params"] = pars_interlayer
    positions, success = analyzer.compute_edc(pars, bz_point, spreadE=spreadE, sample=sample)

    if success:
        if compute_gap:
            gap = analyzer.compute_gap(pars, bz_point)
            results_gap.append((*pars, gap))
        results.append((*pars, *positions))
    else:
        n_pos = 2 if bz_point == "K" else 3
        results.append((*pars, *np.full(n_pos, np.nan)))
        if compute_gap:
            results_gap.append((*pars, np.nan))

df = pd.DataFrame(results, columns=columns)
dirname = get_filename(
    ("edc" + bz_point, theta_deviation, n_shells, spreadE, *args_fn),
    dirname=get_home_dn(machine, "bilayer") + "Data/",
    float_precision=4,
) + "_" + list_fn + "/"
if not Path(dirname).is_dir():
    os.system(f"mkdir -p {dirname}")
output_file = dirname + f"chunk_{ind}_{n_chunks}.h5"
df.to_hdf(output_file, key="results", mode="w", format="table", complevel=5, complib="blosc")

if compute_gap:
    df_gap = pd.DataFrame(results_gap, columns=columns_gap)
    dirname_gap = get_filename(
        ("edcGap" + bz_point, theta_deviation, n_shells, spreadE, *args_fn),
        dirname=get_home_dn(machine, "bilayer") + "Data/",
        float_precision=4,
    ) + "_" + list_fn + "/"
    if not Path(dirname_gap).is_dir():
        os.system(f"mkdir -p {dirname_gap}")
    output_file_gap = dirname_gap + f"chunk_{ind}_{n_chunks}.h5"
    df_gap.to_hdf(output_file_gap, key="results", mode="w", format="table", complevel=5, complib="blosc")

print(f"Finished {bz_point} chunk {ind}/{n_chunks}")
