"""Export EDC TVB–side band distance vs V_G data for standalone plotting.

Computes EDC intensity profiles at Gamma for V_G = 1–20 meV (20 points),
fits 4 Lorentzians, and exports the TVB–side band distances together with
all metadata. Output goes to scripts/plotsPaper/data/.

Usage:
    source ../PyEnv/bin/activate
    python scripts/export_edc_vs_V.py
    python scripts/export_edc_vs_V.py --sample S3 --w1p -1.2 --w1d 0.455 --phiG 175
"""
import sys
from pathlib import Path

import numpy as np
import lmfit
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.constants import (
    TWIST_ANGLES, ENERGY_OFFSETS, EDC_G_POSITIONS, EDC_G_SEED_BOUNDARY,
)

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("scripts") / "plotsPaper" / "data"

N_SHELLS = 1
SPREAD_E = 0.03
EDC_SHIFT_MEV = 0.0


def parse_args():
    sample = "S11"
    w1p = -1.220
    w1d = 0.460
    w2p = -0.1694
    w2d = 0.0215
    phiG_deg = 175.0
    vg_min_meV = 1
    vg_max_meV = 20
    n_vg = 20

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--sample" and i + 1 < len(args):
            sample = args[i + 1]
            i += 2
        elif args[i] == "--w1p" and i + 1 < len(args):
            w1p = float(args[i + 1])
            i += 2
        elif args[i] == "--w1d" and i + 1 < len(args):
            w1d = float(args[i + 1])
            i += 2
        elif args[i] == "--w2p" and i + 1 < len(args):
            w2p = float(args[i + 1])
            i += 2
        elif args[i] == "--w2d" and i + 1 < len(args):
            w2d = float(args[i + 1])
            i += 2
        elif args[i] == "--phiG" and i + 1 < len(args):
            phiG_deg = float(args[i + 1])
            i += 2
        elif args[i] == "--vg-min" and i + 1 < len(args):
            vg_min_meV = float(args[i + 1])
            i += 2
        elif args[i] == "--vg-max" and i + 1 < len(args):
            vg_max_meV = float(args[i + 1])
            i += 2
        elif args[i] == "--n-vg" and i + 1 < len(args):
            n_vg = int(args[i + 1])
            i += 2
        else:
            i += 1

    return sample, w1p, w1d, w2p, w2d, phiG_deg, vg_min_meV, vg_max_meV, n_vg


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _four_lorentzian(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
    return (
        _lorentzian(x, a1, c1, g1)
        + _lorentzian(x, a2, c2, g2)
        + _lorentzian(x, a3, c3, g3)
        + _lorentzian(x, a4, c4, g4)
    )


def compute_edc_distance(vg_ev, moire_ham, interlayer, phiG_rad, sample, boundary_ev):
    n_cells = MoireGeometry.n_cells(N_SHELLS)
    k_list = np.array([np.zeros(2)])

    pars_V = (vg_ev, 0.0, phiG_rad, 0.0)

    evals_raw, evecs_raw = moire_ham.diagonalize(
        k_list, N_SHELLS, interlayer, pars_V
    )
    evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_raw[0]

    ab = np.absolute(evecs_raw) ** 2
    weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells : 22 * (1 + n_cells), :], axis=0)

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_l = index_lvb - 2 * n_cells + 1

    full_energy_values = evals_raw[index_l : index_tvb + 1]
    full_weight_values = weights[index_l : index_tvb + 1]

    min_e = full_energy_values[0]
    max_e = full_energy_values[-1]
    delta = max_e - min_e
    min_e -= delta / 2
    max_e += delta / 2
    n_e = int((max_e - min_e) / 0.005)
    energy_list = np.linspace(min_e, max_e, n_e)
    weight_list = np.zeros(len(energy_list))

    for i in range(len(full_energy_values)):
        weight_list += (
            SPREAD_E / np.pi * full_weight_values[i] / ((energy_list - full_energy_values[i]) ** 2 + SPREAD_E ** 2)
        )

    peaks_idx, _ = find_peaks(weight_list, height=weight_list.max() * 0.005, distance=int(0.01 / 0.005))
    peaks_found = list(zip(energy_list[peaks_idx], weight_list[peaks_idx]))

    tvb_region = [(e, h) for e, h in peaks_found if e > boundary_ev]
    tvb_main = max(tvb_region, key=lambda x: x[1]) if tvb_region else (boundary_ev + 0.1, 10.0)

    lvb_region = [(e, h) for e, h in peaks_found if e < boundary_ev]
    lvb_main = max(lvb_region, key=lambda x: x[1]) if lvb_region else (boundary_ev - 0.1, 10.0)

    eigen_by_energy = sorted(zip(full_energy_values, full_weight_values), key=lambda x: x[0], reverse=True)

    side_candidates = [e for e in eigen_by_energy if e[0] < tvb_main[0] - 0.01 and e[0] > boundary_ev]
    tvb_side = max(side_candidates, key=lambda x: x[1]) if side_candidates else (tvb_main[0] - 0.05, tvb_main[1] * 0.3)

    lvb_side_candidates = [e for e in eigen_by_energy if e[0] < lvb_main[0] - 0.01]
    lvb_side = max(lvb_side_candidates, key=lambda x: x[1]) if lvb_side_candidates else (lvb_main[0] - 0.05, lvb_main[1] * 0.3)

    peak_states = sorted([tvb_main, tvb_side, lvb_main, lvb_side], key=lambda x: x[0], reverse=True)

    model = lmfit.Model(_four_lorentzian)
    params_fit = model.make_params(
        a1=peak_states[0][1], c1=peak_states[0][0], g1=SPREAD_E,
        a2=peak_states[1][1], c2=peak_states[1][0], g2=SPREAD_E,
        a3=peak_states[2][1], c3=peak_states[2][0], g3=SPREAD_E,
        a4=peak_states[3][1], c4=peak_states[3][0], g4=SPREAD_E,
    )
    for p in ["a1", "a2", "a3", "a4"]:
        params_fit[p].set(min=0)
    for p in ["g1", "g2", "g3", "g4"]:
        params_fit[p].set(min=1e-4, max=0.2)
    for i_p, p in enumerate(["c1", "c2", "c3", "c4"]):
        seed = peak_states[i_p][0]
        params_fit[p].set(min=seed - 0.05, max=seed + 0.05)

    result = model.fit(weight_list, params_fit, x=energy_list)

    c1 = result.best_values["c1"]
    c2 = result.best_values["c2"]
    distance_ev = abs(c1 - c2)
    return distance_ev * 1000.0 + EDC_SHIFT_MEV


def main():
    sample, w1p, w1d, w2p, w2d, phiG_deg, vg_min_meV, vg_max_meV, n_vg = parse_args()

    interlayer = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
    phiG_rad = phiG_deg * np.pi / 180.0
    boundary_ev = EDC_G_SEED_BOUNDARY.get(sample, -1.5)

    print("Loading monolayer parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    theta = TWIST_ANGLES[sample]
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    vg_vals_meV = np.linspace(vg_min_meV, vg_max_meV, n_vg)
    vg_vals_ev = vg_vals_meV / 1000.0

    distances = np.zeros(n_vg)

    for i, vg_meV in enumerate(vg_vals_meV):
        print(f"V_G = {vg_meV:.1f} meV ...", flush=True)
        distances[i] = compute_edc_distance(vg_vals_ev[i], moire_ham, interlayer, phiG_rad, sample, boundary_ev)

    exp_positions = EDC_G_POSITIONS[sample]
    arpes_distance = abs(exp_positions[0] - exp_positions[1]) * 1000.0

    print(f"\nARPES TVB–side band distance: {arpes_distance:.2f} meV")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fn = OUTPUT_DIR / f"edc_vs_V_{sample}_n{n_vg}_Vg{vg_min_meV:.0f}-{vg_max_meV:.0f}.npz"
    np.savez(
        out_fn,
        Vg_vals_meV=vg_vals_meV,
        distances_meV=distances,
        arpes_distance_meV=arpes_distance,
        interlayer_w1p=w1p,
        interlayer_w1d=w1d,
        interlayer_w2p=w2p,
        interlayer_w2d=w2d,
        phiG_deg=phiG_deg,
        n_shells=N_SHELLS,
    )
    print(f"Exported: {out_fn}")


if __name__ == "__main__":
    main()
