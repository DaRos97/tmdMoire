"""Export moire bands and a Gamma EDC profile for standalone plotting.

Computes the supercell band structure along a G->K line through Gamma for
V_G = 0, a chosen middle value (10.5 meV by default), and 21 meV. Also
exports the 21 meV Gamma EDC profile and its four-Lorentzian fit. Output goes
to scripts/plotsPaper/data/. The profile uses the same Gamma-only potential
configuration as the band plot (V_K = 0).

Usage:
    source .venv/bin/activate
    python scripts/export_moire_bands.py
    python scripts/export_moire_bands.py --sample S3 --Vg 10.5 --w1p -1.2 --w1d 0.455 --phiG 175
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights
from tmdmoire.bilayer.edc_analyzer import find_peak_seeds_gamma
from tmdmoire.constants import (
    EDC_G_POSITIONS,
    EDC_G_SEED_BOUNDARY,
    ENERGY_OFFSETS,
    TWIST_ANGLES,
)

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("scripts") / "plotsPaper" / "data"

K_RANGE = 0.4
N_K_PTS = 301
N_SHELLS = 2
PROFILE_VG_MEV = 21.0
EDC_SPREAD_E = 0.03
EDC_ENERGY_STEP = 0.005

BAND_LO = 26
BAND_HI = 28


def _lorentz_peak(x, amplitude, center, gamma):
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _four_lorentzians(
    x,
    a1, c1, g1,
    a2, c2, g2,
    a3, c3, g3,
    a4, c4, g4,
):
    return (
        _lorentz_peak(x, a1, c1, g1)
        + _lorentz_peak(x, a2, c2, g2)
        + _lorentz_peak(x, a3, c3, g3)
        + _lorentz_peak(x, a4, c4, g4)
    )


def _compute_gamma_edc_profile(evals_gamma, evecs_gamma, n_cells, sample):
    """Build and fit the broadened Gamma EDC from one diagonalization."""
    weights_all = np.abs(evecs_gamma) ** 2
    layer_weights = (
        np.sum(weights_all[:22, :], axis=0)
        + np.sum(weights_all[22 * n_cells:22 * (n_cells + 1), :], axis=0)
    )

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_low = index_lvb - 2 * n_cells + 1
    energy_values = evals_gamma[index_low:index_tvb + 1]
    state_weights = layer_weights[index_low:index_tvb + 1]

    energy_min = energy_values[0]
    energy_max = energy_values[-1]
    energy_range = energy_max - energy_min
    energy_min -= energy_range / 2
    energy_max += energy_range / 2
    n_energy = int((energy_max - energy_min) / EDC_ENERGY_STEP)
    energy_list = np.linspace(energy_min, energy_max, n_energy)
    weight_list = np.zeros(len(energy_list))

    for energy, weight in zip(energy_values, state_weights):
        weight_list += EDC_SPREAD_E / np.pi * weight / (
            (energy_list - energy) ** 2 + EDC_SPREAD_E ** 2
        )

    peak_states = find_peak_seeds_gamma(
        weight_list,
        energy_list,
        energy_values,
        state_weights,
        boundary_ev=EDC_G_SEED_BOUNDARY.get(sample, -1.5),
    )

    import lmfit

    model = lmfit.Model(_four_lorentzians)
    params_fit = model.make_params(
        a1=peak_states[0][1], c1=peak_states[0][0], g1=EDC_SPREAD_E,
        a2=peak_states[1][1], c2=peak_states[1][0], g2=EDC_SPREAD_E,
        a3=peak_states[2][1], c3=peak_states[2][0], g3=EDC_SPREAD_E,
        a4=peak_states[3][1], c4=peak_states[3][0], g4=EDC_SPREAD_E,
    )
    for name in ("a1", "a2", "a3", "a4"):
        params_fit[name].set(min=0)
    for name in ("g1", "g2", "g3", "g4"):
        params_fit[name].set(min=1e-4, max=0.2)
    for i_peak, name in enumerate(("c1", "c2", "c3", "c4")):
        seed = peak_states[i_peak][0]
        params_fit[name].set(min=seed - 0.05, max=seed + 0.05)

    try:
        result = model.fit(weight_list, params_fit, x=energy_list)
        if not result.success:
            raise RuntimeError("lmfit did not converge")

        fits = [
            (
                result.best_values[f"a{i_peak}"],
                result.best_values[f"c{i_peak}"],
                result.best_values[f"g{i_peak}"],
            )
            for i_peak in range(1, 5)
        ]
        fits.sort(key=lambda fit: fit[1], reverse=True)
        fit_centers = np.array([fit[1] for fit in fits])
        fit_curve = _four_lorentzians(
            energy_list,
            fits[0][0], fits[0][1], fits[0][2],
            fits[1][0], fits[1][1], fits[1][2],
            fits[2][0], fits[2][1], fits[2][2],
            fits[3][0], fits[3][1], fits[3][2],
        )
        fit_redchi = float(result.redchi)
    except Exception as exc:
        print(f"4-Lorentzian EDC fit failed: {exc}")
        fit_curve = np.full_like(energy_list, np.nan)
        fit_centers = np.full(4, np.nan)
        fit_redchi = np.nan

    return {
        "energy_list": energy_list,
        "weight_list": weight_list,
        "fit_4L_curve": fit_curve,
        "fit_4L_centers": fit_centers,
        "fit_4L_redchi": fit_redchi,
    }


def parse_args():
    sample = "S11"
    w1p = -1.220
    w1d = 0.460
    w2p = -0.1694
    w2d = 0.0215
    phiG_deg = 175.0
    vg_middle_mev = 10.5

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
        elif args[i] == "--Vg" and i + 1 < len(args):
            vg_middle_mev = float(args[i + 1])
            i += 2
        else:
            i += 1

    return sample, w1p, w1d, w2p, w2d, phiG_deg, vg_middle_mev


def main():
    sample, w1p, w1d, w2p, w2d, phiG_deg, vg_middle_mev = parse_args()

    interlayer = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
    phiG_rad = phiG_deg * np.pi / 180.0
    vg_values_mev = np.array([0.0, vg_middle_mev, PROFILE_VG_MEV])
    if not 0.0 < vg_middle_mev < PROFILE_VG_MEV:
        raise ValueError("The middle Vg value must be between 0 and 21 meV.")
    vg_values = vg_values_mev / 1000.0
    vg_labels = np.array([f"{vg_mev:g} meV" for vg_mev in vg_values_mev], dtype=object)

    print("Loading monolayer parameters from Inputs/plot_bilayer/")
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    theta = TWIST_ANGLES[sample]
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(N_SHELLS)
    band_start = BAND_LO * n_cells
    band_end = BAND_HI * n_cells
    print(f"n_shells={N_SHELLS}, n_cells={n_cells}, bands {band_start}:{band_end}")

    k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_PTS)
    k_list = np.column_stack([k_vals, np.zeros(N_K_PTS)])

    energy_offset = ENERGY_OFFSETS.get(sample, 0.0)

    export = {
        "k_vals": k_vals,
        "Vg_values_meV": vg_values_mev,
        "Vg_labels": vg_labels,
        "n_shells": N_SHELLS,
        "n_cells": n_cells,
        "n_kpts": N_K_PTS,
        "k_range": K_RANGE,
        "sample": sample,
        "phiG_deg": phiG_deg,
        "Vk_meV": 0.0,
        "phiK_deg": 0.0,
        "interlayer_w1p": w1p,
        "interlayer_w1d": w1d,
        "interlayer_w2p": w2p,
        "interlayer_w2d": w2d,
    }

    gamma_index = int(np.argmin(np.abs(k_vals)))
    for i_probe, (vg, vg_mev) in enumerate(zip(vg_values, vg_values_mev)):
        pars_V = (vg, 0.0, phiG_rad, 0.0)
        print(f"Diagonalizing V_G = {vg_mev:.1f} meV ({N_K_PTS} k-points, {n_cells} cells) ...", flush=True)

        evals_full, evecs_full = moire_ham.diagonalize(
            k_list, N_SHELLS, interlayer, pars_V
        )

        evals = evals_full[:, band_start:band_end] + energy_offset
        evecs = evecs_full[:, :, band_start:band_end]

        weights = compute_weights(evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1)

        export[f"evals_{i_probe}"] = evals
        export[f"weights_{i_probe}"] = weights

        if np.isclose(vg_mev, PROFILE_VG_MEV):
            edc_profile = _compute_gamma_edc_profile(
                evals_full[gamma_index] + energy_offset,
                evecs_full[gamma_index],
                n_cells,
                sample,
            )
            export.update(edc_profile)
            export.update({
                "run_id": f"moire_{sample}_Vg{PROFILE_VG_MEV:g}meV",
                "exp_positions_ev": EDC_G_POSITIONS[sample],
                "selected_Vg_meV": PROFILE_VG_MEV,
                "selected_phiG_deg": phiG_deg,
                "selected_w1p_ev": w1p,
                "selected_w1d_ev": w1d,
                "selected_w2p_ev": w2p,
                "selected_w2d_ev": w2d,
            })
            print(
                "  EDC fit centers: "
                f"{[f'{center:.4f}' for center in edc_profile['fit_4L_centers']]} eV, "
                f"redchi={edc_profile['fit_4L_redchi']:.6f}",
                flush=True,
            )

        print(f"  Done.", flush=True)
        del evals_full, evecs_full, evals, evecs, weights

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vg_middle_str = f"{vg_middle_mev:g}"
    out_fn = OUTPUT_DIR / (
        f"moire_bands_{sample}_k{N_K_PTS}_n{N_SHELLS}_Vg0_{vg_middle_str}_21.npz"
    )
    np.savez(out_fn, **export)
    print(f"Exported: {out_fn}")


if __name__ == "__main__":
    main()
