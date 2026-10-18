"""Compare S11 moire bands and Gamma EDCs near the phase-grid boundary.

Computes the bands around Gamma and the Gamma EDC at Vg=13 meV for
phiG=165, 160, and 155 degrees. Uses the best-fit interlayer couplings from
Inputs/plot_bilayer/interlayer_G.json and saves comparison figures in
scripts/dev/figures/.

Run from the repository root with the project environment:
    python scripts/dev/phase_boundary_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lmfit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tmdmoire.bilayer.edc_analyzer import find_peak_seeds_gamma
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights
from tmdmoire.constants import (
    EDC_G_POSITIONS,
    EDC_G_SEED_BOUNDARY,
    ENERGY_OFFSETS,
    TWIST_ANGLES,
)
from tmdmoire.material import TMDMaterial


INPUT_DIR = REPO_ROOT / "Inputs" / "plot_bilayer"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

SAMPLE = "S11"
V_G_MEV = 13.0
V_K_MEV = 7.7
PHI_K_DEG = 106.0
PHI_G_DEG_VALUES = (165.0, 160.0, 155.0)

N_SHELLS = 2
K_RANGE = 0.4
N_K_POINTS = 301
K_BATCH_SIZE = 8
BAND_LO = 26
BAND_HI = 28

EDC_SPREAD_E = 0.03
EDC_ENERGY_STEP = 0.005


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)


def _four_lorentzians(
    x,
    a1, c1, g1,
    a2, c2, g2,
    a3, c3, g3,
    a4, c4, g4,
):
    return (
        _lorentzian(x, a1, c1, g1)
        + _lorentzian(x, a2, c2, g2)
        + _lorentzian(x, a3, c3, g3)
        + _lorentzian(x, a4, c4, g4)
    )


def _compute_gamma_edc_profile(evals_gamma, evecs_gamma, n_cells, sample, phi_g_deg):
    """Build and fit the broadened Gamma EDC from one diagonalization."""
    evals_shifted = evals_gamma + ENERGY_OFFSETS.get(sample, 0.0)
    weights_all = np.abs(evecs_gamma) ** 2
    central_cell_weights = (
        np.sum(weights_all[:22, :], axis=0)
        + np.sum(weights_all[22 * n_cells:22 * (n_cells + 1), :], axis=0)
    )

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_low = index_lvb - 2 * n_cells + 1
    energy_values = evals_shifted[index_low:index_tvb + 1]
    state_weights = central_cell_weights[index_low:index_tvb + 1]

    energy_min = energy_values[0]
    energy_max = energy_values[-1]
    energy_range = energy_max - energy_min
    energy_min -= energy_range / 2
    energy_max += energy_range / 2
    n_energy = int((energy_max - energy_min) / EDC_ENERGY_STEP)
    energy_list = np.linspace(energy_min, energy_max, n_energy)
    weight_list = np.zeros_like(energy_list)

    for energy, weight in zip(energy_values, state_weights):
        weight_list += EDC_SPREAD_E / np.pi * weight / (
            (energy_list - energy) ** 2 + EDC_SPREAD_E**2
        )

    peak_states = find_peak_seeds_gamma(
        weight_list,
        energy_list,
        energy_values,
        state_weights,
        boundary_ev=EDC_G_SEED_BOUNDARY.get(sample, -1.5),
    )

    fit_curve = np.full_like(energy_list, np.nan)
    fit_centers = np.full(4, np.nan)
    fit_redchi = np.nan
    fit_success = False

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
        fit_success = True
    except Exception as exc:
        print(f"  4-Lorentzian fit failed at phiG={phi_g_deg:.0f} deg: {exc}")

    return {
        "energy_list": energy_list,
        "weight_list": weight_list,
        "fit_curve": fit_curve,
        "fit_centers": fit_centers,
        "fit_redchi": fit_redchi,
        "fit_success": fit_success,
    }


def _load_model_inputs():
    with (INPUT_DIR / "interlayer_G.json").open(encoding="utf-8") as file:
        interlayer_config = json.load(file)

    interlayer_params = {
        name: float(interlayer_config[name])
        for name in ("w1p", "w1d", "w2p", "w2d")
    }
    wse2 = TMDMaterial("WSe2", params=np.load(INPUT_DIR / "tb_WSe2.npy"))
    ws2 = TMDMaterial("WS2", params=np.load(INPUT_DIR / "tb_WS2.npy"))
    return wse2, ws2, interlayer_params


def _diagonalize_phase(moire_ham, interlayer_params, k_points, gamma_index,
                       n_cells, phi_g_deg, energy_offset):
    """Compute selected bands and retain only a small batch of eigenvectors."""
    band_start = BAND_LO * n_cells
    band_end = BAND_HI * n_cells
    n_bands = band_end - band_start
    evals = np.empty((N_K_POINTS, n_bands))
    weights = np.empty((N_K_POINTS, n_bands))
    edc_profile = None

    phi_g = np.deg2rad(phi_g_deg)
    phi_k = np.deg2rad(PHI_K_DEG)
    pars_v = (V_G_MEV / 1000.0, V_K_MEV / 1000.0, phi_g, phi_k)

    for batch_number, start in enumerate(range(0, N_K_POINTS, K_BATCH_SIZE), start=1):
        end = min(start + K_BATCH_SIZE, N_K_POINTS)
        evals_full, evecs_full = moire_ham.diagonalize(
            k_points[start:end], N_SHELLS, interlayer_params, pars_v
        )

        evals[start:end] = evals_full[:, band_start:band_end] + energy_offset
        band_evecs = evecs_full[:, :, band_start:band_end]
        weights[start:end] = compute_weights(
            band_evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1
        )

        if start <= gamma_index < end:
            local_gamma_index = gamma_index - start
            edc_profile = _compute_gamma_edc_profile(
                evals_full[local_gamma_index],
                evecs_full[local_gamma_index],
                n_cells,
                SAMPLE,
                phi_g_deg,
            )

        if batch_number == 1 or batch_number % 5 == 0 or end == N_K_POINTS:
            print(
                f"  phiG={phi_g_deg:.0f} deg: k-points {end}/{N_K_POINTS}",
                flush=True,
            )

        del band_evecs, evals_full, evecs_full

    if edc_profile is None:
        raise RuntimeError("The k-point path did not include Gamma.")

    return evals, weights, edc_profile


def _plot_bands(k_vals, phase_results, interlayer_params, output_dir):
    all_evals = np.concatenate([result["evals"].ravel() for result in phase_results])
    energy_min = np.nanmin(all_evals)
    energy_max = np.nanmax(all_evals)
    energy_pad = 0.1 * (energy_max - energy_min)

    fig, axes = plt.subplots(
        1,
        len(phase_results),
        figsize=(6.5 * len(phase_results), 7),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, result in zip(axes, phase_results):
        evals = result["evals"]
        weights = result["weights"]
        phi_g_deg = result["phi_g_deg"]

        for i_band in range(evals.shape[1]):
            ax.plot(k_vals, evals[:, i_band], color="lightgray", lw=0.5,
                    alpha=0.5, zorder=1)

        max_weight = weights.max()
        if max_weight > 0:
            dot_sizes = 80 * weights / max_weight
            for i_band in range(evals.shape[1]):
                mask = dot_sizes[:, i_band] > 0
                if mask.any():
                    ax.scatter(
                        k_vals[mask], evals[mask, i_band],
                        s=dot_sizes[mask, i_band], c="#1f77b4",
                        alpha=1.0, zorder=2, edgecolors="none", linewidths=0,
                    )

        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        ax.set_title(rf"$\phi_G = {phi_g_deg:.0f}^\circ$", fontsize=14,
                     fontweight="bold")
        ax.set_xlim(-K_RANGE, K_RANGE)
        ax.set_ylim(energy_min - energy_pad, energy_max + energy_pad)

    axes[0].set_ylabel("Energy (eV)", fontsize=12)
    coupling_text = ", ".join(
        f"{name}={value:.4f}" for name, value in interlayer_params.items()
    )
    fig.suptitle(
        f"Moiré bands around $\\Gamma$ for S11 ($\\theta=2.8^\\circ$)\n"
        f"$V_G={V_G_MEV:g}$ meV, $V_K={V_K_MEV:g}$ meV, "
        f"$\\phi_K={PHI_K_DEG:g}^\\circ$, $n_{{\\rm shells}}={N_SHELLS}$\n"
        f"{coupling_text}",
        fontsize=13,
        fontweight="bold",
    )

    output_path = output_dir / "phase_boundary_bands_S11_Vg13meV.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _plot_edcs(phase_results, output_dir):
    exp_positions = EDC_G_POSITIONS[SAMPLE]
    energy_min = min(np.min(result["edc"]["energy_list"]) for result in phase_results)
    energy_max = max(np.max(result["edc"]["energy_list"]) for result in phase_results)

    fig, axes = plt.subplots(
        1,
        len(phase_results),
        figsize=(6.5 * len(phase_results), 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for i_phase, (ax, result) in enumerate(zip(axes, phase_results)):
        edc = result["edc"]
        phi_g_deg = result["phi_g_deg"]
        ax.plot(edc["energy_list"], edc["weight_list"], "k-", lw=1.5,
                label="EDC intensity")

        if edc["fit_success"]:
            ax.plot(
                edc["energy_list"],
                edc["fit_curve"],
                "r--",
                lw=2,
                label=rf"4-Lorentzian fit ($\chi_\nu^2$ = {edc['fit_redchi']:.4f})",
            )

        for i_energy, energy in enumerate(exp_positions):
            ax.axvline(
                x=energy,
                color="#2ecc71",
                ls="--",
                lw=1.5,
                alpha=0.8,
                label="ARPES EDC position" if i_phase == 0 and i_energy == 0 else "_nolegend_",
            )

        for i_center, center in enumerate(edc["fit_centers"]):
            if np.isnan(center):
                continue
            ax.axvline(
                x=center,
                color="#e74c3c",
                ls=":",
                lw=1.5,
                alpha=0.8,
                label="4-Lorentzian center" if i_phase == 0 and i_center == 0 else "_nolegend_",
            )

        ax.set_xlabel("Energy (eV)", fontsize=12)
        ax.set_title(rf"$\phi_G = {phi_g_deg:.0f}^\circ$", fontsize=14,
                     fontweight="bold")
        ax.set_xlim(energy_min, energy_max)

    axes[0].set_ylabel("Intensity (a.u.)", fontsize=12)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Gamma EDC profiles for S11: $V_G={V_G_MEV:g}$ meV, "
        f"$V_K={V_K_MEV:g}$ meV, $\\phi_K={PHI_K_DEG:g}^\\circ$\n"
        "Green: experimental positions; red dotted: fitted centers",
        fontsize=13,
        fontweight="bold",
    )

    output_path = output_dir / "phase_boundary_edc_S11_Vg13meV.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wse2, ws2, interlayer_params = _load_model_inputs()
    theta = TWIST_ANGLES[SAMPLE]
    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)
    n_cells = MoireGeometry.n_cells(N_SHELLS)

    k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_POINTS)
    k_points = np.column_stack((k_vals, np.zeros(N_K_POINTS)))
    gamma_index = int(np.argmin(np.abs(k_vals)))
    if not np.isclose(k_vals[gamma_index], 0.0):
        raise RuntimeError("N_K_POINTS must produce an exact Gamma point.")

    energy_offset = ENERGY_OFFSETS.get(SAMPLE, 0.0)
    print(
        f"S11: theta={theta:g} deg, n_cells={n_cells}, "
        f"k-path=[-{K_RANGE:g}, {K_RANGE:g}] A^-1 ({N_K_POINTS} points)"
    )
    print(
        f"Vg={V_G_MEV:g} meV, Vk={V_K_MEV:g} meV, phiK={PHI_K_DEG:g} deg; "
        f"interlayer={interlayer_params}; energy offset={energy_offset:g} eV"
    )

    phase_results = []
    for phi_g_deg in PHI_G_DEG_VALUES:
        print(f"Diagonalizing phiG={phi_g_deg:g} deg ...", flush=True)
        evals, weights, edc = _diagonalize_phase(
            moire_ham,
            interlayer_params,
            k_points,
            gamma_index,
            n_cells,
            phi_g_deg,
            energy_offset,
        )
        phase_results.append({
            "phi_g_deg": phi_g_deg,
            "evals": evals,
            "weights": weights,
            "edc": edc,
        })
        if edc["fit_success"]:
            centers = ", ".join(f"{center:.4f}" for center in edc["fit_centers"])
            print(
                f"  EDC centers: [{centers}] eV; "
                f"redchi={edc['fit_redchi']:.6g}"
            )

    _plot_bands(k_vals, phase_results, interlayer_params, OUTPUT_DIR)
    _plot_edcs(phase_results, OUTPUT_DIR)
    print("Phase-boundary comparison complete.")


if __name__ == "__main__":
    main()
