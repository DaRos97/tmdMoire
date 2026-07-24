"""Compare moire bands and EDC profiles for multiple moire potential phases.

Computes bands along k-path (+-0.4 Ang^-1) through Gamma and EDC intensity
profiles at the Gamma point for phi_G = 150, 160, 170 deg, with V_G = 12 meV.
Fixed interlayer couplings and V_K = 7.7 meV, phi_K = 106 deg.

Outputs:
    Data/compare_phase_bands/
    ├── phase_compare_*.npz       (cached k_vals, evals, weights, EDC data)
    ├── bands_comparison.png       (Figure 1: 3-panel band comparison)
    ├── edc_comparison.png         (Figure 2: EDC overlay comparison)
    └── edc_fits.png               (Figure 3: 3-panel EDC + 4-Lorentzian fits)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lmfit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights
from tmdmoire.bilayer.edc_analyzer import find_peak_seeds_gamma
from tmdmoire.constants import TWIST_ANGLES, ENERGY_OFFSETS, EDC_G_SEED_BOUNDARY

INPUT_DIR = Path("Inputs") / "plot_bilayer"
OUTPUT_DIR = Path("Data") / "compare_phase_bands"

PHI_G_DEG_VALUES = [150, 160, 170]
VG = 0.012
VK = 0.0077
PHI_K_DEG = 106

INTERLAYER = {"w1p": -1.22, "w1d": 0.46, "w2p": -0.1694, "w2d": 0.0215}

N_SHELLS = 2
SAMPLE = "S11"
K_RANGE = 0.4
N_K_PTS = 201
SPREAD_E = 0.03
ENERGY_GRID_STEP = 0.005

BAND_LO = 26
BAND_HI = 28
PLOT_Y_MIN = -1.5
PLOT_Y_MAX = -1.0


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _four_lorentzian(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
    return (_lorentzian(x, a1, c1, g1) + _lorentzian(x, a2, c2, g2)
            + _lorentzian(x, a3, c3, g3) + _lorentzian(x, a4, c4, g4))


def _cache_filename():
    parts = [
        "phase_compare",
        f"k{N_K_PTS}",
        f"n{N_SHELLS}",
        f"w1p{INTERLAYER['w1p']:.4f}",
        f"w1d{INTERLAYER['w1d']:.4f}",
        f"w2p{INTERLAYER['w2p']:.4f}",
        f"w2d{INTERLAYER['w2d']:.4f}",
    ]
    phi_str = "_".join(f"phi{phi}" for phi in PHI_G_DEG_VALUES)
    parts.append(phi_str)
    parts.append(f"Vg{int(VG * 1000)}")
    parts.append(f"Vk{int(VK * 10000)}")
    parts.append(f"phiK{PHI_K_DEG}")
    return OUTPUT_DIR / ("_".join(parts) + ".npz")


def compute_edc_from_full(evals_full_at_k0, evecs_full_at_k0, n_cells,
                          sample, seed_boundary):
    """Compute EDC intensity profile at Gamma from full eigenvalues/vectors."""
    evals_raw = evals_full_at_k0 + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_full_at_k0

    ab = np.absolute(evecs_raw) ** 2
    weights = (np.sum(ab[:22, :], axis=0)
               + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0))

    index_tvb = 28 * n_cells - 1
    index_lvb = 26 * n_cells - 1
    index_l = index_lvb - 2 * n_cells + 1

    full_energy_values = evals_raw[index_l:index_tvb + 1]
    full_weight_values = weights[index_l:index_tvb + 1]

    min_e = full_energy_values[0]
    max_e = full_energy_values[-1]
    delta = max_e - min_e
    min_e -= delta / 2
    max_e += delta / 2
    n_e = int((max_e - min_e) / ENERGY_GRID_STEP)
    energy_list = np.linspace(min_e, max_e, n_e)
    weight_list = np.zeros(len(energy_list))

    for i in range(len(full_energy_values)):
        weight_list += SPREAD_E / np.pi * full_weight_values[i] / (
            (energy_list - full_energy_values[i]) ** 2 + SPREAD_E ** 2
        )

    peak_states = find_peak_seeds_gamma(
        weight_list, energy_list, full_energy_values, full_weight_values,
        boundary_ev=seed_boundary
    )

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
    for i, p in enumerate(["c1", "c2", "c3", "c4"]):
        seed = peak_states[i][0]
        params_fit[p].set(min=seed - 0.05, max=seed + 0.05)

    result = model.fit(weight_list, params_fit, x=energy_list)

    best_values = {k: v for k, v in result.best_values.items()}

    return {
        "energy_list": energy_list,
        "weight_list": weight_list,
        "full_energy_values": full_energy_values,
        "full_weight_values": full_weight_values,
        "peak_states": peak_states,
        "result": result,
        "best_values": best_values,
        "redchi": result.redchi,
        "success": result.success,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_fn = _cache_filename()

    if cache_fn.exists():
        print(f"Loading cached diagonalization from {cache_fn.name}")
        data = np.load(cache_fn, allow_pickle=True)
        k_vals = data["k_vals"]
        all_evals = [data[f"evals_{phi}"] for phi in PHI_G_DEG_VALUES]
        all_weights = [data[f"weights_{phi}"] for phi in PHI_G_DEG_VALUES]
        all_edc = []
        param_names_lm = ["a1", "c1", "g1", "a2", "c2", "g2",
                          "a3", "c3", "g3", "a4", "c4", "g4"]
        for phi in PHI_G_DEG_VALUES:
            edc = {}
            edc["energy_list"] = data[f"edc_energy_{phi}"]
            edc["weight_list"] = data[f"edc_intensity_{phi}"]
            redchi_key = f"edc_redchi_{phi}"
            if redchi_key in data:
                edc["success"] = True
                edc["best_values"] = {k: float(data[f"edc_param_{k}_{phi}"])
                                      for k in param_names_lm}
                edc["redchi"] = float(data[redchi_key])
            else:
                edc["success"] = False
            all_edc.append(edc)
    else:
        print("Loading monolayer parameters from Inputs/plot_bilayer/")
        tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
        tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")

        wse2 = TMDMaterial("WSe2", params=tb_wse2)
        ws2 = TMDMaterial("WS2", params=tb_ws2)

        theta = TWIST_ANGLES[SAMPLE]
        geometry = MoireGeometry(theta)
        moire_ham = MoireHamiltonian(wse2, ws2, geometry)

        n_cells = MoireGeometry.n_cells(N_SHELLS)
        band_start = BAND_LO * n_cells
        band_end = BAND_HI * n_cells
        energy_offset = ENERGY_OFFSETS.get(SAMPLE, 0.0)
        phi_k = PHI_K_DEG / 180 * np.pi
        seed_boundary = EDC_G_SEED_BOUNDARY.get(SAMPLE, -1.5)

        print(f"n_cells = {n_cells}, bands {band_start}:{band_end}")
        print(f"Interlayer: w1p={INTERLAYER['w1p']}, w1d={INTERLAYER['w1d']}, "
              f"w2p={INTERLAYER['w2p']}, w2d={INTERLAYER['w2d']}")
        print(f"Vg={VG}, Vk={VK:.4f}, phiK={PHI_K_DEG} deg")
        print(f"S11 energy offset = {energy_offset} eV")
        print()

        k_vals = np.linspace(-K_RANGE, K_RANGE, N_K_PTS)
        k_list = np.column_stack([k_vals, np.zeros(N_K_PTS)])
        k0_idx = N_K_PTS // 2

        all_evals = []
        all_weights = []
        all_edc = []

        for phi_g_deg in PHI_G_DEG_VALUES:
            phi_g = phi_g_deg / 180 * np.pi
            pars_V = (VG, VK, phi_g, phi_k)

            label = f"phiG={phi_g_deg}deg"
            print(f"Diagonalizing k-path for {label} ...", flush=True)

            evals_full, evecs_full = moire_ham.diagonalize(
                k_list, N_SHELLS, INTERLAYER, pars_V
            )

            evals = evals_full[:, band_start:band_end] + energy_offset
            evecs = evecs_full[:, :, band_start:band_end]
            weights = compute_weights(evecs, n_cells, pow_factor=2.0, shade_factor_ws2=0.1)

            all_evals.append(evals)
            all_weights.append(weights)

            print(f"  Computing EDC at k=0 for {label} ...", flush=True)
            edc = compute_edc_from_full(
                evals_full[k0_idx], evecs_full[k0_idx], n_cells,
                SAMPLE, seed_boundary
            )
            all_edc.append(edc)

            if edc["success"]:
                res = edc["result"]
                print(f"  EDC fit: c1={res.best_values['c1']:.4f}, "
                      f"c2={res.best_values['c2']:.4f}, "
                      f"c3={res.best_values['c3']:.4f}, "
                      f"c4={res.best_values['c4']:.4f}, "
                      f"redchi={res.redchi:.6f}")
            else:
                print(f"  EDC fit FAILED")
            print()

        print(f"Saving cache to {cache_fn.name}")
        save_dict = {"k_vals": k_vals}
        for phi_g_deg, evals, weights in zip(PHI_G_DEG_VALUES, all_evals, all_weights):
            save_dict[f"evals_{phi_g_deg}"] = evals
            save_dict[f"weights_{phi_g_deg}"] = weights
        for phi_g_deg, edc in zip(PHI_G_DEG_VALUES, all_edc):
            save_dict[f"edc_energy_{phi_g_deg}"] = edc["energy_list"]
            save_dict[f"edc_intensity_{phi_g_deg}"] = edc["weight_list"]
            if edc["success"]:
                res = edc["result"]
                for key, val in res.best_values.items():
                    save_dict[f"edc_param_{key}_{phi_g_deg}"] = val
                save_dict[f"edc_redchi_{phi_g_deg}"] = res.redchi
        np.savez(cache_fn, **save_dict)

    # ─── Figure 1: Band comparison ────────────────────────────────────────

    n_phases = len(PHI_G_DEG_VALUES)
    fig, axes = plt.subplots(1, n_phases, figsize=(8 * n_phases, 7), sharey=True,
                             constrained_layout=True)

    for ax, evals, weights, phi_g_deg in zip(
            axes, all_evals, all_weights, PHI_G_DEG_VALUES):
        for ib in range(evals.shape[1]):
            ax.plot(k_vals, evals[:, ib], color="lightgray", lw=0.5,
                    alpha=0.5, zorder=1)

        w_max = weights.max()
        if w_max > 0:
            w_norm = weights / w_max
            dot_sizes = 80 * w_norm
            for ib in range(evals.shape[1]):
                mask = dot_sizes[:, ib] > 0
                if mask.any():
                    ax.scatter(
                        k_vals[mask], evals[mask, ib],
                        s=dot_sizes[mask, ib],
                        c="#1f77b4", alpha=1.0, zorder=2,
                        edgecolors="none", linewidths=0,
                    )

        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        ax.set_title(rf"$\phi_G = {phi_g_deg}^\circ$", fontsize=14,
                     fontweight="bold")
        ax.set_xlim(-K_RANGE, K_RANGE)
        ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    axes[0].set_ylabel("Energy (eV)", fontsize=12)

    interlayer_str = ", ".join(f"{k}={v}" for k, v in INTERLAYER.items())
    fig.suptitle(
        f"Moire bands around $\\Gamma$ (3 top valence bands)\n"
        f"$V_G = 12$ meV, $V_K = 7.7$ meV, $\\phi_K = {PHI_K_DEG}^\\circ$, "
        f"$n_{{\\rm{{shells}}}}$ = {N_SHELLS}\n"
        f"{interlayer_str}",
        fontsize=12, fontweight="bold", y=1.06,
    )

    fig.savefig(OUTPUT_DIR / "bands_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: bands_comparison.png")

    # ─── Figure 2: EDC overlay comparison ─────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for edc, phi_g_deg, color in zip(all_edc, PHI_G_DEG_VALUES, colors):
        ax.plot(edc["energy_list"], edc["weight_list"], color=color,
                lw=1.5, label=rf"$\phi_G = {phi_g_deg}^\circ$")

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("EDC intensity (a.u.)", fontsize=12)
    ax.set_title(
        f"EDC at $\\Gamma$: $V_G = 12$ meV, $V_K = 7.7$ meV, "
        f"$\\phi_K = {PHI_K_DEG}^\\circ$",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.invert_xaxis()

    fig.savefig(OUTPUT_DIR / "edc_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: edc_comparison.png")

    # ─── Figure 3: EDC fits with individual peaks ─────────────────────────

    fit_colors = ["C0", "C1", "C2", "C3"]
    fit_labels_tmpl = ["TVB main", "TVB side", "LVB main", "LVB side"]

    fig, axes = plt.subplots(1, n_phases, figsize=(8 * n_phases, 7), sharey=True,
                             constrained_layout=True)

    for ax, edc, phi_g_deg in zip(axes, all_edc, PHI_G_DEG_VALUES):
        energy_list = edc["energy_list"]
        weight_list = edc["weight_list"]

        ax.plot(energy_list, weight_list, "k-", lw=1.5,
                label="EDC intensity")

        if edc["success"]:
            best_values = edc["best_values"]
            redchi = edc["redchi"]

            fit_curve = _four_lorentzian(energy_list, **best_values)
            ax.plot(energy_list, fit_curve, "r--", lw=2,
                    label="4-Lorentzian fit")

            for j, color, lbl_tmpl in zip(
                    range(4), fit_colors, fit_labels_tmpl):
                c = best_values[f"c{j + 1}"]
                a = best_values[f"a{j + 1}"]
                g = best_values[f"g{j + 1}"]
                pk = _lorentzian(energy_list, a, c, g)
                ax.plot(energy_list, pk, color=color, ls="-.", lw=1.2,
                        alpha=0.7,
                        label=f"{lbl_tmpl} = {c:.3f} eV")

        ax.set_xlabel("Energy (eV)", fontsize=12)
        ax.set_title(rf"$\phi_G = {phi_g_deg}^\circ$", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.invert_xaxis()

    axes[0].set_ylabel("EDC intensity (a.u.)", fontsize=12)
    fig.suptitle(
        f"EDC at $\\Gamma$: 4-Lorentzian decomposition\n"
        f"$V_G = 12$ meV, $V_K = 7.7$ meV, $\\phi_K = {PHI_K_DEG}^\\circ$",
        fontsize=13, fontweight="bold", y=1.04,
    )

    fig.savefig(OUTPUT_DIR / "edc_fits.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: edc_fits.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
