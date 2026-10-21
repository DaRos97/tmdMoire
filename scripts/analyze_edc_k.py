"""Analyze K-point EDC grid results.

Loads ``combined.h5``, compares the two fitted peak positions with experiment,
plots peak-distance and fit-quality heatmaps, and makes a profile plot for the
best or explicitly selected (Vk, phiK) point.

Usage::

    python scripts/analyze_edc_k.py --id 001
    python scripts/analyze_edc_k.py --id 001 --max-nrmse 0.05
    python scripts/analyze_edc_k.py --id 001 --vk 0.010 --phik 120
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tmdmoire import (
    EDC_K_POSITIONS,
    ENERGY_OFFSETS,
    LATTICE_CONSTANTS,
    MoireGeometry,
    MoireHamiltonian,
    TMDMaterial,
    TWIST_ANGLES,
)
from tmdmoire.utils.paths import get_repo_root


SPREAD_E = 0.03
ENERGY_STEP = 0.005
N_SHELLS = 2


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="default", help="K EDC run ID")
    parser.add_argument("--sample", default="S11", help="Sample name (default: S11)")
    parser.add_argument("--output", type=Path, help="Analysis figure path")
    parser.add_argument(
        "--max-nrmse",
        type=float,
        help="Optional maximum normalized fit RMSE; no fit-quality cutoff by default",
    )
    parser.add_argument(
        "--max-redchi",
        type=float,
        help="Optional maximum lmfit reduced chi-square",
    )
    parser.add_argument(
        "--max-l2",
        type=float,
        help="Optional maximum peak-separation error in meV",
    )
    parser.add_argument(
        "--max-vk",
        type=float,
        help="Optional maximum Vk in meV for distance maps and selection",
    )
    parser.add_argument("--vk", type=float, help="Selected Vk in eV (requires --phik)")
    parser.add_argument("--phik", type=float, help="Selected phiK in degrees (requires --vk)")
    args = parser.parse_args()

    if (args.vk is None) != (args.phik is None):
        parser.error("--vk and --phik must be supplied together")
    for name in ("max_nrmse", "max_redchi", "max_l2", "max_vk"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    return args


def _cell_edges(values, fallback_step):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - fallback_step / 2, values[0] + fallback_step / 2])
    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _make_surface(values, valid, vk, phi_k, vk_values, phi_k_values):
    surface = np.full((len(vk_values), len(phi_k_values)), np.nan)
    vk_to_index = {value: index for index, value in enumerate(vk_values)}
    phi_to_index = {value: index for index, value in enumerate(phi_k_values)}

    for i in np.flatnonzero(valid & np.isfinite(values)):
        row = vk_to_index[vk[i]]
        column = phi_to_index[phi_k[i]]
        if np.isnan(surface[row, column]) or values[i] < surface[row, column]:
            surface[row, column] = values[i]

    return surface


def _plot_surface(ax, surface, phi_edges, vk_edges_mev, colorbar_label, cmap,
                  fallback_text):
    if np.isfinite(surface).any():
        image = ax.pcolormesh(
            phi_edges, vk_edges_mev, surface,
            cmap=cmap, shading="flat",
        )
        colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(colorbar_label)
        best_cell = np.unravel_index(np.nanargmin(surface), surface.shape)
        return best_cell

    ax.text(0.5, 0.5, fallback_text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(phi_edges[0], phi_edges[-1])
    ax.set_ylim(vk_edges_mev[0], vk_edges_mev[-1])
    return None


def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def _load_fixed_params(run_dir, metadata, master_folder):
    fixed = metadata.get("fixed_params", {})
    required = ("Vg_ev", "phiG_deg", "w1p_ev", "w1d_ev", "w2p_ev", "w2d_ev")
    if all(key in fixed for key in required):
        return fixed

    config_path = run_dir / "grid_config.json"
    if not config_path.exists():
        config_path = Path(master_folder) / "Inputs/bilayer_fitting/grid_config_k.json"
    with open(config_path) as stream:
        config = json.load(stream)
    config_fixed = config.get("fixed", {})
    missing = [key for key in required if key not in config_fixed]
    if missing:
        raise ValueError(f"Missing fixed K-sweep parameters: {', '.join(missing)}")
    return config_fixed


def _compute_profile(sample, vk, phi_k_deg, fixed, master_folder):
    monolayer_fns = {
        "WSe2": Path(master_folder)
        / "Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
        "WS2": Path(master_folder)
        / "Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
    }
    wse2 = TMDMaterial("WSe2")
    wse2.load_fitted(str(monolayer_fns["WSe2"]))
    ws2 = TMDMaterial("WS2")
    ws2.load_fitted(str(monolayer_fns["WS2"]))

    theta = TWIST_ANGLES[sample]
    geometry = MoireGeometry(theta)
    n_cells = MoireGeometry.n_cells(N_SHELLS)
    k_k_point = np.array([[4 * np.pi / (3 * LATTICE_CONSTANTS["WSe2"]), 0]])
    interlayer_params = {
        "stacking": "P",
        "w1p": float(fixed["w1p_ev"]),
        "w1d": float(fixed["w1d_ev"]),
        "w2p": float(fixed["w2p_ev"]),
        "w2d": float(fixed["w2d_ev"]),
    }
    phi_g = float(fixed["phiG_deg"]) / 180 * np.pi
    phi_k_rad = phi_k_deg / 180 * np.pi
    pars_v = (float(fixed["Vg_ev"]), vk, phi_g, phi_k_rad)

    moire_ham = MoireHamiltonian(wse2, ws2, geometry)
    evals_raw, evecs_raw = moire_ham.diagonalize(
        k_k_point, N_SHELLS, interlayer_params, pars_v
    )
    evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
    evecs_raw = evecs_raw[0]

    probabilities = np.abs(evecs_raw) ** 2
    weights = (
        np.sum(probabilities[:22, :], axis=0)
        + np.sum(probabilities[22 * n_cells:22 * (1 + n_cells), :], axis=0)
    )

    index_tvb = 28 * n_cells - 1
    lower_candidates = weights[index_tvb - n_cells:index_tvb]
    index_lower = index_tvb - n_cells + np.argmax(lower_candidates) + 1
    energies = evals_raw[index_lower:index_tvb + 1]
    band_weights = weights[index_lower:index_tvb + 1]
    if len(energies) == 0:
        raise ValueError("Could not build the K-point EDC energy window")

    energy_min = energies[0]
    energy_max = energies[-1]
    delta = energy_max - energy_min
    energy_min -= delta / 2
    energy_max += delta / 2
    n_energy = max(2, int((energy_max - energy_min) / ENERGY_STEP))
    energy_list = np.linspace(energy_min, energy_max, n_energy)
    intensity = np.zeros_like(energy_list)
    for energy, weight in zip(energies, band_weights):
        intensity += SPREAD_E / np.pi * weight / ((energy_list - energy) ** 2 + SPREAD_E**2)

    return energy_list, intensity


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    args = _parse_args()
    master_folder = get_repo_root()
    run_dir = Path(master_folder) / "Data" / f"edc_grid_k_{args.id}"
    combined_path = run_dir / "combined.h5"

    if not combined_path.exists():
        raise SystemExit(
            f"Combined file not found: {combined_path}\n"
            f"Run: python scripts/combine_edc_chunks.py --bz-point k --id {args.id}"
        )
    if args.sample not in EDC_K_POSITIONS:
        raise SystemExit(f"Unknown sample {args.sample!r}; known samples: {sorted(EDC_K_POSITIONS)}")
    experimental = np.asarray(EDC_K_POSITIONS[args.sample], dtype=float)
    if not np.all(np.isfinite(experimental)):
        raise SystemExit(f"No finite experimental K EDC positions are defined for {args.sample}")

    with h5py.File(combined_path, "r") as h5_file:
        required_columns = ("Vk", "phiK", "c1", "c2", "a1", "a2", "g1", "g2", "redchi")
        missing = [column for column in required_columns if column not in h5_file]
        if missing:
            raise SystemExit(f"{combined_path} is missing datasets: {', '.join(missing)}")
        data = {column: h5_file[column][:] for column in required_columns}
        # Runs created before the fit-quality metric was added remain analyzable.
        nrmse = h5_file["nrmse"][:] if "nrmse" in h5_file else np.full_like(data["Vk"], np.nan)

    run_metadata_path = run_dir / "run_metadata.json"
    if run_metadata_path.exists():
        with open(run_metadata_path) as stream:
            metadata = json.load(stream)
    else:
        metadata = {}

    vk = np.asarray(data["Vk"], dtype=float)
    phi_k = np.asarray(data["phiK"], dtype=float)
    c1 = np.asarray(data["c1"], dtype=float)
    c2 = np.asarray(data["c2"], dtype=float)
    a1 = np.asarray(data["a1"], dtype=float)
    a2 = np.asarray(data["a2"], dtype=float)
    g1 = np.asarray(data["g1"], dtype=float)
    g2 = np.asarray(data["g2"], dtype=float)
    redchi = np.asarray(data["redchi"], dtype=float)
    nrmse = np.asarray(nrmse, dtype=float)

    if len(vk) == 0 or any(len(values) != len(vk) for values in (phi_k, c1, c2, a1, a2, g1, g2, redchi, nrmse)):
        raise SystemExit(f"Inconsistent or empty datasets in {combined_path}")

    # Canonicalize peak labels by energy: c1/a1/g1 is always the higher-energy peak.
    swap = c1 < c2
    center_high = np.where(swap, c2, c1)
    center_low = np.where(swap, c1, c2)
    amp_high = np.where(swap, a2, a1)
    amp_low = np.where(swap, a1, a2)
    gamma_high = np.where(swap, g2, g1)
    gamma_low = np.where(swap, g1, g2)

    fit_valid = (
        np.isfinite(vk) & np.isfinite(phi_k)
        & np.isfinite(center_high) & np.isfinite(center_low)
        & np.isfinite(amp_high) & np.isfinite(amp_low)
        & np.isfinite(gamma_high) & np.isfinite(gamma_low)
        & (amp_high >= 0) & (amp_low >= 0)
        & (gamma_high > 0) & (gamma_low > 0)
    )
    n_fit_valid = int(fit_valid.sum())

    dist_l1 = np.full(len(vk), np.nan)
    dist_l1[fit_valid] = (
        np.abs(center_high[fit_valid] - experimental[0])
        + np.abs(center_low[fit_valid] - experimental[1])
    )
    experimental_separation = abs(experimental[0] - experimental[1])
    dist_l2 = np.full(len(vk), np.nan)
    dist_l2[fit_valid] = np.abs(
        np.abs(center_high[fit_valid] - center_low[fit_valid]) - experimental_separation
    )

    quality_valid = fit_valid.copy()
    if args.max_nrmse is not None:
        if not np.isfinite(nrmse).any():
            raise SystemExit("This run has no NRMSE data; rerun the K sweep before using --max-nrmse")
        quality_valid &= np.isfinite(nrmse) & (nrmse <= args.max_nrmse)
    if args.max_redchi is not None:
        quality_valid &= np.isfinite(redchi) & (redchi <= args.max_redchi)

    distance_valid = quality_valid.copy()
    if args.max_l2 is not None:
        distance_valid &= np.isfinite(dist_l2) & (dist_l2 * 1000 <= args.max_l2)
    if args.max_vk is not None:
        distance_valid &= vk * 1000 <= args.max_vk

    n_quality_valid = int(quality_valid.sum())
    n_distance_valid = int(distance_valid.sum())
    print(f"Loaded {len(vk)} grid points from {combined_path}")
    print(f"Successful finite two-Lorentzian fits: {n_fit_valid} / {len(vk)}")
    print(f"Fits passing fit-quality filters: {n_quality_valid} / {len(vk)}")
    print(f"Fits used for distance ranking: {n_distance_valid} / {len(vk)}")
    if not np.isfinite(nrmse).any():
        print("NRMSE is unavailable (this may be a result file from before NRMSE was recorded).")
    if args.max_nrmse is not None:
        print(f"Quality filter: NRMSE <= {args.max_nrmse:g}")
    if args.max_redchi is not None:
        print(f"Quality filter: redchi <= {args.max_redchi:g}")
    if args.max_l2 is not None:
        print(f"Distance filter: L2 <= {args.max_l2:g} meV")
    if args.max_vk is not None:
        print(f"Distance filter: Vk <= {args.max_vk:g} meV")

    vk_values = np.unique(vk[np.isfinite(vk)])
    phi_k_values = np.unique(phi_k[np.isfinite(phi_k)])
    if len(vk_values) == 0 or len(phi_k_values) == 0:
        raise SystemExit("No finite Vk/phiK grid coordinates were found")

    dist_l1_surface = _make_surface(dist_l1, distance_valid, vk, phi_k, vk_values, phi_k_values)
    dist_l2_surface = _make_surface(dist_l2, distance_valid, vk, phi_k, vk_values, phi_k_values)
    nrmse_surface = _make_surface(nrmse, fit_valid, vk, phi_k, vk_values, phi_k_values)
    redchi_surface = _make_surface(redchi, fit_valid, vk, phi_k, vk_values, phi_k_values)

    phi_edges = _cell_edges(phi_k_values, 10.0)
    vk_edges_mev = _cell_edges(vk_values, 0.002) * 1000
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    plot_specs = [
        (axes[0, 0], dist_l1_surface * 1000, "L1 peak-position error", "Distance (meV)", "viridis_r", "No fits pass the selected filters"),
        (axes[0, 1], dist_l2_surface * 1000, "L2 peak-separation error", "Distance (meV)", "plasma_r", "No fits pass the selected filters"),
        (axes[1, 0], nrmse_surface, "Two-Lorentzian normalized RMSE", "NRMSE", "magma_r", "NRMSE unavailable"),
        (axes[1, 1], redchi_surface, "Two-Lorentzian reduced chi-square", "Reduced chi-square", "cividis_r", "Reduced chi-square unavailable"),
    ]
    for ax, surface, title, colorbar_label, cmap, fallback in plot_specs:
        best_cell = _plot_surface(
            ax, surface, phi_edges, vk_edges_mev, colorbar_label, cmap, fallback
        )
        if best_cell is not None:
            best_row, best_column = best_cell
            ax.scatter(
                phi_k_values[best_column], vk_values[best_row] * 1000,
                marker="*", s=130, c="cyan", edgecolors="black", linewidths=0.8,
                zorder=5,
            )
        ax.set_xlabel(r"$\phi_K$ (deg)")
        ax.set_ylabel(r"$V_K$ (meV)")
        ax.set_title(title)

    filter_description = []
    if args.max_nrmse is not None:
        filter_description.append(f"NRMSE≤{args.max_nrmse:g}")
    if args.max_redchi is not None:
        filter_description.append(f"redchi≤{args.max_redchi:g}")
    if args.max_l2 is not None:
        filter_description.append(f"L2≤{args.max_l2:g} meV")
    if args.max_vk is not None:
        filter_description.append(f"Vk≤{args.max_vk:g} meV")
    filter_text = ", ".join(filter_description) if filter_description else "no cutoffs"
    fig.suptitle(f"EDC at K: {args.id} ({args.sample}) — {filter_text}", fontsize=14)

    output = args.output or (run_dir / "analysis.png")
    if not output.is_absolute():
        output = Path(master_folder) / output
    output = output.resolve()
    repo_root = Path(master_folder).resolve()
    if not output.is_relative_to(repo_root):
        raise SystemExit(f"Analysis output must stay inside the repository: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis figure saved to {output}")

    candidate_indices = np.flatnonzero(distance_valid & np.isfinite(dist_l1))
    if len(candidate_indices) == 0:
        print("No fit points remain for EDC-position analysis; no profile was generated.")
        return 0

    if args.vk is not None:
        selected_mask = (
            np.isclose(vk, args.vk, rtol=0, atol=1e-8)
            & np.isclose(phi_k, args.phik, rtol=0, atol=1e-6)
            & distance_valid & np.isfinite(dist_l1)
        )
        selected_indices = np.flatnonzero(selected_mask)
        if len(selected_indices) == 0:
            raise SystemExit(
                f"No accepted fit at Vk={args.vk:g} eV, phiK={args.phik:g} deg"
            )
        selected_index = selected_indices[np.argmin(dist_l1[selected_indices])]
    else:
        selected_index = candidate_indices[np.argmin(dist_l1[candidate_indices])]

    print("\nBest accepted EDC fit:")
    print(f"  Vk       = {vk[selected_index] * 1000:.3f} meV")
    print(f"  phiK     = {phi_k[selected_index]:.3f} deg")
    print(f"  c1       = {center_high[selected_index]:.6f} eV (exp: {experimental[0]:.6f} eV)")
    print(f"  c2       = {center_low[selected_index]:.6f} eV (exp: {experimental[1]:.6f} eV)")
    print(f"  L1       = {dist_l1[selected_index] * 1000:.3f} meV")
    print(f"  L2       = {dist_l2[selected_index] * 1000:.3f} meV")
    print(f"  a2/a1    = {amp_low[selected_index] / amp_high[selected_index] if amp_high[selected_index] > 0 else np.nan:.5g}")
    print(f"  redchi   = {redchi[selected_index]:.6g}")
    print(f"  NRMSE    = {nrmse[selected_index]:.6g}")

    fixed = _load_fixed_params(run_dir, metadata, master_folder)
    energy_list, intensity = _compute_profile(
        args.sample, vk[selected_index], phi_k[selected_index], fixed, master_folder
    )
    fit_curve = (
        _lorentzian(energy_list, amp_high[selected_index], center_high[selected_index], gamma_high[selected_index])
        + _lorentzian(energy_list, amp_low[selected_index], center_low[selected_index], gamma_low[selected_index])
    )
    peak_curves = [
        _lorentzian(energy_list, amp_high[selected_index], center_high[selected_index], gamma_high[selected_index]),
        _lorentzian(energy_list, amp_low[selected_index], center_low[selected_index], gamma_low[selected_index]),
    ]

    profile_fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(energy_list, intensity, color="black", lw=1.5, label="EDC intensity")
    ax.plot(
        energy_list, fit_curve, "r--", lw=2,
        label=f"2-Lorentzian fit (redchi={redchi[selected_index]:.4g}, NRMSE={nrmse[selected_index]:.4g})",
    )
    peak_colors = ("C0", "C1")
    for i, (curve, color, center, expected) in enumerate(zip(
        peak_curves, peak_colors,
        (center_high[selected_index], center_low[selected_index]), experimental,
    ), start=1):
        ax.plot(energy_list, curve, color=color, ls="-.", lw=1.2,
                label=f"Peak {i}: {center:.4f} eV (exp. {expected:.4f} eV)")
        ax.axvline(expected, color="green", ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(
        f"EDC at K: Vk={vk[selected_index] * 1000:.2f} meV, "
        f"phiK={phi_k[selected_index]:.1f} deg\n"
        f"Vg={float(fixed['Vg_ev']) * 1000:.2f} meV, phiG={float(fixed['phiG_deg']):.1f} deg"
    )
    ax.legend(fontsize=9)

    profile_path = run_dir / (
        f"edc_profile_Vk{vk[selected_index] * 1000:.1f}meV_"
        f"phiK{phi_k[selected_index]:.1f}deg.png"
    )
    profile_fig.savefig(profile_path, dpi=200, bbox_inches="tight")
    plt.close(profile_fig)
    print(f"EDC profile saved to {profile_path}")

    exported = {
        "run_id": args.id,
        "sample": args.sample,
        "Vk": float(vk[selected_index]),
        "Vk_ev": float(vk[selected_index]),
        "Vk_meV": float(vk[selected_index] * 1000),
        "phiK_deg": float(phi_k[selected_index]),
        "phiK_rad": float(phi_k[selected_index] / 180 * np.pi),
        "fixed_params": {key: float(fixed[key]) for key in (
            "Vg_ev", "phiG_deg", "w1p_ev", "w1d_ev", "w2p_ev", "w2d_ev"
        )},
        "experimental_positions_ev": experimental.tolist(),
        "c1_ev": float(center_high[selected_index]),
        "c2_ev": float(center_low[selected_index]),
        "a1": float(amp_high[selected_index]),
        "a2": float(amp_low[selected_index]),
        "g1_ev": float(gamma_high[selected_index]),
        "g2_ev": float(gamma_low[selected_index]),
        "redchi": _finite_or_none(redchi[selected_index]),
        "nrmse": _finite_or_none(nrmse[selected_index]),
        "L1_meV": float(dist_l1[selected_index] * 1000),
        "L2_meV": float(dist_l2[selected_index] * 1000),
        "analysis_filters": {
            "max_nrmse": args.max_nrmse,
            "max_redchi": args.max_redchi,
            "max_l2_mev": args.max_l2,
            "max_vk_mev": args.max_vk,
        },
    }
    params_path = run_dir / "best_fit.json"
    with open(params_path, "w") as stream:
        json.dump(exported, stream, indent=2)
    print(f"Best-fit parameters saved to {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
