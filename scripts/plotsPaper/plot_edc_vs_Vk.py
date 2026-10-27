"""Plot the K-point EDC peak separation versus Vk with twist uncertainty.

Reads a central combined K-grid HDF5 file and the two twist-endpoint files,
selects a fixed phiK slice and Vk range, saves the selected data as an NPZ, and
plots the TVB-to-moire peak separation with the endpoint-derived uncertainty.

Example:
    .venv/bin/python scripts/plotsPaper/plot_edc_vs_Vk.py \
        --id S11_k_2 --sample S11 --phi-k 120 --vk-min 1 --vk-max 20
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tmdmoire.constants import EDC_K_POSITIONS, TWIST_ANGLES

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"
TWIST_ENDPOINTS = {
    "S11": ("S11_m26", "S11_p30"),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", default="S11_k_2", help="Central K EDC run ID")
    parser.add_argument("--sample", default="S11", help="Central sample")
    parser.add_argument("--phi-k", type=float, default=120.0, help="Fixed phiK in degrees")
    parser.add_argument("--vk-min", type=float, default=1.0, help="Minimum Vk in meV")
    parser.add_argument("--vk-max", type=float, default=20.0, help="Maximum Vk in meV")
    parser.add_argument("--low-endpoint-id", help="Override lower-twist endpoint run ID")
    parser.add_argument("--high-endpoint-id", help="Override higher-twist endpoint run ID")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    if args.vk_min > args.vk_max:
        parser.error("--vk-min must be less than or equal to --vk-max")
    if (args.low_endpoint_id is None) != (args.high_endpoint_id is None):
        parser.error("--low-endpoint-id and --high-endpoint-id must be provided together")
    return args


def _project_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise SystemExit(f"Output must stay inside the repository: {path}")
    return path


def _load_vk_slice(run_id, sample, phi_k_deg, vk_min_mev, vk_max_mev):
    combined_path = REPO_ROOT / "Data" / f"edc_grid_k_{run_id}" / "combined.h5"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Combined K results not found: {combined_path}\n"
            f"Combine them with: .venv/bin/python scripts/combine_edc_chunks.py "
            f"--bz-point k --id {run_id}"
        )

    required = ("Vk", "phiK", "c1", "c2")
    with h5py.File(combined_path, "r") as h5_file:
        missing = [key for key in required if key not in h5_file]
        if missing:
            raise ValueError(f"{combined_path} is missing datasets: {', '.join(missing)}")
        data = {key: h5_file[key][:] for key in required}
        redchi = h5_file["redchi"][:] if "redchi" in h5_file else np.full_like(data["Vk"], np.nan)
        nrmse = h5_file["nrmse"][:] if "nrmse" in h5_file else np.full_like(data["Vk"], np.nan)

    vk = np.asarray(data["Vk"], dtype=float)
    phi_k = np.asarray(data["phiK"], dtype=float)
    center_1 = np.asarray(data["c1"], dtype=float)
    center_2 = np.asarray(data["c2"], dtype=float)
    redchi = np.asarray(redchi, dtype=float)
    nrmse = np.asarray(nrmse, dtype=float)
    if len(vk) == 0 or any(
        len(values) != len(vk)
        for values in (phi_k, center_1, center_2, redchi, nrmse)
    ):
        raise ValueError(f"Inconsistent or empty datasets in {combined_path}")

    phi_mask = np.isclose(phi_k, phi_k_deg, rtol=0, atol=1e-6)
    vk_mev = vk * 1000
    vk_mask = (vk_mev >= vk_min_mev - 1e-8) & (vk_mev <= vk_max_mev + 1e-8)
    selected = np.flatnonzero(phi_mask & vk_mask)
    if len(selected) == 0:
        available = np.unique(phi_k[np.isfinite(phi_k)])
        raise ValueError(
            f"No Vk points in {vk_min_mev:g}–{vk_max_mev:g} meV at "
            f"phiK={phi_k_deg:g}°. Available phiK: {available.tolist()}"
        )
    selected = selected[np.argsort(vk_mev[selected])]

    center_high = np.maximum(center_1[selected], center_2[selected])
    center_low = np.minimum(center_1[selected], center_2[selected])
    fit_valid = np.isfinite(center_high) & np.isfinite(center_low)
    separation_mev = np.full(len(selected), np.nan)
    separation_mev[fit_valid] = np.abs(center_high[fit_valid] - center_low[fit_valid]) * 1000

    return {
        "run_id": run_id,
        "sample": sample,
        "Vk_vals_meV": vk_mev[selected],
        "phiK_vals_deg": phi_k[selected],
        "c1_ev": center_high,
        "c2_ev": center_low,
        "separation_meV": separation_mev,
        "fit_valid": fit_valid,
        "redchi": redchi[selected],
        "nrmse": nrmse[selected],
    }


def _load_fixed_params(run_id):
    metadata_path = REPO_ROOT / "Data" / f"edc_grid_k_{run_id}" / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as stream:
        return json.load(stream).get("fixed_params", {})


def main():
    args = _parse_args()
    if args.sample not in EDC_K_POSITIONS:
        raise SystemExit(f"Unknown sample {args.sample!r}")
    experimental_positions = np.asarray(EDC_K_POSITIONS[args.sample], dtype=float)
    if not np.all(np.isfinite(experimental_positions)):
        raise SystemExit(f"No finite experimental K EDC positions are defined for {args.sample}")
    if args.sample not in TWIST_ENDPOINTS:
        raise SystemExit(f"No twist endpoints configured for {args.sample}")

    low_sample, high_sample = TWIST_ENDPOINTS[args.sample]
    low_id = args.low_endpoint_id or f"{low_sample}_k_2"
    high_id = args.high_endpoint_id or f"{high_sample}_k_2"

    central = _load_vk_slice(
        args.id, args.sample, args.phi_k, args.vk_min, args.vk_max
    )
    low_endpoint = _load_vk_slice(
        low_id, low_sample, args.phi_k, args.vk_min, args.vk_max
    )
    high_endpoint = _load_vk_slice(
        high_id, high_sample, args.phi_k, args.vk_min, args.vk_max
    )
    for endpoint in (low_endpoint, high_endpoint):
        if not np.allclose(
            endpoint["Vk_vals_meV"], central["Vk_vals_meV"], rtol=0, atol=1e-8
        ):
            raise ValueError(
                f"Run {endpoint['run_id']} does not use the same Vk grid as {args.id}"
            )

    central_distance = central["separation_meV"]
    low_distance = low_endpoint["separation_meV"]
    high_distance = high_endpoint["separation_meV"]
    uncertainty_valid = (
        central["fit_valid"]
        & low_endpoint["fit_valid"]
        & high_endpoint["fit_valid"]
    )
    endpoint_low = np.minimum(low_distance, high_distance)
    endpoint_high = np.maximum(low_distance, high_distance)
    error_low = np.full_like(central_distance, np.nan)
    error_high = np.full_like(central_distance, np.nan)
    error_low[uncertainty_valid] = np.maximum(
        central_distance[uncertainty_valid] - endpoint_low[uncertainty_valid], 0
    )
    error_high[uncertainty_valid] = np.maximum(
        endpoint_high[uncertainty_valid] - central_distance[uncertainty_valid], 0
    )

    data_dir = _project_path(args.data_dir)
    figure_dir = _project_path(args.figure_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    phi_str = f"{args.phi_k:g}".replace(".", "p")
    vk_range_str = f"{args.vk_min:g}-{args.vk_max:g}".replace(".", "p")
    run_tag = args.id[len(args.sample) + 1:] if args.id.startswith(f"{args.sample}_") else args.id
    stem = f"edc_vs_Vk_{args.sample}_{run_tag}_phiK{phi_str}_Vk{vk_range_str}meV"
    data_path = data_dir / f"{stem}.npz"
    figure_path = figure_dir / f"{stem}.png"

    fixed = _load_fixed_params(args.id)
    sample_theta = TWIST_ANGLES[args.sample]
    low_theta = TWIST_ANGLES[low_sample]
    high_theta = TWIST_ANGLES[high_sample]
    experimental_separation_mev = abs(
        experimental_positions[0] - experimental_positions[1]
    ) * 1000
    np.savez_compressed(
        data_path,
        sample=args.sample,
        run_id=args.id,
        theta_deg=sample_theta,
        phiK_deg=args.phi_k,
        Vk_vals_meV=central["Vk_vals_meV"],
        central_separation_meV=central_distance,
        low_endpoint_sample=low_sample,
        low_endpoint_run_id=low_id,
        low_endpoint_theta_deg=low_theta,
        low_endpoint_separation_meV=low_distance,
        high_endpoint_sample=high_sample,
        high_endpoint_run_id=high_id,
        high_endpoint_theta_deg=high_theta,
        high_endpoint_separation_meV=high_distance,
        error_low_meV=error_low,
        error_high_meV=error_high,
        central_fit_valid=central["fit_valid"],
        uncertainty_valid=uncertainty_valid,
        experimental_positions_ev=experimental_positions,
        experimental_separation_meV=experimental_separation_mev,
        redchi=central["redchi"],
        nrmse=central["nrmse"],
        fixed_Vg_ev=fixed.get("Vg_ev", np.nan),
        fixed_phiG_deg=fixed.get("phiG_deg", np.nan),
        fixed_w1p_ev=fixed.get("w1p_ev", np.nan),
        fixed_w1d_ev=fixed.get("w1d_ev", np.nan),
        fixed_w2p_ev=fixed.get("w2p_ev", np.nan),
        fixed_w2d_ev=fixed.get("w2d_ev", np.nan),
    )

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    center_valid = central["fit_valid"]
    vk_plot = central["Vk_vals_meV"][center_valid]
    separation_plot = central_distance[center_valid]
    ax.plot(
        vk_plot, separation_plot,
        "ko-", lw=1.5, markersize=6,
        label=f"Computed ({sample_theta:.1f}°)", zorder=2,
    )
    if np.any(uncertainty_valid):
        error_bars = np.vstack((error_low[uncertainty_valid], error_high[uncertainty_valid]))
        ax.errorbar(
            central["Vk_vals_meV"][uncertainty_valid],
            central_distance[uncertainty_valid],
            yerr=error_bars,
            fmt="none",
            ecolor="tab:blue",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            label=f"Twist range ({low_theta:.1f}°–{high_theta:.1f}°)",
            zorder=1,
        )
    ax.axhline(
        experimental_separation_mev,
        color="red", lw=1.5, ls="--",
        label=f"ARPES = {experimental_separation_mev:.1f} meV",
    )
    ax.set_xlabel(r"$V_K$ (meV)")
    ax.set_ylabel(r"TVB–moire peak separation $|E_1-E_2|$ (meV)")
    fixed_label = ""
    if "Vg_ev" in fixed and "phiG_deg" in fixed:
        fixed_label = (
            f"Vg={float(fixed['Vg_ev']) * 1000:.1f} meV, "
            f"phiG={float(fixed['phiG_deg']):.0f}°"
        )
    ax.set_title(
        f"K-point EDC peak separation: {args.sample}, {sample_theta:.3f}°, "
        f"phiK={args.phi_k:.0f}°\n{fixed_label}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Valid central fits: {int(center_valid.sum())}/{len(center_valid)}; "
        f"valid endpoint error bars: {int(uncertainty_valid.sum())}"
    )
    print(f"Saved data: {data_path}")
    print(f"Saved figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
