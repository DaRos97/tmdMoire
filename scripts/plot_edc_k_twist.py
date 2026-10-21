"""Summarize the best L2 point at fixed phiK across S11 twist-angle runs.

Discovers combined K EDC runs by ID suffix, chooses the minimum-L2 point over
Vk at the requested phiK for each twist angle, and plots theta vs. Vk. Marker
color encodes L2 and all markers use the same size.

Examples:
  python scripts/plot_edc_k_twist.py
  python scripts/plot_edc_k_twist.py --suffix _k_1
  python scripts/plot_edc_k_twist.py --phi-k 106 --max-l2 30
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from tmdmoire import EDC_K_POSITIONS, TWIST_ANGLES
from tmdmoire.utils.paths import get_repo_root


RUN_DIR_PREFIX = "edc_grid_k_"
S11_SAMPLES = tuple(
    sample for sample in TWIST_ANGLES
    if sample == "S11" or sample.startswith("S11_")
)
MARKER_SIZE = 100.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suffix", default="_k_2",
        help="Run-ID suffix to discover (default: _k_2)",
    )
    parser.add_argument(
        "--max-l2", type=float, default=30.0,
        help="Keep best points with L2 at or below this value in meV (default: 30)",
    )
    parser.add_argument(
        "--phi-k", type=float, default=106.0,
        help="Select points at this phiK in degrees (default: 106)",
    )
    parser.add_argument(
        "--sample-max-vk", action="append", default=[], metavar="SAMPLE=MEV",
        help="Optional per-sample Vk maximum in meV; may be repeated",
    )
    parser.add_argument("--output", type=Path, help="Summary plot output path")
    parser.add_argument("--csv-output", type=Path, help="Summary CSV output path")
    args = parser.parse_args()
    if not args.suffix:
        parser.error("--suffix must not be empty")
    if args.max_l2 < 0:
        parser.error("--max-l2 must be non-negative")
    args.sample_vk_max = {}
    for entry in args.sample_max_vk:
        try:
            sample, value = entry.rsplit("=", 1)
            max_vk = float(value)
        except ValueError:
            parser.error(f"Invalid --sample-max-vk value {entry!r}; expected SAMPLE=MEV")
        if sample not in TWIST_ANGLES or not np.isfinite(max_vk) or max_vk < 0:
            parser.error(f"Invalid sample or Vk maximum in --sample-max-vk {entry!r}")
        args.sample_vk_max[sample] = max_vk
    return args


def _repo_path(path, repo_root):
    path = Path(path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    path = path.resolve()
    if not path.is_relative_to(Path(repo_root).resolve()):
        raise SystemExit(f"Output must stay inside the repository: {path}")
    return path


def _best_l2_point(
    combined_path, sample, run_id, max_l2_mev, phi_k_deg, max_vk_mev=None
):
    experimental = np.asarray(EDC_K_POSITIONS[sample], dtype=float)
    if not np.all(np.isfinite(experimental)):
        return None

    required = ("Vk", "phiK", "c1", "c2")
    with h5py.File(combined_path, "r") as h5_file:
        missing = [column for column in required if column not in h5_file]
        if missing:
            raise ValueError(f"{combined_path} is missing datasets: {', '.join(missing)}")
        data = {column: h5_file[column][:] for column in required}

    vk = np.asarray(data["Vk"], dtype=float)
    phi_k = np.asarray(data["phiK"], dtype=float)
    c1 = np.asarray(data["c1"], dtype=float)
    c2 = np.asarray(data["c2"], dtype=float)
    if len(vk) == 0 or any(len(values) != len(vk) for values in (phi_k, c1, c2)):
        raise ValueError(f"Inconsistent or empty datasets in {combined_path}")

    center_high = np.maximum(c1, c2)
    center_low = np.minimum(c1, c2)
    valid = (
        np.isfinite(vk) & np.isfinite(phi_k)
        & np.isfinite(center_high) & np.isfinite(center_low)
    )
    l2_mev = np.full(len(vk), np.nan)
    l2_mev[valid] = 1000 * np.abs(
        np.abs(center_high[valid] - center_low[valid])
        - abs(experimental[0] - experimental[1])
    )
    valid &= np.isfinite(l2_mev) & (l2_mev <= max_l2_mev)
    valid &= np.isclose(phi_k, phi_k_deg, rtol=0, atol=1e-6)
    if max_vk_mev is not None:
        valid &= vk * 1000 <= max_vk_mev
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        return None

    best_index = candidates[np.argmin(l2_mev[candidates])]
    return {
        "sample": sample,
        "run_id": run_id,
        "theta_deg": float(TWIST_ANGLES[sample]),
        "Vk_meV": float(vk[best_index] * 1000),
        "phiK_deg": float(phi_k[best_index]),
        "L2_meV": float(l2_mev[best_index]),
        "max_Vk_meV": max_vk_mev,
        "c1_ev": float(center_high[best_index]),
        "c2_ev": float(center_low[best_index]),
    }


def _size_for_l2(l2_mev):
    return MARKER_SIZE_BASE + MARKER_SIZE_PER_MEV * l2_mev


def main():
    args = _parse_args()
    repo_root = get_repo_root()
    data_root = Path(repo_root) / "Data"

    expected_samples = sorted(S11_SAMPLES, key=lambda sample: TWIST_ANGLES[sample])
    rows = []
    missing_runs = []
    missing_combined = []

    for sample in expected_samples:
        run_id = f"{sample}{args.suffix}"
        run_dir = data_root / f"{RUN_DIR_PREFIX}{run_id}"
        combined_path = run_dir / "combined.h5"
        if not run_dir.is_dir():
            missing_runs.append(run_id)
            continue
        if not combined_path.exists():
            missing_combined.append(run_id)
            continue

        max_vk_mev = args.sample_vk_max.get(sample)
        point = _best_l2_point(
            combined_path, sample, run_id, args.max_l2, args.phi_k, max_vk_mev
        )
        if point is None:
            message = (
                f"No fit for {sample} at phiK={args.phi_k:g} deg passes "
                f"L2 <= {args.max_l2:g} meV"
            )
            if max_vk_mev is not None:
                message += f" and Vk <= {max_vk_mev:g} meV"
            print(message)
            continue
        rows.append(point)

    if missing_runs:
        print(f"Runs not found for suffix {args.suffix!r}: {', '.join(missing_runs)}")
    if missing_combined:
        print(f"Runs need combining: {', '.join(missing_combined)}")
    if not rows:
        raise SystemExit("No combined S11 K runs produced a point within the L2 cutoff")

    rows.sort(key=lambda row: row["theta_deg"])
    theta = np.array([row["theta_deg"] for row in rows])
    vk_mev = np.array([row["Vk_meV"] for row in rows])
    l2_mev = np.array([row["L2_meV"] for row in rows])

    suffix_label = args.suffix.strip("_").replace("_", "") or "runs"
    output = _repo_path(
        args.output or data_root / f"edc_k_twist_summary_{suffix_label}.png",
        repo_root,
    )
    csv_output = _repo_path(
        args.csv_output or output.with_suffix(".csv"), repo_root
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    l2_min = float(np.min(l2_mev))
    l2_max = float(np.max(l2_mev))
    if np.isclose(l2_min, l2_max):
        l2_min = max(0.0, l2_min - 0.5)
        l2_max += 0.5

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(theta, vk_mev, linestyle="--", color="0.45", linewidth=1.5, zorder=1)
    l2_norm = PowerNorm(gamma=0.45, vmin=l2_min, vmax=l2_max)
    points = ax.scatter(
        theta, vk_mev,
        c=l2_mev,
        s=MARKER_SIZE,
        cmap="plasma",
        norm=l2_norm,
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Best-fit L2 (meV)")

    zoom_mask = (theta >= 2.5) & (theta <= 2.65)
    if np.any(zoom_mask):
        zoom_theta = theta[zoom_mask]
        zoom_vk = vk_mev[zoom_mask]
        zoom_l2 = l2_mev[zoom_mask]
        inset_l2_min = float(np.min(zoom_l2))
        inset_l2_max = float(np.max(zoom_l2))
        if np.isclose(inset_l2_min, inset_l2_max):
            inset_l2_min = max(0.0, inset_l2_min - 0.5)
            inset_l2_max += 0.5
        inset_l2_norm = PowerNorm(
            gamma=0.45, vmin=inset_l2_min, vmax=inset_l2_max
        )
        inset = ax.inset_axes([0.54, 0.06, 0.42, 0.42])
        inset.plot(zoom_theta, zoom_vk, linestyle="--", color="0.45", linewidth=1.2)
        inset_points = inset.scatter(
            zoom_theta, zoom_vk,
            c=zoom_l2,
            s=MARKER_SIZE,
            cmap="plasma",
            norm=inset_l2_norm,
            edgecolors="black",
            linewidths=0.7,
            zorder=2,
        )
        inset.set_xlim(2.5, 2.65)
        y_min = float(np.min(zoom_vk))
        y_max = float(np.max(zoom_vk))
        y_padding = max(0.5, 0.08 * (y_max - y_min))
        inset.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        inset.set_xticks(np.arange(2.5, 2.6501, 0.05))
        inset.set_title(r"$2.5^\circ$–$2.65^\circ$", fontsize=9)
        inset.set_xlabel(r"$\theta$ (deg)", fontsize=8)
        inset.tick_params(labelsize=8)
        inset.grid(True, alpha=0.2)
        ax.indicate_inset_zoom(inset, edgecolor="0.4", linewidth=0.8)
        inset_colorbar = fig.colorbar(
            inset_points, ax=inset, location="left", fraction=0.06, pad=0.06
        )
        inset_colorbar.set_label("Inset L2 (meV)", fontsize=7)
        inset_colorbar.ax.tick_params(labelsize=7)

    ax.set_xlabel(r"Twist angle $\theta$ (deg)")
    ax.set_ylabel(r"Best-fit $V_K$ (meV)")
    main_tick_start = np.floor(np.min(theta) * 10) / 10
    main_tick_end = np.ceil(np.max(theta) * 10) / 10
    ax.set_xticks(np.arange(main_tick_start, main_tick_end + 0.05, 0.1))
    ax.grid(True, alpha=0.25)
    filter_labels = [f"L2 ≤ {args.max_l2:g} meV"]
    filter_labels.append(f"phiK = {args.phi_k:g}°")
    vk_cutoff_groups = {}
    for sample, max_vk in args.sample_vk_max.items():
        vk_cutoff_groups.setdefault(max_vk, []).append(TWIST_ANGLES[sample])
    for max_vk, angles in sorted(vk_cutoff_groups.items()):
        if len(angles) == 1:
            angle_label = f"at {angles[0]:g}°"
        else:
            angle_label = f"for {len(angles)} angles, θ={min(angles):g}–{max(angles):g}°"
        filter_labels.append(f"Vk ≤ {max_vk:g} meV {angle_label}")
    ax.set_title(
        f"Minimum L2 at fixed phiK by twist angle ({len(rows)} angles; "
        f"{'; '.join(filter_labels)})"
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fields = (
        "sample", "run_id", "theta_deg", "Vk_meV", "phiK_deg", "L2_meV",
        "max_Vk_meV", "c1_ev", "c2_ev",
    )
    with open(csv_output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Best L2 point per twist angle:")
    print("sample       theta    Vk      phiK     L2")
    for row in rows:
        print(
            f"{row['sample']:<12} {row['theta_deg']:>5.1f}°  "
            f"{row['Vk_meV']:>6.2f}  {row['phiK_deg']:>6.1f}°  "
            f"{row['L2_meV']:>7.3f} meV"
        )
    print(f"Plot saved to {output}")
    print(f"Summary saved to {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
