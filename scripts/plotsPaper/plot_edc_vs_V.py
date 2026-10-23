"""Standalone EDC TVB–side band distance vs V_G plot.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_vs_V.py.

Produces:
  edc_vs_V_<sample>_phiG<phase>.png -- TVB-to-side-band peak distance (meV)
                                      vs V_G (meV), with ARPES reference line
                                      and optional twist-angle error bars.

Usage:
    python plot_edc_vs_V.py <data.npz>
    python plot_edc_vs_V.py data.npz --output-dir ./figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TWIST_ENDPOINTS = {
    "S11": (("S11_m26", 2.6), ("S11_p30", 3.0)),
    "S3": (("S3_m16", 1.6), ("S3_p20", 2.0)),
}

CENTRAL_TWIST_ANGLES = {"S11": 2.8, "S3": 1.8}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="EDC-vs-Vg data exported to .npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    parser.add_argument(
        "--twist-low-data",
        type=Path,
        help="Optional lower-angle endpoint data (auto-discovered beside the central file)",
    )
    parser.add_argument(
        "--twist-high-data",
        type=Path,
        help="Optional higher-angle endpoint data (auto-discovered beside the central file)",
    )
    args = parser.parse_args()
    if (args.twist_low_data is None) != (args.twist_high_data is None):
        parser.error("--twist-low-data and --twist-high-data must be provided together")
    return args


def _load_npz(path):
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _infer_twist_data_path(data_path, variant_sample, central_sample):
    """Build the matching endpoint filename used by export_edc_vs_V.py."""
    name = data_path.name
    central_prefix = f"edc_vs_V_{central_sample}_"
    generic_prefix = "edc_vs_V_"
    if name.startswith(central_prefix):
        suffix = name[len(central_prefix):]
    elif name.startswith(generic_prefix):
        # Also support older S11 exports whose filename omits the sample name.
        suffix = name[len(generic_prefix):]
    else:
        return None
    return data_path.with_name(f"edc_vs_V_{variant_sample}_{suffix}")


def _load_twist_endpoints(args, data_path, sample, central_data):
    if sample not in TWIST_ENDPOINTS:
        if args.twist_low_data is not None:
            raise ValueError("Twist endpoint data is configured for S11 and S3 only.")
        return None

    endpoint_specs = TWIST_ENDPOINTS[sample]
    if args.twist_low_data is not None:
        paths = (args.twist_low_data, args.twist_high_data)
    else:
        paths = tuple(
            _infer_twist_data_path(data_path, variant_sample, sample)
            for variant_sample, _ in endpoint_specs
        )

    if any(path is None for path in paths):
        print("Twist error bars omitted: could not infer endpoint data filenames.")
        return None

    present = [path.is_file() for path in paths]
    if not any(present):
        endpoint_samples = " and ".join(spec[0] for spec in endpoint_specs)
        endpoint_range = f"{endpoint_specs[0][1]:.1f}°–{endpoint_specs[1][1]:.1f}°"
        print(
            "Twist error bars omitted: endpoint data not found. Export matching "
            f"{endpoint_samples} data to plot the {endpoint_range} range."
        )
        return None
    if not all(present):
        missing = ", ".join(str(path) for path, exists in zip(paths, present) if not exists)
        raise FileNotFoundError(f"Both twist endpoints are required; missing: {missing}")

    endpoint_distances = []
    for path, (expected_sample, expected_theta) in zip(paths, endpoint_specs):
        endpoint_data = _load_npz(path)
        endpoint_sample = str(endpoint_data.get("sample", np.array("")).item())
        if endpoint_sample != expected_sample:
            raise ValueError(
                f"{path} contains sample {endpoint_sample!r}; expected {expected_sample!r}."
            )
        if "theta_deg" in endpoint_data and not np.isclose(
            float(endpoint_data["theta_deg"]), expected_theta
        ):
            raise ValueError(
                f"{path} has theta={float(endpoint_data['theta_deg']):g}°; "
                f"expected {expected_theta:g}°."
            )
        if not np.allclose(
            endpoint_data["Vg_vals_meV"], central_data["Vg_vals_meV"], rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"{path} does not use the same Vg values as {data_path}.")
        if endpoint_data["distances_meV"].shape != central_data["distances_meV"].shape:
            raise ValueError(f"{path} has a different number of EDC values from {data_path}.")
        for key in (
            "interlayer_w1p", "interlayer_w1d", "interlayer_w2p", "interlayer_w2d", "phiG_deg"
        ):
            if not np.isclose(float(endpoint_data[key]), float(central_data[key])):
                raise ValueError(f"{path} has a different {key} from {data_path}.")
        endpoint_distances.append(endpoint_data["distances_meV"])

    return endpoint_distances


def main():
    args = _parse_args()
    data_path = args.data
    output_dir = args.output_dir

    d = _load_npz(data_path)

    Vg_vals_meV = d["Vg_vals_meV"]
    distances_meV = d["distances_meV"]
    arpes_distance_meV = float(d["arpes_distance_meV"])
    w1p = float(d["interlayer_w1p"])
    w1d = float(d["interlayer_w1d"])
    w2p = float(d["interlayer_w2p"])
    w2d = float(d["interlayer_w2d"])
    phiG_deg = float(d["phiG_deg"])
    sample = str(d["sample"].item()) if "sample" in d else (
        "S3" if "S3" in data_path.name else "S11"
    )
    theta_deg = (
        float(d["theta_deg"])
        if "theta_deg" in d
        else CENTRAL_TWIST_ANGLES.get(sample)
    )
    endpoint_distances = _load_twist_endpoints(args, data_path, sample, d)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    yerr = None
    if endpoint_distances is not None:
        # Show the full interval between the two endpoint curves, including
        # cases where both endpoints fall on the same side of the central curve.
        lower_endpoint = np.minimum(endpoint_distances[0], endpoint_distances[1])
        upper_endpoint = np.maximum(endpoint_distances[0], endpoint_distances[1])
        yerr = np.vstack((
            np.maximum(distances_meV - lower_endpoint, 0.0),
            np.maximum(upper_endpoint - distances_meV, 0.0),
        ))

    computed_label = f"Computed ({theta_deg:.1f}°)" if theta_deg is not None else "Computed"
    ax.plot(Vg_vals_meV, distances_meV, "ko-", lw=1.5, markersize=7, label=computed_label, zorder=2)
    if yerr is not None:
        endpoint_specs = TWIST_ENDPOINTS[sample]
        twist_range = f"{endpoint_specs[0][1]:.1f}°–{endpoint_specs[1][1]:.1f}°"
        ax.errorbar(
            Vg_vals_meV,
            distances_meV,
            yerr=yerr,
            fmt="none",
            ecolor="tab:blue",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            label=f"Twist range ({twist_range})",
            zorder=1,
        )
    ax.axhline(arpes_distance_meV, color="red", lw=1.5, ls="--",
               label=f"ARPES = {arpes_distance_meV:.1f} meV")

    ax.set_xlabel(r"$V_G$ (meV)", fontsize=12)
    ax.set_ylabel(r"$\Delta E$ (meV)", fontsize=12)
    theta_label = f", {theta_deg:.1f}°" if theta_deg is not None else ""
    ax.set_title(
        f"TVB \u2013 side band distance at \u0393 ({sample}{theta_label})\n"
        f"(w1p={w1p}, w1d={w1d}, w2p={w2p}, w2d={w2d}, "
        f"\u03d5_G={phiG_deg:.0f}\u00b0)",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    phase_str = f"{phiG_deg:g}".replace(".", "p")
    out_fn = output_dir / f"edc_vs_V_{sample}_phiG{phase_str}.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")


if __name__ == "__main__":
    main()
