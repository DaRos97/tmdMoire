"""Standalone EDC intensity profile plot for a selected (Vg, phiG) cell.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py (with --vg/--phig)
or scripts/export_moire_bands.py (profiles at each exported V_G value).

Produces:
  edc_profile_4L_<run_id>.png  -- one selected EDC profile from an EDC-grid export
  edc_profiles_<sample>_gamma_phiG<phase>.png -- exported profiles at all V_G values

Usage:
    python plot_edc_profile.py <data.npz>
    python plot_edc_profile.py data.npz --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_edc_profile(data_path, output_dir=None):
    data_path = Path(data_path)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "figures"
    else:
        output_dir = Path(output_dir)

    d = np.load(data_path, allow_pickle=True)

    if "energy_list_0" in d:
        return _plot_gamma_edc_profiles(d, output_dir)

    if "energy_list" not in d:
        print("Error: .npz does not contain EDC profile data.")
        print("Export a selected Gamma EDC profile with:")
        print("  python scripts/export_edc_gamma_data.py --id <id> --vg <Vg> --phig <phiG>")
        print("Or use the moire-band export, which includes profiles for all V_G values:")
        print("  python scripts/export_moire_bands.py")
        sys.exit(1)

    energy_list = d["energy_list"]
    weight_list = d["weight_list"]
    fit_4L_curve = d["fit_4L_curve"]
    fit_4L_centers = d["fit_4L_centers"]
    fit_4L_redchi = float(d["fit_4L_redchi"])
    exp_positions = d["exp_positions_ev"]
    sel_Vg_meV = float(d["selected_Vg_meV"])
    sel_phiG_deg = float(d["selected_phiG_deg"])
    run_id = str(d["run_id"])
    sel_w1p = float(d["selected_w1p_ev"])
    sel_w1d = float(d["selected_w1d_ev"])
    sel_w2p = float(d["selected_w2p_ev"])
    sel_w2d = float(d["selected_w2d_ev"])

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")

    if not np.all(np.isnan(fit_4L_curve)):
        ax.plot(energy_list, fit_4L_curve, "r--", lw=2,
                label=rf"4-Lorentzian fit ($\chi^2_\nu$ = {fit_4L_redchi:.4f})")

    exp_label = "ARPES EDC position"
    for i_e, e_val in enumerate(exp_positions):
        lbl = exp_label if i_e == 0 else ""
        ax.axvline(x=e_val, color="#2ecc71", ls="--", lw=1.5, alpha=0.8, label=lbl)

    center_label = "4-Lor. center"
    for i_c, c_val in enumerate(fit_4L_centers):
        if np.isnan(c_val):
            continue
        lbl = center_label if i_c == 0 else ""
        ax.axvline(x=c_val, color="#e74c3c", ls=":", lw=1.5, alpha=0.8, label=lbl)

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.set_title(
        f"EDC at Gamma: Vg={sel_Vg_meV:.1f} meV, phiG={sel_phiG_deg:.0f} deg\n"
        f"w1p={sel_w1p:.3f}, w1d={sel_w1d:.3f}, w2p={sel_w2p:.3f}, w2d={sel_w2d:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    out_fn = output_dir / f"edc_profile_4L_{run_id}.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")
    return out_fn


def _plot_gamma_edc_profiles(data, output_dir):
    vg_labels = data["Vg_labels"]
    n_profiles = len(vg_labels)
    required_keys = [
        f"{key}_{i_profile}"
        for i_profile in range(n_profiles)
        for key in ("energy_list", "weight_list", "fit_4L_curve", "fit_4L_centers", "fit_4L_redchi")
    ]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"EDC profile data is incomplete; missing: {missing_keys}")

    exp_positions = data["exp_positions_ev"]
    phiG_deg = float(data["phiG_deg"])
    sample = str(data["sample"])
    interlayer = {
        "w1p": float(data["interlayer_w1p"]),
        "w1d": float(data["interlayer_w1d"]),
        "w2p": float(data["interlayer_w2p"]),
        "w2d": float(data["interlayer_w2d"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        n_profiles,
        figsize=(6.5 * n_profiles, 6),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for i_profile, (ax, vg_label) in enumerate(zip(axes, vg_labels)):
        energy_list = data[f"energy_list_{i_profile}"]
        weight_list = data[f"weight_list_{i_profile}"]
        fit_4l_curve = data[f"fit_4L_curve_{i_profile}"]
        fit_4l_centers = data[f"fit_4L_centers_{i_profile}"]
        fit_4l_redchi = float(data[f"fit_4L_redchi_{i_profile}"])

        ax.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")
        if not np.all(np.isnan(fit_4l_curve)):
            ax.plot(
                energy_list,
                fit_4l_curve,
                "r--",
                lw=2,
                label=rf"4-Lorentzian fit ($\chi^2_\nu$ = {fit_4l_redchi:.4f})",
            )

        for i_energy, energy in enumerate(exp_positions):
            ax.axvline(
                x=energy,
                color="#2ecc71",
                ls="--",
                lw=1.5,
                alpha=0.8,
                label="ARPES EDC position" if i_energy == 0 else "",
            )

        for i_center, center in enumerate(fit_4l_centers):
            if np.isnan(center):
                continue
            ax.axvline(
                x=center,
                color="#e74c3c",
                ls=":",
                lw=1.5,
                alpha=0.8,
                label="4-Lor. center" if i_center == 0 else "",
            )

        ax.set_xlabel("Energy (eV)", fontsize=12)
        ax.set_title(rf"$V_G = {vg_label}$", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Intensity (a.u.)", fontsize=12)
    fig.suptitle(
        f"Gamma EDC profiles (sample={sample}, phiG={phiG_deg:.0f} deg)\n"
        f"w1p={interlayer['w1p']:.3f}, w1d={interlayer['w1d']:.3f}, "
        f"w2p={interlayer['w2p']:.4f}, w2d={interlayer['w2d']:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    phase_str = f"{phiG_deg:g}".replace(".", "p")
    out_fn = output_dir / f"edc_profiles_{sample}_gamma_phiG{phase_str}.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")
    return out_fn


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_edc_profile.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = None

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    plot_edc_profile(data_path, output_dir)


if __name__ == "__main__":
    main()
