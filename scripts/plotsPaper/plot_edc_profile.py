"""Standalone EDC intensity profile plot for a selected (Vg, phiG) cell.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_edc_gamma_data.py (with --vg/--phig).

Produces:
  edc_profile_4L.png  -- EDC intensity + 4-Lorentzian fit + ARPES reference lines

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


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python plot_edc_profile.py <data.npz> [--output-dir <dir>]")
        sys.exit(1)

    data_path = Path(args[0])
    output_dir = Path(__file__).resolve().parent / "figures"

    i = 1
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    d = np.load(data_path)

    if "energy_list" not in d:
        print("Error: .npz does not contain EDC profile data.")
        print("Export with --vg/--phig to include it:")
        print("  python scripts/export_edc_gamma_data.py --id <id> --vg <Vg> --phig <phiG>")
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

    fig.savefig(output_dir / f"edc_profile_4L_{run_id}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / f'edc_profile_4L_{run_id}.png'}")


if __name__ == "__main__":
    main()
