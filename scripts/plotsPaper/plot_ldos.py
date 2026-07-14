"""Standalone LDOS plot.

Zero dependency on tmdmoire. Requires only numpy + matplotlib.
Reads the .npz produced by scripts/export_ldos.py.

Produces:
  ldos_<name>.png  --  pcolormesh: energy (x) vs position (y)
                      with stacking site labels (W/W, Se/W, W/S, W/W).

Usage:
    python plot_ldos.py <data.npz>
    python plot_ldos.py data.npz --output-dir ./figures
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    args = sys.argv[1:]
    if not args or args[0].endswith(".py"):
        print("Usage: python plot_ldos.py <data.npz> [--output-dir <dir>]")
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

    d = np.load(data_path, allow_pickle=True)

    ldos = d["ldos"]
    r_list = d["r_list"]
    e_list = d["e_list"]
    rL = float(d["rL"])

    Vg_meV = float(d["Vg"]) * 1000
    phiG_deg = float(d["phiG_deg"])
    n_shells = int(d["n_shells"])
    k_pts = int(d["k_pts"])
    theta = float(d["theta_deg"])
    Vk_meV = float(d["Vk"]) * 1000
    phiK_deg = float(d["phiK_deg"])
    w1p = float(d["interlayer_w1p"])
    w1d = float(d["interlayer_w1d"])
    sample = str(d["sample"])
    eta = float(d["eta"])

    r_norm = np.linalg.norm(r_list, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    s_ = 20

    X, Y = np.meshgrid(e_list, r_norm)
    mesh = ax.pcolormesh(X, Y, ldos, cmap="hot", shading="auto")
    ax.invert_yaxis()

    ax.set_yticks([0, rL / 3, 2 * rL / 3, rL],
                  [r"W/W", r"Se/W", r"W/S", r"W/W"],
                  size=s_)
    ax.set_xlabel("Energy [eV]", size=s_)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(1.3, 0.02, "low", ha="left", va="bottom",
                 transform=cbar.ax.transAxes, fontsize=s_)
    cbar.ax.text(1.3, 0.98, "high", ha="left", va="top",
                 transform=cbar.ax.transAxes, fontsize=s_)

    title = (f"{sample}  {theta}\u00b0  "
             f"V$_G$={Vg_meV:.1f} meV  "
             f"$\\phi_G$={phiG_deg:.0f}\u00b0  "
             f"V$_K$={Vk_meV:.1f} meV  "
             f"$\\phi_K$={phiK_deg:.0f}\u00b0\n"
             f"n$_\\mathrm{{shells}}$={n_shells}  "
             f"k$_\\mathrm{{pts}}$={k_pts}  "
             f"$\\eta$={eta*1000:.0f} meV  "
             f"w1p={w1p:.2f}  w1d={w1d:.2f}")
    ax.set_title(title, size=s_ - 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_fn = output_dir / "ldos.png"
    fig.savefig(out_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fn}")


if __name__ == "__main__":
    main()
