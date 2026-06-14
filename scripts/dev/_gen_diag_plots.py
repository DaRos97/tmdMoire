"""Generate diag_* band-line plots for all existing diag.npz folders.

Usage:
    source ../PyEnv/bin/activate
    python scripts/dev/_gen_diag_plots.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tmdmoire.plotting.bilayer import plot_diag_half_bands, plot_diag_bands_over_arpes

DATA_ROOT = Path("Data") / "plot_bilayer_moire"
ARPES_KGK = Path("Inputs") / "plot_bilayer" / "i06_sum_S11_KGK_BE.txt"
ARPES_KK = Path("Inputs") / "plot_bilayer" / "S11_KK_80eV_LV_BE.txt"
ARPES_E_STEP = 0.005
ARPES_KGK_E_MIN = -3.51
ARPES_KGK_K_MIN = -1.4526
ARPES_KGK_K_STEP = 0.00328294
ARPES_KK_E_MIN = -3.47
ARPES_KK_K_MIN = -1.43687
ARPES_KK_K_STEP = 0.00328303


def load_arpes():
    igt = np.loadtxt(ARPES_KGK)
    igt_kk = np.loadtxt(ARPES_KK)
    n_k_kgk, n_e_kgk = igt.shape
    n_k_kk, n_e_kk = igt_kk.shape
    k_kgk = np.linspace(ARPES_KGK_K_MIN, ARPES_KGK_K_MIN + (n_k_kgk - 1) * ARPES_KGK_K_STEP, n_k_kgk)
    k_kgk -= k_kgk[np.argmin(np.abs(k_kgk))]
    e_kgk = np.linspace(ARPES_KGK_E_MIN, ARPES_KGK_E_MIN + (n_e_kgk - 1) * ARPES_E_STEP, n_e_kgk)
    k_kk = np.linspace(ARPES_KK_K_MIN, ARPES_KK_K_MIN + (n_k_kk - 1) * ARPES_KK_K_STEP, n_k_kk)
    k_kk -= k_kk[np.argmin(np.abs(k_kk))]
    e_kk = np.linspace(ARPES_KK_E_MIN, ARPES_KK_E_MIN + (n_e_kk - 1) * ARPES_E_STEP, n_e_kk)
    return k_kgk, e_kgk, igt, k_kk, e_kk, igt_kk


def main():
    print("Loading ARPES intensity data")
    k_kgk_arp, e_kgk_arp, int_kgk, k_kk_arp, e_kk_arp, int_kk = load_arpes()

    for diag_dir in sorted(DATA_ROOT.glob("diag_*")):
        diag_file = diag_dir / "diag.npz"
        if not diag_file.exists():
            continue

        # Check if plots already exist
        if (diag_dir / "diag_half_bands.png").exists() and (diag_dir / "diag_bands_over_arpes.png").exists():
            print(f"Skipping {diag_dir.name} (plots already exist)")
            continue

        print(f"Processing {diag_dir.name}")
        data = np.load(diag_file, allow_pickle=True)
        evals_kgk = data["evals_kgk"]
        evals_kmkp = data["evals_kmkp"]
        norm_kgk = data["norm_kgk"]
        norm_kmkp = data["norm_kmkp"]

        print(f"  evals_kgk: {evals_kgk.shape}, evals_kmkp: {evals_kmkp.shape}")

        plot_diag_half_bands(
            norm_kgk, norm_kmkp, evals_kgk, evals_kmkp,
            k_kgk_arp, k_kk_arp, e_kgk_arp,
            int_kgk, int_kk,
            save_dir=diag_dir
        )

        plot_diag_bands_over_arpes(
            norm_kgk, norm_kmkp, evals_kgk, evals_kmkp,
            k_kgk_arp, k_kk_arp, e_kgk_arp,
            int_kgk, int_kk,
            save_dir=diag_dir
        )

    print("Done")


if __name__ == "__main__":
    main()
