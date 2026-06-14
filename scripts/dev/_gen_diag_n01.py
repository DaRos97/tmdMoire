"""Compute diag.npz + band-line plots for n_shells=0,1 from existing n_shells=2 params.

For each existing diag_* folder in Data/plot_bilayer_moire/, parse the parameters
from the folder name and recompute diagonalization for n_shells=0 and n_shells=1,
keeping all other parameters identical. Saves diag.npz and generates
diag_half_bands.png / diag_bands_over_arpes.png in new diag_* folders.

Usage:
    source ../PyEnv/bin/activate
    python scripts/dev/_gen_diag_n01.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.plotting.bilayer import plot_diag_half_bands, plot_diag_bands_over_arpes
from tmdmoire.utils.kpoints import get_k_list
from tmdmoire.constants import ENERGY_OFFSETS

DATA_ROOT = Path("Data") / "plot_bilayer_moire"
INPUT_DIR = Path("Inputs") / "plot_bilayer"
BAND_LO = 18
BAND_HI = 28

ARPES_KGK = INPUT_DIR / "i06_sum_S11_KGK_BE.txt"
ARPES_KK = INPUT_DIR / "S11_KK_80eV_LV_BE.txt"
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


def parse_diag_dir(name):
    """Parse diag_k{kpts}_n{nshells}_{w1p}_{w1d}_{w2p}_{w2d}_{Vg_meV}_{phiG_deg}_{Vk_meV}_{phiK_deg}"""
    parts = name.split("_")
    # parts[0]="diag", parts[1]="k300", parts[2]="n2", parts[3]="-1.3532", ...
    k_pts = int(parts[1][1:])   # k300 -> 300
    n_shells = int(parts[2][1:])  # n2 -> 2
    w1p = float(parts[3])
    w1d = float(parts[4])
    w2p = float(parts[5])
    w2d = float(parts[6])
    Vg_meV = float(parts[7])
    phiG_deg = float(parts[8])
    Vk_meV = float(parts[9])
    phiK_deg = float(parts[10])
    return k_pts, n_shells, w1p, w1d, w2p, w2d, Vg_meV, phiG_deg, Vk_meV, phiK_deg


def compute_diag_dir_name(k_pts, n_shells, interlayer, moire):
    Vg_meV = moire["Vg"] * 1000
    Vk_meV = moire["Vk"] * 1000
    phiG_deg = moire["phiG"] * 180 / np.pi
    phiK_deg = moire["phiK"] * 180 / np.pi
    return (f"diag_k{k_pts}_n{n_shells}_"
            f"{interlayer['w1p']:.4f}_{interlayer['w1d']:.4f}_{interlayer['w2p']:.4f}_{interlayer['w2d']:.4f}_"
            f"{Vg_meV:.1f}_{phiG_deg:.1f}_{Vk_meV:.1f}_{phiK_deg:.1f}")


def compute_bands_and_plots(wse2, ws2, interlayer, moire, k_pts, n_shells, diag_dir, arpes_data):
    """Diagonalize for given parameters, save diag.npz, generate band-line plots."""
    k_kgk_arp, e_kgk_arp, int_kgk, k_kk_arp, e_kk_arp, int_kk = arpes_data

    geometry = MoireGeometry(2.8)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(n_shells)
    pars_v = (moire["Vg"], moire["Vk"], moire["phiG"], moire["phiK"])

    # For n_shells=0, turn off moire potential
    if n_shells == 0:
        pars_v = (0.0, 0.0, 0.0, 0.0)

    print(f"    Diagonalizing (n_shells={n_shells}, n_cells={n_cells})...")
    k_list_gkm, norm_gkm = get_k_list("G-K-M", k_pts, tmd="WSe2", return_norm=True)

    band_start = BAND_LO * n_cells
    band_end = BAND_HI * n_cells

    evals_gkm_full, evecs_gkm_full = moire_ham.diagonalize(
        k_list_gkm, n_shells, interlayer, pars_v
    )
    evals_gkm = evals_gkm_full[:, band_start:band_end]
    evecs_gkm = evecs_gkm_full[:, :, band_start:band_end]

    energy_offset = ENERGY_OFFSETS.get("S11", 0.0)
    evals_gkm += energy_offset

    n_gkm = len(norm_gkm)
    k_rev = -k_list_gkm[::-1]
    norm_rev = -norm_gkm[::-1]
    evals_rev = evals_gkm[::-1]
    evecs_rev = evecs_gkm[::-1]

    k_list_kgk = np.concatenate([k_rev[:-1], k_list_gkm])
    norm_kgk = np.concatenate([norm_rev[:-1], norm_gkm])
    evals_kgk = np.concatenate([evals_rev[:-1], evals_gkm])
    evecs_kgk = np.concatenate([evecs_rev[:-1], evecs_gkm])
    norm_kgk -= norm_kgk[n_gkm - 1]

    k_m = k_list_gkm[-1]
    k_reflected = np.array([2 * k_m - k for k in k_list_gkm[::-1]])
    evals_reflected = evals_gkm[::-1]
    evecs_reflected = evecs_gkm[::-1]

    k_list_kmkp = np.concatenate([k_list_gkm, k_reflected[1:]])
    evals_kmkp = np.concatenate([evals_gkm, evals_reflected[1:]])
    evecs_kmkp = np.concatenate([evecs_gkm, evecs_reflected[1:]])

    norm_kmkp = np.zeros(len(k_list_kmkp))
    for i in range(1, len(k_list_kmkp)):
        norm_kmkp[i] = norm_kmkp[i - 1] + np.linalg.norm(k_list_kmkp[i] - k_list_kmkp[i - 1])
    norm_kmkp -= norm_kmkp[n_gkm - 1]

    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_file = diag_dir / "diag.npz"
    print(f"    Saving {diag_file}")
    np.savez(
        diag_file,
        evals_kgk=evals_kgk,
        evecs_kgk=evecs_kgk,
        evals_kmkp=evals_kmkp,
        evecs_kmkp=evecs_kmkp,
        norm_kgk=norm_kgk,
        norm_kmkp=norm_kmkp,
        k_list_kgk=k_list_kgk,
        k_list_kmkp=k_list_kmkp,
    )

    print("    Plotting band-line plots")
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


def main():
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")
    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    print("Loading ARPES intensity data")
    arpes_data = load_arpes()

    existing = sorted(DATA_ROOT.glob("diag_*"))
    print(f"Found {len(existing)} existing diag folder(s):")
    for d in existing:
        print(f"  {d.name}")

    for src_dir in existing:
        k_pts, n_shells_src, w1p, w1d, w2p, w2d, Vg_meV, phiG_deg, Vk_meV, phiK_deg = parse_diag_dir(src_dir.name)

        interlayer = {"w1p": w1p, "w1d": w1d, "w2p": w2p, "w2d": w2d}
        moire = {
            "Vg": Vg_meV / 1000,
            "Vk": Vk_meV / 1000,
            "phiG": phiG_deg * np.pi / 180,
            "phiK": phiK_deg * np.pi / 180,
        }

        print(f"\n{'='*70}")
        print(f"Source: {src_dir.name}")
        print(f"  interlayer: w1p={w1p:.4f} w1d={w1d:.4f} w2p={w2p:.4f} w2d={w2d:.4f}")
        print(f"  moire: Vg={Vg_meV:.1f} meV phiG={phiG_deg:.1f}deg Vk={Vk_meV:.1f} meV phiK={phiK_deg:.1f}deg")
        print(f"  k_pts={k_pts}")

        for n_shells in [0, 1]:
            diag_name = compute_diag_dir_name(k_pts, n_shells, interlayer, moire)
            diag_dir = DATA_ROOT / diag_name
            if diag_dir.exists():
                print(f"  Skipping n_shells={n_shells} ({diag_name}) — already exists")
                continue
            print(f"  Computing n_shells={n_shells} -> {diag_name}")
            compute_bands_and_plots(wse2, ws2, interlayer, moire, k_pts, n_shells, diag_dir, arpes_data)

    print("\nDone")


if __name__ == "__main__":
    main()
