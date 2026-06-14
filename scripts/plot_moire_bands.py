"""Plot moire bilayer bands with intensity spreading.

Computes the full supercell Hamiltonian along a single G->K->M path,
mirrors it to produce K'->G->K and K->M->K' plots via reverse+attach,
and generates ARPES-like intensity heatmaps with Gaussian/Lorentzian spreading.

Usage:
    source ../PyEnv/bin/activate
    python scripts/plot_moire_bands.py [--k-pts 300] [--n-shells 2] ...

All parameters are loaded from Inputs/plot_bilayer/:
    tb_WSe2.npy, tb_WS2.npy, interlayer_G.json, interlayer_K.json

Outputs:
    Data/plot_bilayer_moire/diag_k<k-pts>_n<n-shells>/diag.npz
    Data/plot_bilayer_moire/diag_k<k-pts>_n<n-shells>/intensity_<type>_<sk>_<se>_<pow>_<shade_ws2>_<shade_e>/spread.npz
    Data/plot_bilayer_moire/diag_k<k-pts>_n<n-shells>/intensity_<...>/*.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tmdmoire.material import TMDMaterial
from tmdmoire.bilayer.geometry import MoireGeometry
from tmdmoire.bilayer.hamiltonian import MoireHamiltonian
from tmdmoire.bilayer.intensity import compute_weights, spread_intensity
from tmdmoire.utils.kpoints import get_k_list
from tmdmoire.plotting.bilayer import (
    plot_moire_bands_simulated,
    plot_arpes_data,
    plot_moire_bands_half_arpes,
    plot_moire_bands_simulated_with_arpes,
    plot_diag_half_bands,
    plot_diag_bands_over_arpes,
)
from tmdmoire.constants import ENERGY_OFFSETS


CACHE_ROOT = Path("Data") / "plot_bilayer_moire"
INPUT_DIR = Path("Inputs") / "plot_bilayer"

ARPES_KGK_FILE = INPUT_DIR / "i06_sum_S11_KGK_BE.txt"
ARPES_KK_FILE = INPUT_DIR / "S11_KK_80eV_LV_BE.txt"

ARPES_KGK_E_MIN = -3.51
ARPES_KGK_K_MIN = -1.4526
ARPES_KGK_K_STEP = 0.00328294

ARPES_KK_E_MIN = -3.47
ARPES_KK_K_MIN = -1.43687
ARPES_KK_K_STEP = 0.00328303

ARPES_E_STEP = 0.005

BAND_LO = 18
BAND_HI = 28


def parse_args():
    parser = argparse.ArgumentParser(description="Plot moire bilayer bands with intensity")
    parser.add_argument("--k-pts", type=int, default=300, help="Number of points along G->K->M path")
    parser.add_argument("--n-shells", type=int, default=2, help="Number of moire shells")
    parser.add_argument("--spread-type", choices=["Gauss", "Lorentz"], default="Gauss")
    parser.add_argument("--spread-k", type=float, default=0.005, help="k spreading width (A^-1)")
    parser.add_argument("--spread-e", type=float, default=0.015, help="Energy spreading width (eV)")
    parser.add_argument("--pow-factor", type=float, default=2.0, help="Eigenvector exponent")
    parser.add_argument("--shade-ws2", type=float, default=0.1, help="WS2 shading factor")
    parser.add_argument("--shade-e-factor", type=float, default=3.0, help="Energy shading factor at E_max")
    parser.add_argument("--e-min", type=float, default=-3.5, help="Minimum energy (eV)")
    parser.add_argument("--e-max", type=float, default=0.0, help="Maximum energy (eV)")
    parser.add_argument("--delta-e", type=float, default=0.01, help="Energy grid spacing (eV)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and recompute")
    parser.add_argument("--sample", default="S11", help="Sample name for energy offset")
    parser.add_argument("--theta", type=float, default=None, help="Twist angle (deg, overrides sample)")
    parser.add_argument("--Vg", type=float, default=None, help="Override moire potential at Gamma (eV)")
    parser.add_argument("--Vk", type=float, default=None, help="Override moire potential at K (eV)")
    return parser.parse_args()


def load_params():
    """Load all parameters from Inputs/plot_bilayer/."""
    tb_wse2 = np.load(INPUT_DIR / "tb_WSe2.npy")
    tb_ws2 = np.load(INPUT_DIR / "tb_WS2.npy")
    with open(INPUT_DIR / "interlayer_G.json") as f:
        interlayer_g = json.load(f)
    with open(INPUT_DIR / "interlayer_K.json") as f:
        interlayer_k = json.load(f)

    interlayer = {
        "w1p": interlayer_g["w1p"],
        "w1d": interlayer_g["w1d"],
        "w2p": interlayer_g["w2p"],
        "w2d": interlayer_g["w2d"],
    }
    moire = {
        "Vg": interlayer_g["Vg"],
        "phiG": interlayer_g["phiG_deg"] * np.pi / 180,
        "Vk": interlayer_k["Vk"],
        "phiK": interlayer_k["phiK_deg"] * np.pi / 180,
    }
    return tb_wse2, tb_ws2, interlayer, moire


def compute_diag_dir_name(n_shells, k_pts, interlayer, moire):
    """Human-readable directory name for diagonalization cache."""
    Vg_meV = moire["Vg"] * 1000
    Vk_meV = moire["Vk"] * 1000
    phiG_deg = moire["phiG"] * 180 / np.pi
    phiK_deg = moire["phiK"] * 180 / np.pi
    return (f"diag_k{k_pts}_n{n_shells}_"
            f"{interlayer['w1p']:.4f}_{interlayer['w1d']:.4f}_{interlayer['w2p']:.4f}_{interlayer['w2d']:.4f}_"
            f"{Vg_meV:.1f}_{phiG_deg:.1f}_{Vk_meV:.1f}_{phiK_deg:.1f}")


def save_metadata(diag_dir, k_pts, n_shells, n_cells, interlayer, moire, theta, sample):
    """Save metadata.json with all run parameters."""
    meta = {
        "k_pts": k_pts,
        "n_shells": n_shells,
        "n_cells": n_cells,
        "theta_deg": theta,
        "sample": sample,
        "interlayer": {
            "w1p": interlayer["w1p"],
            "w1d": interlayer["w1d"],
            "w2p": interlayer["w2p"],
            "w2d": interlayer["w2d"],
        },
        "moire": {
            "Vg_ev": moire["Vg"],
            "Vg_meV": moire["Vg"] * 1000,
            "phiG_deg": moire["phiG"] * 180 / np.pi,
            "phiG_rad": moire["phiG"],
            "Vk_ev": moire["Vk"],
            "Vk_meV": moire["Vk"] * 1000,
            "phiK_deg": moire["phiK"] * 180 / np.pi,
            "phiK_rad": moire["phiK"],
        },
    }
    meta_fn = diag_dir / "metadata.json"
    with open(meta_fn, "w") as f:
        json.dump(meta, f, indent=2)


def compute_intensity_dir_name(spread_type, spread_k, spread_e, pow_factor,
                                shade_ws2, shade_e_factor):
    """Human-readable directory name for intensity spreading cache."""
    type_prefix = "G" if spread_type == "Gauss" else "L"
    return (f"intensity_{type_prefix}_{spread_k:.4f}_{spread_e:.4f}_{pow_factor:.2f}"
            f"_{shade_ws2:.2f}_{shade_e_factor:.2f}")


def load_arpes_intensity():
    """Load ARPES intensity grids from .txt files."""
    intensity_kgk = np.loadtxt(ARPES_KGK_FILE)
    intensity_kk = np.loadtxt(ARPES_KK_FILE)

    n_k_kgk, n_e_kgk = intensity_kgk.shape
    n_k_kk, n_e_kk = intensity_kk.shape

    k_kgk = np.linspace(
        ARPES_KGK_K_MIN,
        ARPES_KGK_K_MIN + (n_k_kgk - 1) * ARPES_KGK_K_STEP,
        n_k_kgk
    )
    k_kgk -= k_kgk[np.argmin(np.abs(k_kgk))]

    e_kgk = np.linspace(
        ARPES_KGK_E_MIN,
        ARPES_KGK_E_MIN + (n_e_kgk - 1) * ARPES_E_STEP,
        n_e_kgk
    )

    k_kk = np.linspace(
        ARPES_KK_K_MIN,
        ARPES_KK_K_MIN + (n_k_kk - 1) * ARPES_KK_K_STEP,
        n_k_kk
    )
    k_kk -= k_kk[np.argmin(np.abs(k_kk))]

    e_kk = np.linspace(
        ARPES_KK_E_MIN,
        ARPES_KK_E_MIN + (n_e_kk - 1) * ARPES_E_STEP,
        n_e_kk
    )

    return k_kgk, e_kgk, intensity_kgk, k_kk, e_kk, intensity_kk


def main():
    args = parse_args()

    print("Loading parameters from Inputs/plot_bilayer/")
    tb_wse2, tb_ws2, interlayer, moire = load_params()

    if args.Vg is not None:
        moire["Vg"] = args.Vg
    if args.Vk is not None:
        moire["Vk"] = args.Vk

    wse2 = TMDMaterial("WSe2", params=tb_wse2)
    ws2 = TMDMaterial("WS2", params=tb_ws2)

    theta = args.theta
    if theta is None:
        from tmdmoire.constants import TWIST_ANGLES
        theta = TWIST_ANGLES.get(args.sample, 2.8)

    geometry = MoireGeometry(theta)
    moire_ham = MoireHamiltonian(wse2, ws2, geometry)

    n_cells = MoireGeometry.n_cells(args.n_shells)
    pars_v = (moire["Vg"], moire["Vk"], moire["phiG"], moire["phiK"])

    e_list = np.linspace(args.e_min, args.e_max, int((args.e_max - args.e_min) / args.delta_e))

    diag_dir_name = compute_diag_dir_name(args.n_shells, args.k_pts, interlayer, moire)
    intensity_dir_name = compute_intensity_dir_name(
        args.spread_type, args.spread_k, args.spread_e, args.pow_factor,
        args.shade_ws2, args.shade_e_factor
    )

    diag_dir = CACHE_ROOT / diag_dir_name
    diag_file = diag_dir / "diag.npz"
    intensity_dir = diag_dir / intensity_dir_name
    spread_file = intensity_dir / "spread.npz"

    diag_cached = diag_file.exists() and not args.no_cache
    spread_cached = spread_file.exists() and not args.no_cache

    if diag_cached:
        print(f"Loading diagonalization from {diag_dir_name}")
        d = np.load(diag_file, allow_pickle=True)
        evals_kgk = d["evals_kgk"]
        evecs_kgk = d["evecs_kgk"]
        evals_kmkp = d["evals_kmkp"]
        evecs_kmkp = d["evecs_kmkp"]
        norm_kgk = d["norm_kgk"]
        norm_kmkp = d["norm_kmkp"]
        norm_kgk_mono = d["norm_kgk_mono"]
        k_list_kgk = d["k_list_kgk"]
        k_list_kmkp = d["k_list_kmkp"]
        save_metadata(diag_dir, args.k_pts, args.n_shells, n_cells, interlayer, moire, theta, args.sample)
    else:
        print(f"Building G->K->M path (k_pts={args.k_pts})")
        k_list_gkm, norm_gkm = get_k_list("G-K-M", args.k_pts, tmd="WSe2", return_norm=True)

        band_start = BAND_LO * n_cells
        band_end = BAND_HI * n_cells

        print("Diagonalizing G->K->M path (full Hamiltonian)")
        evals_gkm_full, evecs_gkm_full = moire_ham.diagonalize(
            k_list_gkm, args.n_shells, interlayer, pars_v
        )
        evals_gkm = evals_gkm_full[:, band_start:band_end]
        evecs_gkm = evecs_gkm_full[:, :, band_start:band_end]

        energy_offset = ENERGY_OFFSETS.get(args.sample, 0.0)
        evals_gkm += energy_offset

        # Gamma-centered plot: reverse GKM, prepend to GKM -> MKGKM, center at G
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

        # M-centered plot: reflect GKM about M, append to GKM -> GKM K'G', center at M
        k_m = k_list_gkm[-1]
        k_reflected = np.array([2 * k_m - k for k in k_list_gkm[::-1]])
        evals_reflected = evals_gkm[::-1]
        evecs_reflected = evecs_gkm[::-1]

        k_list_kmkp = np.concatenate([k_list_gkm, k_reflected[1:]])
        evals_kmkp = np.concatenate([evals_gkm, evals_reflected[1:]])
        evecs_kmkp = np.concatenate([evecs_gkm, evecs_reflected[1:]])

        # Cumulative distance along mirrored path, centered at M
        norm_kmkp = np.zeros(len(k_list_kmkp))
        for i in range(1, len(k_list_kmkp)):
            norm_kmkp[i] = norm_kmkp[i - 1] + np.linalg.norm(k_list_kmkp[i] - k_list_kmkp[i - 1])
        norm_kmkp -= norm_kmkp[n_gkm - 1]

        # Monotonic cumulative distance for M plot resampling (same as norm_kmkp now)
        norm_kgk_mono = np.zeros(len(k_list_kgk))
        for i in range(1, len(k_list_kgk)):
            norm_kgk_mono[i] = norm_kgk_mono[i - 1] + np.linalg.norm(k_list_kgk[i] - k_list_kgk[i - 1])

        diag_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving diagonalization to {diag_file}")
        np.savez(
            diag_file,
            evals_kgk=evals_kgk,
            evecs_kgk=evecs_kgk,
            evals_kmkp=evals_kmkp,
            evecs_kmkp=evecs_kmkp,
            norm_kgk=norm_kgk,
            norm_kmkp=norm_kmkp,
            norm_kgk_mono=norm_kgk_mono,
            k_list_kgk=k_list_kgk,
            k_list_kmkp=k_list_kmkp,
        )
        save_metadata(diag_dir, args.k_pts, args.n_shells, n_cells, interlayer, moire, theta, args.sample)

    print("Loading ARPES intensity data for band-line plots")
    k_kgk_arpes, e_kgk_arpes, intensity_kgk, k_kk_arpes, e_kk_arpes, intensity_kk = load_arpes_intensity()

    print("Plotting half-ARPES / half-bands split")
    plot_diag_half_bands(
        norm_kgk, norm_kmkp, evals_kgk, evals_kmkp,
        k_kgk_arpes, k_kk_arpes, e_kgk_arpes,
        intensity_kgk, intensity_kk,
        save_dir=diag_dir
    )

    print("Plotting bands over ARPES background")
    plot_diag_bands_over_arpes(
        norm_kgk, norm_kmkp, evals_kgk, evals_kmkp,
        k_kgk_arpes, k_kk_arpes, e_kgk_arpes,
        intensity_kgk, intensity_kk,
        save_dir=diag_dir
    )

    if args.n_shells == 0:
        print("Exporting top 8 valence bands to txt")
        band_cols_top8 = list(range(-1, -9, -1))
        header = "k_Angstrom\tBand27_eV\tBand26_eV\tBand25_eV\tBand24_eV\tBand23_eV\tBand22_eV\tBand21_eV\tBand20_eV"
        out_kgk = np.column_stack([norm_kgk] + [evals_kgk[:, i] for i in band_cols_top8])
        np.savetxt(diag_dir / "bands_KpGK.txt", out_kgk, fmt="%.8f", delimiter="\t",
                    header=header, comments="")
        print(f"  {diag_dir / 'bands_KpGK.txt'}")
        out_kmkp = np.column_stack([norm_kmkp] + [evals_kmkp[:, i] for i in band_cols_top8])
        np.savetxt(diag_dir / "bands_KpMK.txt", out_kmkp, fmt="%.8f", delimiter="\t",
                    header=header, comments="")
        print(f"  {diag_dir / 'bands_KpMK.txt'}")

    if spread_cached:
        print(f"Loading spread intensity from {intensity_dir_name}")
        s = np.load(spread_file, allow_pickle=True)
        spread_kgk = s["spread_kgk"]
        spread_kmkp = s["spread_kmkp"]
    else:
        print(f"Computing weights (pow_factor={args.pow_factor}, shade_ws2={args.shade_ws2})")
        weights_kgk = compute_weights(evecs_kgk, n_cells, args.pow_factor, args.shade_ws2)
        weights_kmkp = compute_weights(evecs_kmkp, n_cells, args.pow_factor, args.shade_ws2)

        print(f"Spreading intensity ({args.spread_type}, spread_k={args.spread_k}, spread_e={args.spread_e})")
        spread_kgk = spread_intensity(
            weights_kgk, k_list_kgk, evals_kgk, e_list,
            args.spread_k, args.spread_e, args.spread_type
        )
        spread_kmkp = spread_intensity(
            weights_kmkp, k_list_kmkp, evals_kmkp, e_list,
            args.spread_k, args.spread_e, args.spread_type
        )

        print(f"Saving spread intensity to {spread_file}")
        intensity_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            spread_file,
            spread_kgk=spread_kgk,
            spread_kmkp=spread_kmkp,
        )

    print("Exporting intensity to txt (ARPES format)")
    np.savetxt(intensity_dir / "intensity_KpGK.txt", spread_kgk, fmt="%.8f", delimiter="\t")
    print(f"  {intensity_dir / 'intensity_KpGK.txt'}")
    np.savetxt(intensity_dir / "intensity_KpMK.txt", spread_kmkp, fmt="%.8f", delimiter="\t")
    print(f"  {intensity_dir / 'intensity_KpMK.txt'}")

    print("Saving intensity axis metadata")
    import json
    meta_fn = intensity_dir / "intensity_meta.json"
    with open(meta_fn, "w") as f:
        json.dump({
            "k_KpGK": norm_kgk.tolist(),
            "k_KpMK": norm_kmkp.tolist(),
            "e_list": e_list.tolist(),
            "e_min": args.e_min,
            "e_max": args.e_max,
            "delta_e": args.delta_e,
            "k_pts": args.k_pts,
            "n_shells": args.n_shells,
            "spread_type": args.spread_type,
            "spread_k": args.spread_k,
            "spread_e": args.spread_e,
            "pow_factor": args.pow_factor,
            "shade_ws2": args.shade_ws2,
            "shade_e_factor": args.shade_e_factor,
        }, f, indent=2)
    print(f"  {meta_fn}")

    print("Plotting simulated bands")
    plot_moire_bands_simulated(
        norm_kgk, norm_kmkp, e_list, spread_kgk, spread_kmkp,
        shade_factor_e=args.shade_e_factor, save_dir=intensity_dir
    )

    print("Loading bilayer ARPES data for overlay")
    from tmdmoire.bilayer.data import BilayerData
    from tmdmoire.utils.paths import get_repo_root
    repo_root = get_repo_root()
    bilayer_data = BilayerData(repo_root, pts=200)

    print("Plotting simulated bands with ARPES overlay")
    plot_moire_bands_simulated_with_arpes(
        norm_kgk, norm_kmkp, e_list, spread_kgk, spread_kmkp,
        bilayer_data, shade_factor_e=args.shade_e_factor, save_dir=intensity_dir
    )

    print("Plotting ARPES data")
    plot_arpes_data(
        k_kgk_arpes, k_kk_arpes, e_kgk_arpes,
        intensity_kgk, intensity_kk,
        save_dir=intensity_dir
    )

    print("Plotting half-ARPES / half-simulated")
    spread_kgk_resampled = _resample_to_arpes(spread_kgk, norm_kgk, e_list, k_kgk_arpes, e_kgk_arpes)
    spread_kmkp_resampled = _resample_to_arpes(spread_kmkp, norm_kmkp, e_list, k_kk_arpes, e_kk_arpes)

    plot_moire_bands_half_arpes(
        k_kgk_arpes, k_kk_arpes, e_kgk_arpes,
        spread_kgk_resampled, spread_kmkp_resampled,
        intensity_kgk, intensity_kk,
        shade_factor_e=args.shade_e_factor, save_dir=intensity_dir
    )

    print("Done")


def _resample_to_arpes(spread, norm, e_list, k_arpes, e_arpes):
    """Resample simulated intensity to ARPES grid using bilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (norm, e_list), spread,
        method="linear", bounds_error=False, fill_value=0.0
    )
    kk, ee = np.meshgrid(k_arpes, e_arpes, indexing="ij")
    return interp((kk, ee))


if __name__ == "__main__":
    main()
