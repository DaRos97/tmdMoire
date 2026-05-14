"""Export top 8 valence bands of bilayer (n_shells=0) along two BZ cuts.

Computes eigenvalues using fitted monolayer + interlayer parameters
and saves to .txt files for:
  - Kp-G-K path (201 k-points)
  - Kp-M-K path (101 k-points)

Output format: tab-separated columns
  Column 0: cumulative |k| distance (Å⁻¹)
  Columns 1-8: top 8 valence band energies (eV), band 27 first, band 20 last

Usage
-----
    python scripts/export_bilayer_bands.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tmdmoire import TMDMaterial, MoireHamiltonian, MoireGeometry, get_repo_root
from tmdmoire.constants import ENERGY_OFFSETS
from tmdmoire.utils.kpoints import get_k_list

master_folder = get_repo_root()

# ── Output directory ─────────────────────────────────────────────────────────
out_dir = os.path.join(master_folder, "Data", "interlayer_fit")
os.makedirs(out_dir, exist_ok=True)

# ── Load monolayer parameters ────────────────────────────────────────────────
monolayer_fns = {
    "WSe2": master_folder + "/Inputs/bilayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/bilayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

# ── Load interlayer coupling parameters ──────────────────────────────────────
interlayer_arr = np.load(master_folder + "/Inputs/bilayer_fitting/interlayer_params.npy")
interlayer_params = {"w1p": interlayer_arr[0], "w1d": interlayer_arr[1],
                     "w2p": interlayer_arr[2], "w2d": interlayer_arr[3]}
pars_V = (0.0, 0.0, 0.0, 0.0)

print(f"Interlayer params: w1p={interlayer_params['w1p']:+.4f}, "
      f"w1d={interlayer_params['w1d']:+.4f}, "
      f"w2p={interlayer_params['w2p']:+.4f}, "
      f"w2d={interlayer_params['w2d']:+.4f} eV")

# ── Build Hamiltonian helper ─────────────────────────────────────────────────
geometry = MoireGeometry(0.0)
ham = MoireHamiltonian(wse2, ws2, geometry)

def compute_bands(k_list):
    """Return (n_kpts, 44) eigenvalues with S11 energy offset applied."""
    mono_hams_wse2 = [wse2.build_hamiltonian(k) for k in k_list]
    mono_hams_ws2 = [ws2.build_hamiltonian(k) for k in k_list]
    evals, _ = ham.diagonalize(k_list, n_shells=0,
                               interlayer_params=interlayer_params,
                               pars_V=pars_V,
                               mono_hams_wse2=mono_hams_wse2,
                               mono_hams_ws2=mono_hams_ws2)
    return evals + ENERGY_OFFSETS["S11"]

def load_arpes_kpgk():
    """Load ARPES data for Kp-G-K path (3 bands)."""
    raw = []
    for ib in range(1, 4):
        fn = os.path.join(master_folder, "Inputs", "bilayer_fitting", f"WSe2WS2_Band{ib}.txt")
        with open(fn) as f:
            lines = f.readlines()
        temp = []
        for line in lines:
            parts = line.split("\t")
            k = float(parts[0])
            e_str = parts[1].strip()
            if e_str == "" or e_str == "NAN":
                temp.append([k, np.nan])
            else:
                temp.append([k, float(e_str)])
        raw.append(np.array(temp))

    sym = []
    for ib in range(3):
        rd = raw[ib]
        neg_mask = rd[:, 0] < 0
        pos_mask = rd[:, 0] > 0
        neg = rd[neg_mask].copy()
        pos = rd[pos_mask].copy()
        neg[:, 0] = np.abs(neg[:, 0])
        neg = neg[::-1]
        nk_neg = neg.shape[0]
        nk_pos = pos.shape[0]
        nk = min(nk_neg, nk_pos)
        k_neg = neg[:nk, 0]
        k_pos = pos[:nk, 0]
        e_neg = neg[:nk, 1]
        e_pos = pos[:nk, 1]
        mask_neg = ~np.isnan(e_neg)
        mask_pos = ~np.isnan(e_pos)
        mask_both = mask_neg & mask_pos
        mask_only_neg = mask_neg & ~mask_pos
        mask_only_pos = ~mask_neg & mask_pos
        k_out = k_neg.copy()
        e_out = np.full(nk, np.nan)
        e_out[mask_both] = (e_neg[mask_both] + e_pos[mask_both]) / 2
        e_out[mask_only_neg] = e_neg[mask_only_neg]
        e_out[mask_only_pos] = e_pos[mask_only_pos]
        valid = ~np.isnan(e_out)
        result = np.column_stack([k_out[valid], e_out[valid]])
        extra_neg = neg[nk:]
        extra_pos = pos[nk:]
        if len(extra_neg) > 0:
            v = ~np.isnan(extra_neg[:, 1])
            result = np.vstack([result, extra_neg[v]])
        if len(extra_pos) > 0:
            v = ~np.isnan(extra_pos[:, 1])
            result = np.vstack([result, extra_pos[v]])
        result = result[result[:, 0].argsort()]
        sym.append(result)

    k_list = []
    e_list = [[], [], []]
    for ib in range(3):
        for row in sym[ib]:
            if row[0] not in k_list:
                k_list.append(row[0])
    k_list = sorted(k_list)
    for ib in range(3):
        k_band = sym[ib][:, 0]
        e_band = sym[ib][:, 1]
        energies = []
        for k in k_list:
            idx = np.where(np.abs(k_band - k) < 1e-8)[0]
            if len(idx) > 0:
                energies.append(e_band[idx[0]])
            else:
                energies.append(np.nan)
        e_list[ib] = energies
    return np.array(k_list), e_list

def plot_bands(norm_k, evals_top8, arpes_data=None, path_label="Kp-G-K", save_fn=None):
    """Plot top 8 valence bands with optional ARPES overlay."""
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    colors_bands = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    band_labels = [f"Band {27-i}" for i in range(8)]

    for i in range(8):
        ax.plot(norm_k, evals_top8[:, i], color=colors_bands[i],
                lw=2, label=band_labels[i])

    if arpes_data is not None:
        arpes_k, arpes_energies = arpes_data
        colors_arpes = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for ib in range(3):
            e_arr = np.array(arpes_energies[ib])
            valid = ~np.isnan(e_arr)
            ax.scatter(arpes_k[valid], e_arr[valid], s=15, c=colors_arpes[ib],
                       alpha=0.6, zorder=3, edgecolors='none',
                       label=f"ARPES Band {ib+1}" if ib < 3 else "")

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"$|k|$ (Å⁻¹)", fontsize=13)
    ax.set_ylabel("Energy (eV)", fontsize=13)
    ax.set_title(f"Bilayer top 8 valence bands — {path_label}", fontsize=14, fontweight="bold")

    param_text = (f"w1p={interlayer_params['w1p']:+.3f}  "
                  f"w1d={interlayer_params['w1d']:+.3f}  "
                  f"w2p={interlayer_params['w2p']:+.3f}  "
                  f"w2d={interlayer_params['w2d']:+.3f}")
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.legend(fontsize=9, loc="lower right", ncol=2)

    fig.savefig(save_fn, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {save_fn}")
    plt.close(fig)

# ── Kp-G-K path (201 k-points) ───────────────────────────────────────────────
k_list_kpgk, norm_kpgk = get_k_list("Kp-G-K", 201, tmd="WSe2", return_norm=True)
evals_kpgk = compute_bands(k_list_kpgk)
band_indices = list(range(27, 19, -1))
out_kpgk = np.column_stack([norm_kpgk] + [evals_kpgk[:, i] for i in band_indices])

fn_kpgk = os.path.join(out_dir, "bilayer_bands_KpGK.txt")
header_kpgk = "k_Angstrom\tBand27_eV\tBand26_eV\tBand25_eV\tBand24_eV\tBand23_eV\tBand22_eV\tBand21_eV\tBand20_eV"
np.savetxt(fn_kpgk, out_kpgk, fmt="%.8f", delimiter="\t", header=header_kpgk, comments="")
print(f"Saved Kp-G-K data: {fn_kpgk}  ({out_kpgk.shape[0]} k-points)")

arpes_k, arpes_e = load_arpes_kpgk()
plot_bands(norm_kpgk, out_kpgk[:, 1:], arpes_data=(arpes_k, arpes_e),
           path_label="Kp-G-K",
           save_fn=os.path.join(out_dir, "bilayer_bands_KpGK.png"))

# ── Kp-M-K path (101 k-points) ───────────────────────────────────────────────
k_list_kpmk, norm_kpmk = get_k_list("Kp-M-K", 101, tmd="WSe2", return_norm=True)
evals_kpmk = compute_bands(k_list_kpmk)
out_kpmk = np.column_stack([norm_kpmk] + [evals_kpmk[:, i] for i in band_indices])

fn_kpmk = os.path.join(out_dir, "bilayer_bands_KpMK.txt")
header_kpmk = "k_Angstrom\tBand27_eV\tBand26_eV\tBand25_eV\tBand24_eV\tBand23_eV\tBand22_eV\tBand21_eV\tBand20_eV"
np.savetxt(fn_kpmk, out_kpmk, fmt="%.8f", delimiter="\t", header=header_kpmk, comments="")
print(f"Saved Kp-M-K data: {fn_kpmk}  ({out_kpmk.shape[0]} k-points)")

plot_bands(norm_kpmk, out_kpmk[:, 1:], arpes_data=None,
           path_label="Kp-M-K",
           save_fn=os.path.join(out_dir, "bilayer_bands_KpMK.png"))
