"""Test EDC intensity profile at K with fixed parameters.

Uses Vk=10 meV, phiK=120 deg, and fixed Gamma parameters from grid_config_k.json.
Computes the EDC intensity profile, fits 2 Lorentzians (main band + side band),
and plots the result.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lmfit

from tmdmoire import TMDMaterial, MoireGeometry, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS, LATTICE_CONSTANTS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parameters ──────────────────────────────────────────────────────────────

Vk = 0.010           # 10 meV
phiK_deg = 120
phiK = phiK_deg / 180 * np.pi
spreadE = 0.03
sample = "S11"
n_shells = 2
theta = TWIST_ANGLES[sample]

# Fixed Gamma params from grid_config_k.json
Vg = 0.015
phiG_deg = 180
phiG = phiG_deg / 180 * np.pi
w1p = -1.3232
w1d = 0.5012
w2p = -0.1774
w2d = 0.0295

# ─── Load materials ──────────────────────────────────────────────────────────

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

pars_interlayer = {"stacking": "P", "w1p": w1p, "w2p": w2p, "w1d": w1d, "w2d": w2d}

# ─── Geometry ────────────────────────────────────────────────────────────────

geometry = MoireGeometry(theta)
n_cells = MoireGeometry.n_cells(n_shells)

# K-point of the monolayer BZ
k_K = np.array([[4 * np.pi / (3 * LATTICE_CONSTANTS["WSe2"]), 0]])

pars_V = (Vg, Vk, phiG, phiK)

# ─── Diagonalize ─────────────────────────────────────────────────────────────

moire_ham = MoireHamiltonian(wse2, ws2, geometry)
evals_raw, evecs_raw = moire_ham.diagonalize(k_K, n_shells, pars_interlayer, pars_V)
evals_raw = evals_raw[0] + ENERGY_OFFSETS.get(sample, 0.0)
evecs_raw = evecs_raw[0]

ab = np.absolute(evecs_raw) ** 2
weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0)

index_tvb = 28 * n_cells - 1
index_l = index_tvb - n_cells + np.argmax(weights[index_tvb - n_cells:index_tvb]) + 1

full_energy_values = evals_raw[index_l:index_tvb + 1]
full_weight_values = weights[index_l:index_tvb + 1]

min_e = full_energy_values[0]
max_e = full_energy_values[-1]
delta = max_e - min_e
min_e -= delta / 2
max_e += delta / 2
n_e = int((max_e - min_e) / 0.005)
energy_list = np.linspace(min_e, max_e, n_e)
weight_list = np.zeros(len(energy_list))

for i in range(len(full_energy_values)):
    weight_list += spreadE / np.pi * full_weight_values[i] / (
        (energy_list - full_energy_values[i]) ** 2 + spreadE ** 2
    )

# ─── Find 2 peak seeds ──────────────────────────────────────────────────────

sorted_indices = np.argsort(full_weight_values)[::-1]
peak_states = []
seen_centers = []
for si in sorted_indices:
    e = full_energy_values[si]
    w = full_weight_values[si]
    if w < 1e-4:
        break
    too_close = any(abs(e - c) < 0.01 for c in seen_centers)
    if not too_close:
        peak_states.append((e, w))
        seen_centers.append(e)
    if len(peak_states) == 2:
        break

peak_states.sort(key=lambda x: x[0], reverse=True)
print(f"Peak seeds: main={peak_states[0][0]:.4f} eV (w={peak_states[0][1]:.4f}), "
      f"side={peak_states[1][0]:.4f} eV (w={peak_states[1][1]:.4f})")

# ─── Fit with 2 Lorentzians ─────────────────────────────────────────────────

def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def _two_lorentzian(x, a1, c1, g1, a2, c2, g2):
    return _lorentzian(x, a1, c1, g1) + _lorentzian(x, a2, c2, g2)

model = lmfit.Model(_two_lorentzian)

params_fit = model.make_params(
    a1=peak_states[0][1], c1=peak_states[0][0], g1=spreadE,
    a2=peak_states[1][1], c2=peak_states[1][0], g2=spreadE,
)

for p in ["a1", "a2"]:
    params_fit[p].set(min=0)
for p in ["g1", "g2"]:
    params_fit[p].set(min=1e-4, max=0.2)

result = model.fit(weight_list, params_fit, x=energy_list)

c1 = result.best_values["c1"]
c2 = result.best_values["c2"]
a1 = result.best_values["a1"]
a2 = result.best_values["a2"]
g1 = result.best_values["g1"]
g2 = result.best_values["g2"]

print(f"Fitted peaks:")
print(f"  Main band: E = {c1:.4f} eV, amp = {a1:.4f}, gamma = {g1:.4f}")
print(f"  Side band: E = {c2:.4f} eV, amp = {a2:.4f}, gamma = {g2:.4f}")
print(f"  redchi = {result.redchi:.6f}, success = {result.success}")

fit_curve = result.best_fit

# ─── Individual peak contributions ──────────────────────────────────────────

peak1 = _lorentzian(energy_list, a1, c1, g1)
peak2 = _lorentzian(energy_list, a2, c2, g2)

# ─── Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

ax.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")
ax.plot(energy_list, fit_curve, "r--", lw=2, label="2-Lorentzian fit")

colors = ["C0", "C1"]
labels = [
    f"Main band = {c1:.3f} eV",
    f"Side band = {c2:.3f} eV",
]
peaks = [peak1, peak2]
for pk, col, lab in zip(peaks, colors, labels):
    ax.plot(energy_list, pk, color=col, ls="-.", lw=1.2, alpha=0.7, label=lab)

ax.set_xlabel("Energy (eV)", fontsize=12)
ax.set_ylabel("Intensity (a.u.)", fontsize=12)
ax.set_title(
    f"EDC at K: Vk={Vk*1000:.0f} meV, phiK={phiK_deg} deg\n"
    f"Vg={Vg*1000:.0f} meV, phiG={phiG_deg} deg\n"
    f"w1p={w1p:.3f}, w1d={w1d:.3f}, w2p={w2p:.3f}, w2d={w2d:.3f}",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper right")

out_dir = Path("Data/edc_intensity_test")
out_dir.mkdir(parents=True, exist_ok=True)
out_fn = out_dir / f"edc_k_Vk{Vk*1000:.0f}meV_phiK{phiK_deg}deg.png"
fig.savefig(out_fn, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to {out_fn}")
