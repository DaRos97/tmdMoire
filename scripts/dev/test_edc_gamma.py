"""Test EDC intensity profile at Gamma with fixed parameters.

Uses Vg=15 meV, phiG=180 deg, and fitted interlayer couplings.
Computes the EDC intensity profile, fits 4 Lorentzians (TVB main/side + LVB main/side),
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

from tmdmoire import TMDMaterial, MoireGeometry, EDCAnalyzer, MoireHamiltonian
from tmdmoire import TWIST_ANGLES, ENERGY_OFFSETS
from tmdmoire.utils.paths import get_repo_root

master_folder = get_repo_root()

# ─── Parameters ──────────────────────────────────────────────────────────────

Vg = 0.015          # 15 meV
phiG_deg = 180
phiG = phiG_deg / 180 * np.pi
spreadE = 0.03
sample = "S11"
n_shells = 2
theta = TWIST_ANGLES[sample]

# ─── Load materials ──────────────────────────────────────────────────────────

monolayer_fns = {
    "WSe2": master_folder + "/Inputs/monolayer_fitting/tb_WSe2_abs_8_4_5_2_0_K_0.0001_0.13_0.005_1_0.01_5.npy",
    "WS2": master_folder + "/Inputs/monolayer_fitting/tb_WS2_abs_8_4_5_2_0_K_0_0.125_0.011_1_0.01_5.npy",
}

wse2 = TMDMaterial("WSe2")
wse2.load_fitted(monolayer_fns["WSe2"])
ws2 = TMDMaterial("WS2")
ws2.load_fitted(monolayer_fns["WS2"])

# ─── Load interlayer params ──────────────────────────────────────────────────

interlayer_params = np.load(master_folder + "/Inputs/bilayer_fitting/interlayer_params.npy")
w1p, w1d, w2p, w2d = interlayer_params
print(f"Interlayer params: w1p={w1p:.4f}, w1d={w1d:.4f}, w2p={w2p:.4f}, w2d={w2d:.4f}")

pars_interlayer = {"stacking": "P", "w1p": w1p, "w2p": w2p, "w1d": w1d, "w2d": w2d}

# ─── Geometry and config ─────────────────────────────────────────────────────

geometry = MoireGeometry(theta)
n_cells = MoireGeometry.n_cells(n_shells)

k_list = np.array([np.zeros(2)])
Vk = 0.0
phiK = 0.0
pars_V = (Vg, Vk, phiG, phiK)

config = {
    "n_shells": n_shells,
    "n_cells": n_cells,
    "k_point": k_list,
    "theta_deg": theta,
    "interlayer_params": pars_interlayer,
    "pars_V": pars_V,
}

analyzer = EDCAnalyzer(wse2, ws2, geometry, config)

# ─── Compute EDC (uses 3-Voigt fit internally) ───────────────────────────────

params_gamma = (Vg, phiG)
positions_old, success_old = analyzer.compute_edc(
    params_gamma, "G", spreadE=spreadE, sample=sample
)

if success_old:
    print(f"Old 3-Voigt fit: TVB={positions_old[0]:.4f}, Side={positions_old[1]:.4f}, LVB={positions_old[2]:.4f}")

# ─── Recompute eigenvalues and build intensity ───────────────────────────────

moire_ham = MoireHamiltonian(wse2, ws2, geometry)
evals_raw, evecs_raw = moire_ham.diagonalize(
    k_list, n_shells, pars_interlayer, pars_V
)
evals_raw = evals_raw[0]
evecs_raw = evecs_raw[0]
evals_raw += ENERGY_OFFSETS.get(sample, 0.0)

ab = np.absolute(evecs_raw) ** 2
weights = np.sum(ab[:22, :], axis=0) + np.sum(ab[22 * n_cells:22 * (1 + n_cells), :], axis=0)

index_tvb = 28 * n_cells - 1
index_lvb = 26 * n_cells - 1
index_l = index_lvb - 2 * n_cells + 1

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

# ─── Find 4 peak positions from weights ──────────────────────────────────────

# Find the 4 highest-weight states to seed the fit
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
    if len(peak_states) == 4:
        break

peak_states.sort(key=lambda x: x[0], reverse=True)
print(f"Peak seeds: TVB_main={peak_states[0][0]:.4f}, TVB_side={peak_states[1][0]:.4f}, "
      f"LVB_main={peak_states[2][0]:.4f}, LVB_side={peak_states[3][0]:.4f}")

# ─── Fit with 4 Lorentzians ─────────────────────────────────────────────────

def _lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def _four_lorentzian(x, a1, c1, g1, a2, c2, g2, a3, c3, g3, a4, c4, g4):
    return (_lorentzian(x, a1, c1, g1) + _lorentzian(x, a2, c2, g2)
            + _lorentzian(x, a3, c3, g3) + _lorentzian(x, a4, c4, g4))

model = lmfit.Model(_four_lorentzian)

params_fit = model.make_params(
    a1=peak_states[0][1], c1=peak_states[0][0], g1=spreadE,
    a2=peak_states[1][1], c2=peak_states[1][0], g2=spreadE,
    a3=peak_states[2][1], c3=peak_states[2][0], g3=spreadE,
    a4=peak_states[3][1], c4=peak_states[3][0], g4=spreadE,
)

for p in ["a1", "a2", "a3", "a4"]:
    params_fit[p].set(min=0)
for p in ["g1", "g2", "g3", "g4"]:
    params_fit[p].set(min=1e-4, max=0.2)

# Enforce ordering: c1 > c2 > c3 > c4
params_fit["c1"].set(min=peak_states[0][0] - 0.05, max=peak_states[0][0] + 0.05)
params_fit["c2"].set(min=peak_states[1][0] - 0.05, max=peak_states[1][0] + 0.05)
params_fit["c3"].set(min=peak_states[2][0] - 0.05, max=peak_states[2][0] + 0.05)
params_fit["c4"].set(min=peak_states[3][0] - 0.05, max=peak_states[3][0] + 0.05)

result = model.fit(weight_list, params_fit, x=energy_list)

c1 = result.best_values["c1"]
c2 = result.best_values["c2"]
c3 = result.best_values["c3"]
c4 = result.best_values["c4"]
a1 = result.best_values["a1"]
a2 = result.best_values["a2"]
a3 = result.best_values["a3"]
a4 = result.best_values["a4"]
g1 = result.best_values["g1"]
g2 = result.best_values["g2"]
g3 = result.best_values["g3"]
g4 = result.best_values["g4"]

print(f"Fitted peaks:")
print(f"  TVB main:  E = {c1:.4f} eV, amp = {a1:.4f}, gamma = {g1:.4f}")
print(f"  TVB side:  E = {c2:.4f} eV, amp = {a2:.4f}, gamma = {g2:.4f}")
print(f"  LVB main:  E = {c3:.4f} eV, amp = {a3:.4f}, gamma = {g3:.4f}")
print(f"  LVB side:  E = {c4:.4f} eV, amp = {a4:.4f}, gamma = {g4:.4f}")
print(f"  redchi = {result.redchi:.6f}, success = {result.success}")

fit_curve = result.best_fit

# ─── Individual peak contributions ──────────────────────────────────────────

peak1 = _lorentzian(energy_list, a1, c1, g1)
peak2 = _lorentzian(energy_list, a2, c2, g2)
peak3 = _lorentzian(energy_list, a3, c3, g3)
peak4 = _lorentzian(energy_list, a4, c4, g4)

# ─── Plot ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

ax.plot(energy_list, weight_list, "k-", lw=1.5, label="EDC intensity")
ax.plot(energy_list, fit_curve, "r--", lw=2, label="4-Lorentzian fit")

colors = ["C0", "C1", "C2", "C3"]
labels = [
    f"TVB main = {c1:.3f} eV",
    f"TVB side = {c2:.3f} eV",
    f"LVB main = {c3:.3f} eV",
    f"LVB side = {c4:.3f} eV",
]
peaks = [peak1, peak2, peak3, peak4]
for pk, col, lab in zip(peaks, colors, labels):
    ax.plot(energy_list, pk, color=col, ls="-.", lw=1.2, alpha=0.7, label=lab)

ax.set_xlabel("Energy (eV)", fontsize=12)
ax.set_ylabel("Intensity (a.u.)", fontsize=12)
ax.set_title(
    f"EDC at Gamma: Vg={Vg*1000:.0f} meV, phiG={phiG_deg} deg\n"
    f"w1p={w1p:.3f}, w1d={w1d:.3f}, w2p={w2p:.3f}, w2d={w2d:.3f}",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper right")

out_dir = Path("Data/edc_intensity_test")
out_dir.mkdir(parents=True, exist_ok=True)
out_fn = out_dir / f"edc_gamma_Vg{Vg*1000:.0f}meV_phiG{phiG_deg}deg.png"
fig.savefig(out_fn, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved to {out_fn}")
