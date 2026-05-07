"""Load and plot bilayer ARPES data.

Usage: python scripts/plot_bilayer_data.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdmoire import BilayerData, detect_machine, get_master_folder, plot_bilayer_data

machine = detect_machine(os.getcwd())
master_folder = get_master_folder(os.getcwd())

bilayer = BilayerData(master_folder, pts=200)

print(f"Loaded {bilayer.n_bands} bands")
print(f"Raw momentum range: {bilayer.momentum_range[0]:.3f} to {bilayer.momentum_range[1]:.3f} Å¹")
print(f"Max |k| after symmetrization: {bilayer.max_abs_k:.3f} Å⁻¹")
print(f"Fit data shape: {bilayer.fit_data.shape}")

for ib in range(bilayer.n_bands):
    rd = bilayer.raw_data[ib]
    sd = bilayer.sym_data[ib]
    raw_valid = (~np.isnan(rd[:, 1])).sum()
    sym_valid = (~np.isnan(sd[:, 1])).sum()
    k_max = sd[:, 0].max()
    print(f"  Band {ib+1}: raw={len(rd)} ({raw_valid} valid) → sym={len(sd)} ({sym_valid} valid), |k|_max={k_max:.3f}")

plot_bilayer_data(bilayer)
