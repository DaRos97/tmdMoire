"""Test script for parameter plotting with random parameters within bounds."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tmdmoire import TMDMaterial
from tmdmoire.plotting import plot_parameters_absolute

tmd = "WSe2"
material = TMDMaterial(tmd)
dft_pars = material.dft_params

bounds = material.get_bounds_absolute(5, 2, 4, 1, 0)
np.random.seed(42)
random_pars = np.array([
    np.random.uniform(low, high) for low, high in bounds
])

legend_info = (
    tmd,
    (0.0, 0.1, 0.1, 0.0, 0.0, 10.0),
    "absolute",
    (5, 2, 4, 1, 0),
    [0.069, 0.0, 0.05, 0.02, 0.0, 0.01],
)

output_path = "/tmp/params_test_plot.png"
plot_parameters_absolute(random_pars, tmd, (5, 2, 4, 1, 0), legend_info, save_path=output_path)

print(f"Test plot saved to: {output_path}")
print(f"DFT params range: [{dft_pars.min():.3f}, {dft_pars.max():.3f}]")
print(f"Random params range: [{random_pars.min():.3f}, {random_pars.max():.3f}]")
print(f"Max |diff|: {np.max(np.abs(random_pars - dft_pars)):.3f}")
print(f"Mean |diff|: {np.mean(np.abs(random_pars - dft_pars)):.3f}")
