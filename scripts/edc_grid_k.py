"""EDC intensity grid sweep at K for HPC.

Loads grid parameters from Inputs/bilayer_fitting/grid_config_k.json,
sweeps 2D parameter space (Vk, phiK) with fixed Gamma parameters from
the Gamma best fit, fits Lorentzians to EDC, computes band gap, and saves results.

Usage:
    python scripts/edc_grid_k.py --chunk <id>/<total>
    python scripts/edc_grid_k.py --chunk 0/128 --run-id 001
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: implement K-point grid sweep
# - Read grid_config_k.json
# - Fixed params: Vg, phiG, w1p, w1d, w2p, w2d from Gamma best fit
# - Sweep Vk, phiK
# - Compute EDC at K (2 Lorentzians) + band gap
# - Save to Data/edc_grid_k_run_<id>/

print("edc_grid_k.py: not yet implemented")
