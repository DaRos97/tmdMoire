"""Analyze EDC K grid results.

Loads combined.h5 from a run directory, computes distance from experimental
peak positions at K, and produces a 2D heatmap of minimum distance over (Vk, phiK).

Usage:
    python scripts/analyze_edc_k.py --run-id 001
    python scripts/analyze_edc_k.py --run-id 001 --output Figures/edc_k_analysis.png
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: implement K-point analysis
# - Load combined.h5 from edc_grid_k_run_<id>/
# - Compute distance from EDC_K_POSITIONS
# - 2D heatmap: phiK (x) × Vk (y), color = min distance
# - Mark global best-fit point
# - Also plot band gap heatmap

print("analyze_edc_k.py: not yet implemented")
