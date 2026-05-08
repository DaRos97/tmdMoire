from .constants import (
    LATTICE_CONSTANTS, TMD_NAMES, PARAMETER_NAMES, FORMATTED_NAMES,
    DFT_INITIAL_PARAMS, TWIST_ANGLES, ENERGY_OFFSETS, EDC_G_POSITIONS,
    EDC_K_POSITIONS, SAMPLE_PARAMS, ENERGY_BOUNDS, ORBITAL_CHARACTER,
    MONOLAYER_OFFSETS, M_LIST, J_PLUS, J_MINUS, J_MX_PLUS, J_MX_MINUS,
    A_1, A_2, IND_OPO, IND_IPO, IND_ILC, TVB2, TVB4, BCB2,
    IND_OFF, IND_SOC, IND_PZ, IND_PXY, IND_EPS, IND_T1, IND_T5, IND_T6,
    xz_i, yz_i, zo_i, xo_i, yo_i, z2_i, xy_i, x2_i, ze_i, xe_i, ye_i,
)
from .material import TMDMaterial

# Monolayer subpackage
from .monolayer import MonolayerData, ParameterFitter, GridScorer, MonolayerHamiltonian

# Bilayer subpackage
from .bilayer import MoireGeometry, BilayerData, BilayerFitter, MoireHamiltonian, EDCAnalyzer

# Plotting
from .plotting import (
    plot_data_pipeline, plot_bands, plot_parameters_absolute,
    plot_orbital_content, plot_top_results,
    plot_bilayer_data, plot_bilayer_fit,
)

# Utils
from .utils import get_repo_root, prepare_run_dir, get_filename, R_z, get_k_list

# Backward compatibility aliases
ARPESData = MonolayerData

__all__ = [
    # Classes
    "TMDMaterial",
    "MonolayerData",
    "ARPESData",  # backward compat
    "ParameterFitter",
    "GridScorer",
    "MonolayerHamiltonian",
    "MoireGeometry",
    "BilayerData",
    "BilayerFitter",
    "MoireHamiltonian",
    "EDCAnalyzer",
    # Functions
    "get_repo_root",
    "prepare_run_dir",
    "get_filename",
    "R_z",
    "get_k_list",
    "plot_data_pipeline",
    "plot_bands",
    "plot_parameters_absolute",
    "plot_orbital_content",
    "plot_top_results",
    "plot_bilayer_data",
    "plot_bilayer_fit",
    # Constants
    "LATTICE_CONSTANTS",
    "TMD_NAMES",
    "PARAMETER_NAMES",
    "FORMATTED_NAMES",
    "DFT_INITIAL_PARAMS",
    "TWIST_ANGLES",
    "ENERGY_OFFSETS",
    "EDC_G_POSITIONS",
    "EDC_K_POSITIONS",
    "SAMPLE_PARAMS",
    "ENERGY_BOUNDS",
    "ORBITAL_CHARACTER",
    "MONOLAYER_OFFSETS",
    "M_LIST",
    "J_PLUS",
    "J_MINUS",
    "J_MX_PLUS",
    "J_MX_MINUS",
    "A_1",
    "A_2",
    "IND_OPO",
    "IND_IPO",
    "IND_ILC",
    "TVB2",
    "TVB4",
    "BCB2",
    "IND_OFF",
    "IND_SOC",
    "IND_PZ",
    "IND_PXY",
    "IND_EPS",
    "IND_T1",
    "IND_T5",
    "IND_T6",
    "xz_i", "yz_i", "zo_i", "xo_i", "yo_i", "z2_i", "xy_i", "x2_i", "ze_i", "xe_i", "ye_i",
]
