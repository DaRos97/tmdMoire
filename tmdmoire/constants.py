"""Physical constants, DFT parameters, and configuration data.

All hardcoded values extracted from the original CORE_functions.py.
This module is the single source of truth for:
- Lattice constants for WS2 and WSe2
- DFT-derived initial tight-binding parameters (43 per material)
- Twist angles for experimental samples
- ARPES EDC peak positions
- Orbital index mappings and symmetry relations
"""
import numpy as np

# ─── Lattice constants (Angstrom) ────────────────────────────────────────────

LATTICE_CONSTANTS = {"WS2": 3.18, "WSe2": 3.32}
"""Lattice constants for each TMD monolayer in Angstrom."""

TMD_NAMES = ["WSe2", "WS2"]
"""Supported transition metal dichalcogenide materials."""

# ─── Parameter naming ────────────────────────────────────────────────────────

PARAMETER_NAMES = [
    "e1", "e3", "e4", "e6", "e7", "e9", "e10",
    "t1_11", "t1_22", "t1_33", "t1_44", "t1_55", "t1_66", "t1_77", "t1_88",
    "t1_99", "t1_1010", "t1_1111", "t1_35", "t1_68", "t1_911",
    "t1_12", "t1_34", "t1_45", "t1_67", "t1_78", "t1_910", "t1_1011",
    "t5_41", "t5_32", "t5_52", "t5_96", "t5_116", "t5_107", "t5_98", "t5_118",
    "t6_96", "t6_116", "t6_98", "t6_118",
    "offset", "L_W", "L_S",
]
"""Human-readable names for the 43 tight-binding parameters."""

FORMATTED_NAMES = [
    r"$\epsilon_1$", r"$\epsilon_3$", r"$\epsilon_4$", r"$\epsilon_6$",
    r"$\epsilon_7$", r"$\epsilon_9$", r"$\epsilon_{10}$",
    "$t^{(1)}_{1,1}$", "$t^{(1)}_{2,2}$", "$t^{(1)}_{3,3}$", "$t^{(1)}_{4,4}$",
    "$t^{(1)}_{5,5}$", "$t^{(1)}_{6,6}$", "$t^{(1)}_{7,7}$", "$t^{(1)}_{8,8}$",
    "$t^{(1)}_{9,9}$", "$t^{(1)}_{10,10}$", "$t^{(1)}_{11,11}$",
    "$t^{(1)}_{3,5}$", "$t^{(1)}_{6,8}$", "$t^{(1)}_{9,11}$",
    "$t^{(1)}_{1,2}$", "$t^{(1)}_{3,4}$", "$t^{(1)}_{4,5}$", "$t^{(1)}_{6,7}$",
    "$t^{(1)}_{7,8}$", "$t^{(1)}_{9,10}$", "$t^{(1)}_{10,11}$",
    "$t^{(5)}_{4,1}$", "$t^{(5)}_{3,2}$", "$t^{(5)}_{5,2}$", "$t^{(5)}_{9,6}$",
    "$t^{(5)}_{11,6}$", "$t^{(5)}_{10,7}$", "$t^{(5)}_{9,8}$", "$t^{(5)}_{11,8}$",
    "$t^{(6)}_{9,6}$", "$t^{(6)}_{11,6}$", "$t^{(6)}_{9,8}$", "$t^{(6)}_{11,8}$",
    "$offset$", r"$\lambda_W$", r"$\lambda_{Se}$",
]
"""LaTeX-formatted parameter names for plotting."""

# ─── DFT initial parameters ──────────────────────────────────────────────────

DFT_INITIAL_PARAMS = {
    "WS2": [
        1.3754, -1.1278, -1.5534, -0.0393, 0.1984, -3.3706, -2.3461,
        -0.2011, 0.0263, -0.1749, 0.8726, -0.2187, -0.3716, 0.3537,
        -0.6892, -0.2112, 0.9673, 0.0143, -0.0818, 0.4896, -0.0315,
        -0.3106, -0.1105, -0.0989, -0.1467, -0.3030, 0.1645, -0.1018,
        -0.8855, -1.4376, 2.3121, -1.0130, -0.9878, 1.5629, -0.9491, 0.6718,
        -0.0659, -0.1533, -0.2618, -0.2736,
        -1.350,
        0.2874, 0.0556,
    ],
    "WSe2": [
        1.0349, -0.9573, -1.3937, -0.1667, 0.0984, -3.3642, -2.1820,
        -0.1395, 0.0129, -0.2171, 0.9763, -0.1985, -0.3330, 0.3190,
        -0.5837, -0.2399, 1.0470, 0.0029, -0.0912, 0.4233, -0.0377,
        -0.2321, -0.0797, -0.0920, -0.1250, -0.2456, 0.1857, -0.1027,
        -0.7744, -1.4014, 2.0858, -0.8998, -0.9044, 1.4030, -0.8548, 0.5711,
        -0.0676, -0.1608, -0.2618, -0.2424,
        -0.736,
        0.2874, 0.2470,
    ],
}
"""DFT-derived initial tight-binding parameters for each TMD (43 values).
Order: 7 on-site energies, 21 t1 hoppings, 8 t5 hoppings, 4 t6 hoppings,
1 offset, 2 SOC strengths (L_W, L_S).
"""

# ─── Sample configuration ────────────────────────────────────────────────────

TWIST_ANGLES = {"S3": 1.8, "S11": 2.8}
"""Twist angles in degrees for experimental samples (from LEED)."""

ENERGY_OFFSETS = {"S3": 0, "S11": -0.47}
"""Energy offsets in eV applied to each sample's ARPES data."""

EDC_G_POSITIONS = {"S11": np.array([-1.1599, -1.2531, -1.82]), "S3": (np.nan,)}
"""Experimental EDC peak positions at Gamma (eV): TVB, side band, WS2 band."""

EDC_K_POSITIONS = {"S11": np.array([-0.8990, -1.0696]), "S3": (np.nan,)}
"""Experimental EDC peak positions at K (eV): TVB, moire band."""

SAMPLE_PARAMS = {
    "S11": [0.0, -3.5, 810, 2371, 89, 1899],
    "S3": [0, -2.5, 697, 2156, 108, 1681],
    "S11zoom": [-0.6, -1.8, 840, 2980, 86, 1147],
}
"""ARPES image parameters: [E_max, E_min, pixel_k_minus1, pixel_k_plus1, pixel_E_max, pixel_E_min]."""

ENERGY_BOUNDS = {"S11zoom": (-0.6, -1.8), "S11": (-0.0, -3.5), "S3": (-0.2, -1.8)}
"""Energy window (max, min) in eV for each sample."""

ORBITAL_CHARACTER = {
    "WSe2": {"G": (0.2740, 0.6606), "K": (0.1856, 0.2116, 0.8144, 0.7763)},
    "WS2": {"G": (0.3205, 0.6571), "K": (0.1960, 0.2366, 0.8040, 0.7575)},
}
"""DFT-derived orbital occupations for constraints.
G: (p_z^e, d_z2) for degenerate TVB.
K: (p_-1^e tvb1, p_-1^e tvb2, d_2 tvb1, d_2 tvb2).
"""

MONOLAYER_OFFSETS = {"WSe2": -0.052, "WS2": 0.01}
"""Energy shifts applied to KMKp path bands for each TMD."""

# ─── Symmetry relations ──────────────────────────────────────────────────────

M_LIST = [[-1, 1], [-1, 0], [0, -1], [1, -1], [1, 0], [0, 1]]
"""Six nearest-neighbor displacement vectors in reciprocal lattice index space.
Used for moire potential coupling between mini-BZ cells.
"""

J_PLUS = ((3, 5), (6, 8), (9, 11))
"""Orbital pairs with symmetric (cosine) hopping terms."""

J_MINUS = ((1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11))
"""Orbital pairs with antisymmetric (sine) hopping terms."""

J_MX_PLUS = ((3, 1), (5, 1), (4, 2), (10, 6), (9, 7), (11, 7), (10, 8))
"""Orbital pairs for M-X coupling with plus symmetry."""

J_MX_MINUS = ((4, 1), (3, 2), (5, 2), (9, 6), (11, 6), (10, 7), (9, 8), (11, 8))
"""Orbital pairs for M-X coupling with minus symmetry."""

# ─── Real-space lattice vectors ──────────────────────────────────────────────

A_1 = np.array([1, 0])
A_2 = np.array([-1 / 2, np.sqrt(3) / 2])
"""Primitive lattice vectors (dimensionless, scaled by lattice constant)."""

# ─── Orbital indices ─────────────────────────────────────────────────────────

xz_i = 0
yz_i = 1
zo_i = 2
xo_i = 3
yo_i = 4
z2_i = 5
xy_i = 6
x2_i = 7
ze_i = 8
xe_i = 9
ye_i = 10
"""Indices for the 11 orbitals in the spin-up block of the Hamiltonian."""

IND_OPO = [0, 1, 2, 5, 8, 11, 12, 13, 16, 19]
"""Out-of-plane orbital indices (both spin blocks): d_xz, d_yz, p_z^o, d_z2, p_z^e."""

IND_IPO = [3, 4, 6, 7, 9, 10, 14, 15, 17, 18, 20, 21]
"""In-plane orbital indices (both spin blocks): p_x^o, p_y^o, d_xy, d_x2-y2, p_x^e, p_y^e."""

IND_ILC = [2, 5, 8, 13, 16, 19]
"""Interlayer-coupling orbital indices: p_z^o, d_z2, p_z^e (both spins)."""

TVB2 = [12, 13]
"""Top valence band indices for WS2 (2 bands near TVB)."""

TVB4 = [10, 11, 12, 13]
"""Top valence band indices for WSe2 (4 bands near TVB)."""

BCB2 = [14, 15]
"""Bottom conduction band indices."""

# ─── Parameter group indices ─────────────────────────────────────────────────

IND_OFF = [40]
"""Index of the global energy offset parameter."""

IND_SOC = [41, 42]
"""Indices of the SOC strength parameters (L_W, L_S)."""

IND_PZ = [3, 5, 12, 15, 19, 20, 24, 26, 31, 32, 34, 36, 37, 38]
"""Parameter indices associated with z-oriented orbitals."""

IND_PXY = [4, 6, 13, 14, 16, 17, 25, 27, 33, 35, 39]
"""Parameter indices associated with xy-oriented orbitals."""

IND_EPS = list(np.arange(7))
"""Indices of the 7 on-site energy parameters."""

IND_T1 = list(np.arange(7, 28))
"""Indices of the 21 nearest-neighbor hopping parameters."""

IND_T5 = list(np.arange(28, 36))
"""Indices of the 8 M-X coupling hopping parameters."""

IND_T6 = list(np.arange(36, 40))
"""Indices of the 4 second-nearest-neighbor hopping parameters."""
