"""TMD material class: encapsulates lattice constants, DFT parameters, and Hamiltonian builders.

The ``TMDMaterial`` class is the central object for a single TMD monolayer
(WSe2 or WS2). It holds the 43 tight-binding parameters (DFT-derived or
fitted), and provides methods to construct the hopping matrices, on-site
energies, and spin-orbit coupling Hamiltonian from those parameters.

The 43 parameters are indexed as follows:
    - 0–6:   on-site energies (epsilon_1, epsilon_3, epsilon_4, epsilon_6,
             epsilon_7, epsilon_9, epsilon_10)
    - 7–27:  21 nearest-neighbor hopping parameters (t1)
    - 28–35: 8 M-X coupling hopping parameters (t5)
    - 36–39: 4 second-nearest-neighbor hopping parameters (t6)
    - 40:    global energy offset
    - 41–42: spin-orbit coupling strengths (L_W, L_S)
"""
import numpy as np
from .constants import (
    LATTICE_CONSTANTS,
    DFT_INITIAL_PARAMS,
    IND_OFF,
    IND_SOC,
    IND_PZ,
    IND_PXY,
    IND_EPS,
    IND_T1,
    IND_T5,
    IND_T6,
    J_PLUS,
    J_MINUS,
    J_MX_PLUS,
    J_MX_MINUS,
    A_1,
    A_2,
)


def _R_z(t):
    """2D rotation matrix by angle t (radians)."""
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _find_t(params: np.ndarray) -> list[np.ndarray]:
    """Build 6 hopping matrices (t1–t6) from the 43-parameter array.

    Maps the flat parameter vector into six 11×11 hopping matrices,
    enforcing all symmetry relations between equivalent matrix elements.
    Only the independent parameters are read from ``params``; the
    dependent ones are computed from symmetry.

    Parameters
    ----------
    params : np.ndarray
        Flat array of 43 tight-binding parameters.

    Returns
    -------
    list[np.ndarray]
        Six 11×11 hopping matrices [t1, t2, t3, t4, t5, t6].
        t1–t3 are nearest-neighbor, t4–t5 are M-X coupling,
        t6 is second-nearest-neighbor.
    """
    t = [np.zeros((11, 11)) for _ in range(6)]
    # Independent parameters
    t[0][0, 0] = params[7]
    t[0][1, 1] = params[8]
    t[0][2, 2] = params[9]
    t[0][3, 3] = params[10]
    t[0][4, 4] = params[11]
    t[0][5, 5] = params[12]
    t[0][6, 6] = params[13]
    t[0][7, 7] = params[14]
    t[0][8, 8] = params[15]
    t[0][9, 9] = params[16]
    t[0][10, 10] = params[17]
    t[0][2, 4] = params[18]
    t[0][5, 7] = params[19]
    t[0][8, 10] = params[20]
    t[0][0, 1] = params[21]
    t[0][2, 3] = params[22]
    t[0][3, 4] = params[23]
    t[0][5, 6] = params[24]
    t[0][6, 7] = params[25]
    t[0][8, 9] = params[26]
    t[0][9, 10] = params[27]
    t[4][3, 0] = params[28]
    t[4][2, 1] = params[29]
    t[4][4, 1] = params[30]
    t[4][8, 5] = params[31]
    t[4][10, 5] = params[32]
    t[4][9, 6] = params[33]
    t[4][8, 7] = params[34]
    t[4][10, 7] = params[35]
    t[5][8, 5] = params[36]
    t[5][10, 5] = params[37]
    t[5][8, 7] = params[38]
    t[5][10, 7] = params[39]

    # Dependent parameters from symmetry rotation (list_1: pairs with 3-fold symmetry)
    list_1 = ((1, 2, -1), (4, 5, 3), (7, 8, 6), (10, 11, 9))
    for a, b, g in list_1:
        a -= 1
        b -= 1
        g -= 1
        t[1][a, a] = 1 / 4 * t[0][a, a] + 3 / 4 * t[0][b, b]
        t[1][b, b] = 3 / 4 * t[0][a, a] + 1 / 4 * t[0][b, b]
        t[1][a, b] = np.sqrt(3) / 4 * (t[0][a, a] - t[0][b, b]) - t[0][a, b]
        t[2][a, b] = -np.sqrt(3) / 4 * (t[0][a, a] - t[0][b, b]) - t[0][a, b]
        if g >= 0:
            t[1][g, g] = t[0][g, g]
            t[1][g, b] = np.sqrt(3) / 2 * t[0][g, a] - 1 / 2 * t[0][g, b]
            t[2][g, b] = -np.sqrt(3) / 2 * t[0][g, a] - 1 / 2 * t[0][g, b]
            t[1][g, a] = np.sqrt(3) / 2 * t[0][g, b] + 1 / 2 * t[0][g, a]
            t[2][g, a] = -np.sqrt(3) / 2 * t[0][g, b] + 1 / 2 * t[0][g, a]

    # Dependent parameters from M-X coupling symmetry (list_2)
    list_2 = ((1, 2, 4, 5, 3), (7, 8, 10, 11, 9))
    for a, b, ap, bp, gp in list_2:
        a -= 1
        b -= 1
        ap -= 1
        bp -= 1
        gp -= 1
        t[3][ap, a] = 1 / 4 * t[4][ap, a] + 3 / 4 * t[4][bp, b]
        t[3][bp, b] = 3 / 4 * t[4][ap, a] + 1 / 4 * t[4][bp, b]
        t[3][bp, a] = t[3][ap, b] = -np.sqrt(3) / 4 * t[4][ap, a] + np.sqrt(3) / 4 * t[4][bp, b]
        t[3][gp, a] = -np.sqrt(3) / 2 * t[4][gp, b]
        t[3][gp, b] = -1 / 2 * t[4][gp, b]

    t[3][8, 5] = t[4][8, 5]
    t[3][9, 5] = -np.sqrt(3) / 2 * t[4][10, 5]
    t[3][10, 5] = -1 / 2 * t[4][10, 5]
    return t


def _find_e(params: np.ndarray) -> np.ndarray:
    """Extract 11 on-site energies from the parameter array.

    Uses symmetry: epsilon[1] = epsilon[0], epsilon[4] = epsilon[3],
    epsilon[7] = epsilon[6], epsilon[10] = epsilon[9].

    Parameters
    ----------
    params : np.ndarray
        Flat array of 43 tight-binding parameters.

    Returns
    -------
    np.ndarray
        Array of 11 on-site energies.
    """
    e = np.zeros(11)
    e[0] = params[0]
    e[1] = e[0]
    e[2] = params[1]
    e[3] = params[2]
    e[4] = e[3]
    e[5] = params[3]
    e[6] = params[4]
    e[7] = e[6]
    e[8] = params[5]
    e[9] = params[6]
    e[10] = e[9]
    return e


def _find_HSO(SO_pars: np.ndarray) -> np.ndarray:
    """Construct the 22×22 spin-orbit coupling Hamiltonian.

    Built from the metal (W) and chalcogen (S/Se) SOC strengths.
    The matrix couples spin-up and spin-down blocks according to
    the angular momentum algebra of d and p orbitals.

    Parameters
    ----------
    SO_pars : np.ndarray
        Array of 2 SOC strengths: [L_W, L_S] in eV.

    Returns
    -------
    np.ndarray
        22×22 complex SOC Hamiltonian matrix.
    """
    l_M = SO_pars[0]
    l_X = SO_pars[1]
    HSO = np.zeros((22, 22), dtype=complex)
    s3 = np.sqrt(3)
    # Spin-up block couplings
    HSO[0, 1] = l_M * 1j / 2
    HSO[0, 11 + 5] = l_M * s3 / 2
    HSO[0, 11 + 6] = -l_M * 1j / 2
    HSO[0, 11 + 7] = -l_M / 2
    HSO[1, 0] = -l_M * 1j / 2
    HSO[1, 11 + 5] = l_M * 1j * s3 / 2
    HSO[1, 11 + 6] = -l_M / 2
    HSO[1, 11 + 7] = l_M * 1j / 2
    HSO[2, 11 + 9] = -l_X / 2
    HSO[2, 11 + 10] = -l_X * 1j / 2
    HSO[3, 4] = l_X * 1j / 2
    HSO[3, 11 + 8] = l_X / 2
    HSO[4, 3] = -l_X * 1j / 2
    HSO[4, 11 + 8] = l_X * 1j / 2
    HSO[5, 11 + 0] = -l_M * s3 / 2
    HSO[5, 11 + 1] = -l_M * 1j * s3 / 2
    HSO[6, 7] = -l_M * 1j
    HSO[6, 11 + 0] = l_M * 1j / 2
    HSO[6, 11 + 1] = l_M / 2
    HSO[7, 6] = l_M * 1j
    HSO[7, 11 + 0] = l_M / 2
    HSO[7, 11 + 1] = -l_M * 1j / 2
    HSO[8, 11 + 3] = -l_X / 2
    HSO[8, 11 + 4] = -l_X * 1j / 2
    HSO[9, 10] = l_X * 1j / 2
    HSO[9, 11 + 2] = l_X / 2
    HSO[10, 9] = -l_X * 1j / 2
    HSO[10, 11 + 2] = l_X * 1j / 2
    # Spin-down block couplings
    HSO[11 + 0, 5] = -l_M * s3 / 2
    HSO[11 + 0, 6] = -l_M * 1j / 2
    HSO[11 + 0, 7] = l_M / 2
    HSO[11 + 0, 11 + 1] = -l_M * 1j / 2
    HSO[11 + 1, 5] = l_M * 1j * s3 / 2
    HSO[11 + 1, 6] = l_M / 2
    HSO[11 + 1, 7] = l_M * 1j / 2
    HSO[11 + 1, 11 + 0] = l_M * 1j / 2
    HSO[11 + 2, 9] = l_X / 2
    HSO[11 + 2, 10] = -l_X * 1j / 2
    HSO[11 + 3, 8] = -l_X / 2
    HSO[11 + 3, 11 + 4] = -l_X * 1j / 2
    HSO[11 + 4, 8] = l_X * 1j / 2
    HSO[11 + 4, 11 + 3] = l_X * 1j / 2
    HSO[11 + 5, 0] = l_M * s3 / 2
    HSO[11 + 5, 1] = -l_M * 1j * s3 / 2
    HSO[11 + 6, 0] = l_M * 1j / 2
    HSO[11 + 6, 1] = -l_M / 2
    HSO[11 + 6, 11 + 7] = l_M * 1j
    HSO[11 + 7, 0] = -l_M / 2
    HSO[11 + 7, 1] = -l_M * 1j / 2
    HSO[11 + 7, 11 + 6] = -l_M * 1j
    HSO[11 + 8, 3] = l_X / 2
    HSO[11 + 8, 4] = -l_X * 1j / 2
    HSO[11 + 9, 2] = -l_X / 2
    HSO[11 + 9, 11 + 10] = -l_X * 1j / 2
    HSO[11 + 10, 2] = l_X * 1j / 2
    HSO[11 + 10, 11 + 9] = l_X * 1j / 2
    return HSO


class TMDMaterial:
    """Represents a TMD monolayer (WSe2 or WS2) with its tight-binding parameters.

    Encapsulates the lattice constant, DFT-derived initial parameters,
    optionally fitted parameters, and provides methods to construct
    Hamiltonian components from the parameter vector.

    Parameters
    ----------
    name : str
        Material name, must be "WSe2" or "WS2".
    params : np.ndarray, optional
        Fitted parameter array (43 values). If None, uses DFT initial params.

    Examples
    --------
    >>> mat = TMDMaterial("WSe2")
    >>> mat.lattice_constant
    3.32
    >>> t = mat.build_hopping_matrices()  # 6 matrices of shape (11, 11)
    >>> HSO = mat.build_soc_hamiltonian()  # 22x22 matrix
    """

    def __init__(self, name: str, params: np.ndarray | None = None):
        if name not in LATTICE_CONSTANTS:
            raise ValueError(f"Unknown TMD: {name}")
        self.name = name
        self.dft_params = np.array(DFT_INITIAL_PARAMS[name])
        self.fitted_params = params

    @property
    def lattice_constant(self) -> float:
        """Lattice constant in Angstrom."""
        return LATTICE_CONSTANTS[self.name]

    @property
    def params(self) -> np.ndarray:
        """Active parameter array (fitted if available, otherwise DFT)."""
        return self.fitted_params if self.fitted_params is not None else self.dft_params

    def load_fitted(self, filepath: str):
        """Load fitted parameters from a .npy file.

        Parameters
        ----------
        filepath : str
            Path to a .npy file containing a 43-element array.
        """
        self.fitted_params = np.load(filepath)

    def build_hopping_matrices(self, params: np.ndarray | None = None) -> list[np.ndarray]:
        """Build the 6 hopping matrices (t1–t6) from parameters.

        Parameters
        ----------
        params : np.ndarray, optional
            43-element parameter array. Uses ``self.params`` if not given.

        Returns
        -------
        list[np.ndarray]
            Six 11×11 hopping matrices.
        """
        return _find_t(params if params is not None else self.params)

    def build_onsite_energies(self, params: np.ndarray | None = None) -> np.ndarray:
        """Extract the 11 on-site energies from parameters.

        Parameters
        ----------
        params : np.ndarray, optional
            43-element parameter array. Uses ``self.params`` if not given.

        Returns
        -------
        np.ndarray
            Array of 11 on-site energies.
        """
        return _find_e(params if params is not None else self.params)

    def build_soc_hamiltonian(self, params: np.ndarray | None = None) -> np.ndarray:
        """Construct the 22×22 spin-orbit coupling Hamiltonian.

        Parameters
        ----------
        params : np.ndarray, optional
            43-element parameter array. Uses ``self.params`` if not given.
            Only the last two elements (L_W, L_S) are used.

        Returns
        -------
        np.ndarray
            22×22 complex SOC Hamiltonian.
        """
        p = params if params is not None else self.params
        return _find_HSO(p[-2:])

    def get_bounds_relative(self, rp: float, rpz: float, rpxy: float, rl: float) -> list[tuple]:
        """Generate parameter bounds as relative fractions of DFT values.

        Parameters
        ----------
        rp : float
            Relative bound for general parameters.
        rpz : float
            Relative bound for z-orbital parameters.
        rpxy : float
            Relative bound for xy-orbital parameters.
        rl : float
            Relative bound for SOC parameters.

        Returns
        -------
        list[tuple]
            List of (lower, upper) bounds for each of the 43 parameters.
        """
        bounds = []
        for i in range(self.params.shape[0]):
            if i in IND_OFF:
                bounds.append((-3, 0))
            elif i in IND_SOC:
                r = rl * abs(self.params[i])
                bounds.append((self.params[i] - r, self.params[i] + r))
            elif i in IND_PZ:
                r = rpz * abs(self.params[i])
                bounds.append((self.params[i] - r, self.params[i] + r))
            elif i in IND_PXY:
                r = rpxy * abs(self.params[i])
                bounds.append((self.params[i] - r, self.params[i] + r))
            else:
                r = rp * abs(self.params[i])
                bounds.append((self.params[i] - r, self.params[i] + r))
        return bounds

    def get_bounds_absolute(self, peps: float, pt1: float, pt5: float, pt6: float, pl: float) -> list[tuple]:
        """Generate parameter bounds as absolute values.

        Parameters
        ----------
        peps : float
            Absolute bound for on-site energy parameters.
        pt1 : float
            Absolute bound for t1 hopping parameters.
        pt5 : float
            Absolute bound for t5 hopping parameters.
        pt6 : float
            Absolute bound for t6 hopping parameters.
        pl : float
            Absolute bound for SOC parameters.

        Returns
        -------
        list[tuple]
            List of (lower, upper) bounds for each of the 43 parameters.
        """
        bounds = []
        for i in range(self.params.shape[0]):
            if i in IND_OFF:
                bounds.append((-3, 0))
            elif i in IND_SOC:
                bounds.append((-pl, pl))
            elif i in IND_EPS:
                bounds.append((-peps, peps))
            elif i in IND_T1:
                bounds.append((-pt1, pt1))
            elif i in IND_T5:
                bounds.append((-pt5, pt5))
            elif i in IND_T6:
                bounds.append((-pt6, pt6))
            else:
                raise ValueError(f"Index not in any list for bounds: {i}")
        return bounds

    def parameter_distance(self, params: np.ndarray | None = None) -> float:
        """Compute normalized distance from DFT parameter values.

        Mean absolute relative deviation of all parameters (except offset)
        from their DFT-derived initial values.

        Parameters
        ----------
        params : np.ndarray, optional
            43-element parameter array. Uses ``self.params`` if not given.

        Returns
        -------
        float
            Normalized parameter distance.
        """
        p = params if params is not None else self.params
        distance = (
            np.sum(np.absolute((p[:-3] - self.dft_params[:-3]) / self.dft_params[:-3]))
            + np.sum(np.absolute((p[-2:] - self.dft_params[-2:]) / self.dft_params[-2:]))
        )
        distance /= p.shape[0] - 1
        return distance
