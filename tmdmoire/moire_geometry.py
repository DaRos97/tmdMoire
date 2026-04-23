"""Moiré lattice geometry for twisted heterobilayers.

The ``MoireGeometry`` class encapsulates all geometric properties of a
twisted WSe2/WS2 heterobilayer: the moiré lattice constant, the rotation
of the mini-Brillouin zone, reciprocal lattice vectors, and the lookup
table mapping mini-BZ indices to reciprocal lattice vector combinations.

For a twist angle θ between the two layers, the moiré period is:

    a_moiré = 1 / √(1/a_WSe2² + 1/a_WS2² - 2·cos(θ)/(a_WSe2·a_WS2))

The mini-BZ rotation η describes how the moiré reciprocal lattice is
oriented relative to the monolayer Brillouin zone.
"""
import numpy as np
from .constants import LATTICE_CONSTANTS, M_LIST


def _R_z(t):
    """2D rotation matrix by angle t (radians)."""
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


class MoireGeometry:
    """Encapsulates the geometry of a twisted TMD heterobilayer.

    Computes all moiré lattice properties from the twist angle between
    the WSe2 and WS2 layers.

    Parameters
    ----------
    theta_deg : float
        Twist angle between the two monolayers in degrees.

    Attributes
    ----------
    theta_deg : float
        Twist angle in degrees.

    Examples
    --------
    >>> geo = MoireGeometry(2.8)  # S11 sample
    >>> geo.moire_length
    49.88...
    >>> len(geo.reciprocal_vectors())
    7
    >>> MoireGeometry.n_cells(2)
    19
    """

    def __init__(self, theta_deg: float):
        self.theta_deg = theta_deg

    @property
    def theta_rad(self) -> float:
        """Twist angle in radians."""
        return self.theta_deg / 180 * np.pi

    @property
    def moire_length(self) -> float:
        """Real-space moiré lattice constant in Angstrom.

        Computed from the lattice mismatch and twist angle between
        WSe2 (3.32 Å) and WS2 (3.18 Å).
        """
        a_WS2 = LATTICE_CONSTANTS["WS2"]
        a_WSe2 = LATTICE_CONSTANTS["WSe2"]
        return 1 / np.sqrt(
            1 / a_WSe2**2 + 1 / a_WS2**2 - 2 * np.cos(self.theta_rad) / a_WSe2 / a_WS2
        )

    @property
    def mini_bz_rotation(self) -> float:
        """Rotation angle η of the mini-BZ relative to the monolayer BZ (radians).

        This is the angle of the segment connecting the K points of the
        two layers.
        """
        a_WS2 = LATTICE_CONSTANTS["WS2"]
        a_WSe2 = LATTICE_CONSTANTS["WSe2"]
        return np.arctan(
            np.tan(self.theta_rad / 2) * (a_WSe2 + a_WS2) / (a_WSe2 - a_WS2)
        )

    def reciprocal_vectors(self) -> list[np.ndarray]:
        """Compute the 7 moiré reciprocal lattice vectors.

        Returns
        -------
        list[np.ndarray]
            List of 7 vectors: [0, G1, G2, ..., G6] where G1–G6 are
            the six first-shell reciprocal lattice vectors of the moiré
            lattice, rotated by the mini-BZ rotation angle.
        """
        eta = self.mini_bz_rotation
        a_moire = self.moire_length
        Mat = _R_z(eta)
        G1 = 4 * np.pi / np.sqrt(3) / a_moire * np.array([np.sqrt(3) / 2, 1 / 2])
        G_M = [np.zeros(2), Mat @ G1]
        for i in range(1, 6):
            G_M.append(_R_z(np.pi / 3 * i) @ G_M[1])
        return G_M

    def lattice_vectors(self, tmd: str) -> tuple[list, list]:
        """Compute real-space and reciprocal lattice vectors for a single layer.

        Parameters
        ----------
        tmd : str
            Material name ("WSe2" or "WS2").

        Returns
        -------
        tuple[list, list]
            (As, Bs): six real-space and six reciprocal lattice vectors.
        """
        from .constants import A_1

        theta = self.theta_rad
        a = LATTICE_CONSTANTS[tmd]
        As = [_R_z(theta) @ A_1 * a]
        for i in range(1, 6):
            As.append(_R_z(np.pi / 3 * i) @ As[0])
        area = As[0][0] * As[1][1] - As[0][1] * As[1][0]
        Bs = [2 * np.pi * np.array([As[1][1], -As[1][0]]) / area]
        for i in range(1, 6):
            Bs.append(_R_z(np.pi / 3 * i) @ Bs[0])
        Bs = Bs[1:] + Bs[:1]
        return As, Bs

    @staticmethod
    def n_cells(n_shells: int) -> int:
        """Compute the number of mini-BZ cells for a given number of shells.

        Parameters
        ----------
        n_shells : int
            Number of shells of mini-BZs around the central zone.

        Returns
        -------
        int
            Total number of cells: 1 + 3·n·(n+1).
            For n_shells=2: 19 cells → 836×836 Hamiltonian.
        """
        return int(1 + 3 * n_shells * (n_shells + 1))

    @staticmethod
    def lu_table(n_shells: int) -> list[tuple]:
        """Build the lookup table mapping mini-BZ indices to reciprocal lattice vector indices.

        Each mini-BZ cell is identified by a pair of integers (n, m)
        corresponding to the coefficients of the two primitive reciprocal
        lattice vectors.

        Parameters
        ----------
        n_shells : int
            Number of shells around the central mini-BZ.

        Returns
        -------
        list[tuple]
            List of (n, m) index pairs, one per mini-BZ cell.
        """
        n_cells = int(1 + 3 * n_shells * (n_shells + 1))
        lu = []
        for n in range(0, n_shells + 1):
            i = 0
            j = 0
            for s in range(np.sign(n) * (1 + (n - 1) * n * 3), n * (n + 1) * 3 + 1):
                if s == np.sign(n) * (1 + (n - 1) * n * 3):
                    lu.append((n, 0))
                else:
                    lu.append(
                        (
                            lu[-1][0] + M_LIST[i][0],
                            lu[-1][1] + M_LIST[i][1],
                        )
                    )
                    if j == n - 1:
                        i += 1
                        j = 0
                    else:
                        j += 1
        return lu
