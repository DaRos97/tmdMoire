"""Shared utilities: machine detection, path resolution, filename generation.

These functions are used by both the monolayer and bilayer workflows.
"""
import os
import shutil
import numpy as np


SOURCE_CONFIG = "Inputs/grid_config.json"
"""Default path to the grid configuration file."""


def prepare_run_dir(run_id: str, material: str) -> str:
    """Create the run output directory and copy grid_config.json into it.

    Creates ``Data/<material>_run_<run_id>/`` if it does not exist, and
    copies ``Inputs/grid_config.json`` into it (only if the destination
    does not exist or is older than the source).

    Parameters
    ----------
    run_id : str
        Run identifier.
    material : str
        Material name (e.g. "WSe2" or "WS2").

    Returns
    -------
    str
        Path to the run directory.
    """
    run_dir = os.path.join("Data", f"{material}_run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    dst = os.path.join(run_dir, "grid_config.json")
    if not os.path.exists(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    elif os.path.getmtime(SOURCE_CONFIG) > os.path.getmtime(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    return run_dir


def detect_machine(cwd: str) -> str:
    """Detect the computing machine from the current working directory.

    Parameters
    ----------
    cwd : str
        Current working directory path.

    Returns
    -------
    str
        Machine identifier: "loc" (local), "hpc" (Baobab/Yggdrasil),
        or "maf" (Mafalda cluster).
    """
    if cwd[6:11] == "dario":
        return "loc"
    elif cwd[:20] == "/home/users/r/rossid":
        return "hpc"
    elif cwd[:13] == "/users/rossid":
        return "maf"
    raise ValueError(f"Unknown machine for cwd: {cwd}")


def get_master_folder(cwd: str) -> str:
    """Determine the repository root directory from the current working directory.

    Parameters
    ----------
    cwd : str
        Current working directory path.

    Returns
    -------
    str
        Path to the repository root (contains ``Inputs/`` subdirectory).
    """
    if cwd[6:11] == "dario":
        return cwd[:40]
    elif cwd[:20] == "/home/users/r/rossid":
        return cwd[:20] + "/git/MoireBands/Code/"
    elif cwd[:13] == "/users/rossid":
        return cwd[:13] + "/git/MoireBands/Code/"
    else:
        return ''


def get_home_dn(machine: str, context: str = "monolayer") -> str:
    """Get the data output directory for the current machine.

    Parameters
    ----------
    machine : str
        Machine identifier from :func:`detect_machine`.
    context : str
        Either "monolayer" or "bilayer" to select the correct subdirectory.

    Returns
    -------
    str
        Absolute path to the data directory.
    """
    base = {
        "loc": "/home/dario/Desktop/git/MoireBands/Code/",
        "hpc": "/home/users/r/rossid/",
        "maf": "/users/rossid/",
    }[machine]
    if context == "monolayer":
        return base + "monolayer_v3.0/"
    elif context == "bilayer":
        return base + "bilayer_v2.0/"
    return base


def get_filename(*args, dirname="", extension="", float_precision=6) -> str:
    """Generate a filename from a sequence of arguments.

    Concatenates arguments with underscores, formatting floats to the
    specified precision. Tuples are recursively expanded. Dicts are skipped.

    Parameters
    ----------
    *args : str, int, float, or tuple
        Components to include in the filename.
    dirname : str
        Directory prefix (must end with "/").
    extension : str
        File extension (must start with ".").
    float_precision : int
        Number of decimal places for float values.

    Returns
    -------
    str
        Constructed filename path.

    Examples
    --------
    >>> get_filename(("test", 1, 2.5), dirname="/tmp/", extension=".npy")
    '/tmp/test_1_2.500000.npy'
    """
    if dirname and dirname[-1] != "/":
        raise ValueError(f"directory name {dirname} must end with '/'")
    if extension and extension[0] != ".":
        raise ValueError(f"extension name {extension} must begin with '.'")
    filename = dirname
    for i, a in enumerate(args):
        t = type(a)
        if t in (str, np.str_):
            filename += str(a)
        elif t in (int, np.int64, np.int32):
            filename += str(a)
        elif t in (float, np.float32, np.float64):
            filename += f"{a:.{float_precision}f}"
        elif t == tuple:
            filename += get_filename(*a, float_precision=float_precision)
        elif t == dict:
            continue
        else:
            raise TypeError(f"Parameter {a} has unsupported type: {t}")
        if i != len(args) - 1:
            filename += "_"
    filename += extension
    return filename


def R_z(t):
    """2D rotation matrix by angle t (radians).

    Parameters
    ----------
    t : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        2×2 rotation matrix.
    """
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def get_k_list(cut, kPts, tmd="WSe2", endpoint=False, returnNorm=False):
    """Generate momentum points along a high-symmetry path.

    Parameters
    ----------
    cut : str
        Path specification, e.g. "Kp-G-K" or "K-M-Kp".
        Supported points: Kp, K, M, G.
    kPts : int
        Total number of points along the path.
    tmd : str
        Material name for lattice constant lookup.
    endpoint : bool
        Whether to include the final point of the last segment.
    returnNorm : bool
        If True, also return cumulative distance along the path.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Momentum points of shape (kPts, 2), optionally with cumulative
        distances of shape (kPts,).
    """
    from .constants import LATTICE_CONSTANTS

    b2 = 4 * np.pi / np.sqrt(3) / LATTICE_CONSTANTS[tmd] * np.array([0, 1])
    b1 = R_z(-np.pi / 3) @ b2
    b6 = R_z(-2 * np.pi / 3) @ b2
    K = (b1 + b6) / 3
    Kp = R_z(np.pi / 3) @ K
    G = np.array([0, 0])
    M = b1 / 2
    dic_Kpts = {"K": K, "Kp": Kp, "G": G, "M": M}
    terms = cut.split("-")
    dks = np.zeros(len(terms) - 1)
    for i in range(len(terms) - 1):
        dks[i] = np.linalg.norm(dic_Kpts[terms[i + 1]] - dic_Kpts[terms[i]])
    tot_k = np.sum(dks)
    ls = np.zeros(len(terms) - 1, dtype=int)
    for i in range(len(terms) - 1):
        ls[i] = int(dks[i] / tot_k * kPts)
        if i == len(terms) - 2 and endpoint:
            ls[i] += 1
    kPts = ls.sum() + 1
    res = np.zeros((kPts, 2))
    for i in range(len(terms) - 1):
        for p in range(ls[i]):
            ind = p if i == 0 else p + ls[:i].sum()
            res[ind] = dic_Kpts[terms[i]] + (dic_Kpts[terms[i + 1]] - dic_Kpts[terms[i]]) / ls[i] * p
        if i == len(terms) - 2:
            res[-1] = dic_Kpts[terms[i + 1]]
    if returnNorm:
        norm = np.zeros(kPts)
        for i in range(1, kPts):
            norm[i] = norm[i - 1] + np.linalg.norm(res[i] - res[i - 1])
        return res, norm
    return res
