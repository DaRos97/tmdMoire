"""Path resolution, repo root detection, filename generation."""
import os
import shutil
import subprocess
import numpy as np


SOURCE_CONFIG = "Inputs/monolayer_fitting/fit_config.json"
"""Default path to the fit configuration file."""


def get_repo_root() -> str:
    """Determine the repository root directory using git.

    Returns
    -------
    str
        Absolute path to the repository root.

    Raises
    ------
    RuntimeError
        If the current directory is not inside a git repository.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def prepare_run_dir(run_id: str, material: str) -> str:
    """Create the run output directory and copy fit_config.json into it.

    Creates ``Data/<material>_run_<run_id>/`` if it does not exist, and
    copies ``Inputs/monolayer_fitting/fit_config.json`` into it (only if the destination
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
    dst = os.path.join(run_dir, "fit_config.json")
    if not os.path.exists(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    elif os.path.getmtime(SOURCE_CONFIG) > os.path.getmtime(dst):
        shutil.copy2(SOURCE_CONFIG, dst)
    return run_dir


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
