import numpy as np


def np2scalar(x, dtype=float):
    """
    Convert any scalar numpy value to scalar with type dtype,
    while handling regular Python values.

    Args:
        x: A value that could be either a NumPy array/scalar or a regular Python value

    Returns:
        scalar [float|int]: The dtype representation of the input value

    Raises:
        ValueError: If input is a non-scalar NumPy array
    """
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise ValueError("Input array must be scalar (size 1)")

        return dtype(x.item())

    if isinstance(x, (np.number, np.bool_)):
        return dtype(x.item())

    return dtype(x)
