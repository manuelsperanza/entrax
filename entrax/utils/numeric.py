import numpy as np
import numpy.typing as npt
from functools import reduce
from math import gcd, ceil, log10
from typing import Sequence, cast

def float_gcd(a:float, b:float, tol:float=1e-9)->float:
    if tol <= 0:
        raise ValueError("Tolerance must be positive")
    
    # Calculate scaling factor from tolerance
    try:
        exponent = -round(log10(tol))
        factor = 10 ** exponent
    except ValueError:
        raise ValueError("Tolerance must be a power of 10 (e.g., 1e-3, 1e-6)") from None

    # Scale and convert to integers
    scaled_a = round(a * factor)
    scaled_b = round(b * factor)
    
    # Compute GCD of scaled integers
    common_divisor = gcd(abs(scaled_a), abs(scaled_b))
    
    # Return scaled-back result
    return float(common_divisor / factor)

def float_gcd_reduce(arr: Sequence[float])->float:
    return reduce(lambda a, b: float_gcd(a, b), arr)

def linespace_extra(points:npt.NDArray[np.float64], n:int)->Sequence[float]:
    if len(points) < 2:
        raise ValueError("not enough points")
    points.sort()
    if len(points) == 2:
        return cast(Sequence[float], np.linspace(points[0], points[1], n).tolist())
    else:
        segments = np.abs(np.diff(points))
        length = np.abs(points[-1]-points[0])
        unit = float_gcd_reduce(segments)
        unit /= ceil((n * unit) / length)

    return cast(Sequence[float], np.arange(points[0], points[-1]+unit, unit).tolist())