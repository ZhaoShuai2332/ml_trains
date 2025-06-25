# Algorithm 1: Q18.14 Fixed-Point Conversion via Mersenne-Prime Reduction

import os
import math
import numpy as np

# Constants
P = 2**31 - 1       # Modulus (Mersenne prime)
SHIFT = 14          # Number of fractional bits (Q18.14)
WIDTH = P.bit_length()  # Total bit-width = 31
THRESHOLD = 1 << (WIDTH - 1)  # 2^(WIDTH-1)


def float_to_fix(x: float) -> int:
    """
    Convert a scalar real x to a signed Q18.14 fixed-point integer modulo P.

    Parameters
    ----------
    x : float
        Finite real number input.

    Returns
    -------
    int
        Signed fixed-point integer z ∈ [−2^(WIDTH−1), 2^(WIDTH−1)−1].

    Algorithm
    ---------
    1. y ← round(x × 2^SHIFT)
    2. z ← (y mod 2^WIDTH) + ⌊y / 2^WIDTH⌋
       if z ≥ P: z ← z − P
    3. if z ≥ THRESHOLD: z ← z − 2^WIDTH
    """
    if not math.isfinite(x):
        raise ValueError("Input x must be finite.")

    # Step 1: scaling
    y = round(x * (1 << SHIFT))

    # Step 2: fast modular reduction (fold high bits into low bits)
    z = (y & P) + (y >> WIDTH)
    if z >= P:
        z -= P

    # Step 3: two's-complement mapping
    if z >= THRESHOLD:
        z -= (1 << WIDTH)

    return z


def array_to_fix(arr: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of array of floats to signed Q18.14 fixed-point mod P.

    Parameters
    ----------
    arr : np.ndarray
        Array of finite real values.

    Returns
    -------
    np.ndarray
        Array of dtype int32, same shape as `arr`, with values in [−2^(WIDTH−1), 2^(WIDTH−1)−1].
    """
    # Convert to numpy array and ensure it's numeric
    arr = np.asarray(arr)
    
    # Check if array is numeric type
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(np.float64)
        except (ValueError, TypeError):
            raise ValueError(f"Input array contains non-numeric data of type {arr.dtype}")
    
    # Check for finite values
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input array must contain only finite values.")

    # Step 1: scale & round
    y = np.rint(arr * (1 << SHIFT)).astype(np.int64)

    # Step 2: fast modular reduction
    z = (y & P) + (y >> WIDTH)
    z -= (z >= P) * P

    # Step 3: two's-complement mapping
    z = np.where(z >= THRESHOLD, z - (1 << WIDTH), z)

    return z.astype(np.int32)
