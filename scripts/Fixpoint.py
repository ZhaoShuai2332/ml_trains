import numpy as np

class FixedPoint:
    """
    Fixed-point Q20.16 representation using 32-bit two's complement.

    - int_bits: total bits for integer part including sign (20)
    - frac_bits: bits for fractional part (16)
    - total_bits: sum of int_bits and frac_bits (32)
    """
    def __init__(self, value, int_bits=20, frac_bits=16, from_float=False):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.mask = (1 << self.total_bits) - 1
        self.sign_bit = 1 << (self.total_bits - 1)

        if from_float or isinstance(value, (float, np.floating)):
            # Convert float to signed integer representation
            scaled = int(round(value * (1 << self.frac_bits)))
            # Saturate to Q20.16 limits
            min_val = -(1 << (self.total_bits - 1))
            max_val =  (1 << (self.total_bits - 1)) - 1
            if scaled < min_val:
                scaled = min_val
            elif scaled > max_val:
                scaled = max_val
            # Store as raw two's-complement bits
            self.raw = scaled & self.mask
        else:
            # Assume value is raw bits
            self.raw = value & self.mask

    def _signed(self):
        """Return signed integer interpretation of raw bits"""
        if self.raw & self.sign_bit:
            return self.raw - (1 << self.total_bits)
        return self.raw

    def to_float(self):
        """Convert fixed-point value to Python float"""
        signed = self._signed()
        return float(signed) / (1 << self.frac_bits)

    def get_raw_value(self):
        """Get raw 32-bit two's-complement value"""
        return self.raw

    def print_binary(self):
        """Print raw bits in binary"""
        bits = bin(self.raw)[2:].zfill(self.total_bits)
        print(f"{bits}  (sign+int:{self.int_bits} bits, frac:{self.frac_bits} bits)")

    def __add__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
        raw = (self.raw + other.raw) & self.mask
        return FixedPoint(raw, self.int_bits, self.frac_bits)

    def __sub__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
        raw = (self.raw - other.raw) & self.mask
        return FixedPoint(raw, self.int_bits, self.frac_bits)

    def __mul__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
        # Multiply signed values, then scale down by frac_bits
        prod = self._signed() * other._signed()
        scaled = prod >> self.frac_bits
        # Saturate to Q20.16 limits
        min_val = -(1 << (self.total_bits - 1))
        max_val =  (1 << (self.total_bits - 1)) - 1
        if scaled < min_val:
            scaled = min_val
        elif scaled > max_val:
            scaled = max_val
        raw = scaled & self.mask
        return FixedPoint(raw, self.int_bits, self.frac_bits)

    def __repr__(self):
        return (f"FixedPoint(raw=0x{self.raw:08X}, Q{self.int_bits}.{self.frac_bits}, "
                f"float={self.to_float()})")

def parse_float_to_fixed_array(float_array: np.ndarray, 
                               int_bits: int = 20, 
                               frac_bits: int = 16) -> np.ndarray:
    """
    Convert a float array to a fixed-point array.
    """
    shape = float_array.shape
    flat = float_array.flatten()
    fixed_list = [FixedPoint(v, int_bits, frac_bits, from_float=True) for v in flat]
    return np.array(fixed_list, dtype=object).reshape(shape)

def parse_fixed_to_float_array(fixed_array: np.ndarray) -> np.ndarray:
    """
    Convert a fixed-point array to a float array.
    """
    shape = fixed_array.shape
    flat = fixed_array.flatten()
    float_list = [v.to_float() for v in flat]
    return np.array(float_list, dtype=np.float32).reshape(shape)
