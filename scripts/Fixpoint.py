import numpy as np

class FixedPoint:
    """
    Fixed-point Q18.14 representation with proper modular arithmetic handling.

    - int_bits: total bits for integer part including sign (18)
    - frac_bits: bits for fractional part (14)
    - total_bits: sum of int_bits and frac_bits (32)
    - P: modulus for all operations (2^31 - 1)
    """
    P = 2**31 - 1  # 2147483647

    def __init__(self, value, int_bits=18, frac_bits=14, from_float=False):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.mask = (1 << self.total_bits) - 1
        self.sign_bit = 1 << (self.total_bits - 1)
        self.max_positive = (1 << (self.total_bits - 1)) - 1
        self.min_negative = -(1 << (self.total_bits - 1))

        if from_float or isinstance(value, (float, np.floating)):
            # Convert float to Q18.14 raw integer
            scaled = int(round(value * (1 << self.frac_bits)))
            # Clamp to valid range before modular reduction
            scaled = max(self.min_negative, min(self.max_positive, scaled))
            # Store as unsigned representation for modular arithmetic
            if scaled < 0:
                self.raw = (scaled + (1 << self.total_bits)) % self.P
            else:
                self.raw = scaled % self.P
        else:
            # Assume value is already a raw integer; reduce modulo P
            self.raw = int(value) % self.P

    def _signed(self):
        """Return signed integer interpretation with proper modular handling"""
        # Handle modular arithmetic properly
        if self.raw >= self.P // 2:
            # Large values close to P should be treated as negative
            return self.raw - self.P
        elif self.raw >= self.sign_bit:
            # Standard two's complement negative
            return self.raw - (1 << self.total_bits)
        else:
            # Positive value
            return self.raw

    def to_float(self):
        """Convert fixed-point value to Python float with proper handling"""
        signed_val = self._signed()
        return float(signed_val) / (1 << self.frac_bits)

    def __repr__(self):
        return (f"FixedPoint(raw=0x{self.raw:08X}, Q{self.int_bits}.{self.frac_bits}, "
                f"float={self.to_float()})")

    # Binary operators with proper modular arithmetic
    def __add__(self, other):
        """Addition operator with proper overflow handling"""
        if not isinstance(other, FixedPoint):
            return NotImplemented
        
        # Convert to signed values for proper arithmetic
        a_signed = self._signed()
        b_signed = other._signed()
        result_signed = a_signed + b_signed
        
        # Clamp result to valid range
        result_signed = max(self.min_negative, min(self.max_positive, result_signed))
        
        # Convert back to unsigned representation
        if result_signed < 0:
            raw_result = (result_signed + (1 << self.total_bits)) % self.P
        else:
            raw_result = result_signed % self.P
            
        return FixedPoint(raw_result, self.int_bits, self.frac_bits)

    def __sub__(self, other):
        """Subtraction operator with proper overflow handling"""
        if not isinstance(other, FixedPoint):
            return NotImplemented
            
        # Convert to signed values for proper arithmetic
        a_signed = self._signed()
        b_signed = other._signed()
        result_signed = a_signed - b_signed
        
        # Clamp result to valid range
        result_signed = max(self.min_negative, min(self.max_positive, result_signed))
        
        # Convert back to unsigned representation
        if result_signed < 0:
            raw_result = (result_signed + (1 << self.total_bits)) % self.P
        else:
            raw_result = result_signed % self.P
            
        return FixedPoint(raw_result, self.int_bits, self.frac_bits)

    def __mul__(self, other):
        """Multiplication operator with proper scaling and overflow handling"""
        if not isinstance(other, FixedPoint):
            return NotImplemented
            
        # Convert to signed values for proper arithmetic
        a_signed = self._signed()
        b_signed = other._signed()
        
        # Multiply and scale down by fractional bits
        result_signed = (a_signed * b_signed) >> self.frac_bits
        
        # Clamp result to valid range
        result_signed = max(self.min_negative, min(self.max_positive, result_signed))
        
        # Convert back to unsigned representation
        if result_signed < 0:
            raw_result = (result_signed + (1 << self.total_bits)) % self.P
        else:
            raw_result = result_signed % self.P
            
        return FixedPoint(raw_result, self.int_bits, self.frac_bits)

# Helper functions

def parse_float_to_fixed_array(float_array: np.ndarray,
                               int_bits: int = 18,
                               frac_bits: int = 14) -> np.ndarray:
    """
    Convert a float array to a fixed-point array using modular arithmetic under P.
    """
    flat = float_array.flatten()
    fixed_list = [FixedPoint(v, int_bits, frac_bits, from_float=True) for v in flat]
    return np.array(fixed_list, dtype=object).reshape(float_array.shape)


def parse_fixed_to_float_array(fixed_array: np.ndarray) -> np.ndarray:
    """
    Convert a fixed-point array back to floats.
    """
    flat = fixed_array.flatten()
    float_list = [v.to_float() if hasattr(v, "to_float") else float(v) for v in flat]
    return np.array(float_list, dtype=np.float32).reshape(fixed_array.shape)
