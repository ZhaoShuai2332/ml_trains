"""
Module: dgk_comparator

Simulates the DGK comparison protocol using modular arithmetic.
Integrates a dedicated ModInt class for safe, efficient operations under a prime modulus.

References
----------
[1] Dai, W., Lin, C., & Song, D. (2010). DGK: Efficient and Correct Comparison for Secure Protocols. Crypto'10.

Author: Shuai Zhao
Date: 2025-06-18
"""
import random
from typing import List, Tuple, Optional


class ModInt:
    """
    Represents an integer modulo a global prime modulus.

    Supports addition, subtraction, multiplication, exponentiation,
    and modular inversion via Fermat's little theorem.
    """
    __modulus = 10**9 + 7  # Default prime modulus

    @classmethod
    def set_modulus(cls, modulus: int):
        """
        Set the global modulus for all ModInt instances.

        Parameters
        ----------
        modulus : int
            A positive integer to serve as the modulus.

        Raises
        ------
        ValueError
            If modulus is not a positive integer.
        """
        if modulus <= 0:
            raise ValueError("Modulus must be positive")
        cls.__modulus = modulus

    @classmethod
    def get_modulus(cls) -> int:
        """
        Retrieve the current global modulus.

        Returns
        -------
        int
            The modulus used for all ModInt arithmetic.
        """
        return cls.__modulus

    def __init__(self, value: int = 0):
        """
        Initialize a ModInt by reducing value into [0, modulus).

        Parameters
        ----------
        value : int, optional
            The integer to wrap; defaults to zero.
        """
        self.__value = value % ModInt.__modulus
        # Ensure non-negative representative
        if self.__value < 0:
            self.__value += ModInt.__modulus

    def __repr__(self):
        return f"ModInt({self.__value} mod {ModInt.__modulus})"

    def __str__(self):
        return str(self.__value)

    @property
    def value(self) -> int:
        """
        Get the underlying integer value.
        """
        return self.__value

    def __add__(self, other):
        """
        Modular addition with another ModInt or integer.
        """
        if isinstance(other, ModInt):
            return ModInt(self.__value + other.__value)
        if isinstance(other, int):
            return ModInt(self.__value + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Modular subtraction with another ModInt or integer.
        """
        if isinstance(other, ModInt):
            return ModInt(self.__value - other.__value)
        if isinstance(other, int):
            return ModInt(self.__value - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, int):
            return ModInt(other - self.__value)
        return NotImplemented

    def __mul__(self, other):
        """
        Modular multiplication with another ModInt or integer.
        """
        if isinstance(other, ModInt):
            return ModInt(self.__value * other.__value)
        if isinstance(other, int):
            return ModInt(self.__value * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent: int):
        """
        Modular exponentiation via built-in pow with modulus.
        """
        return ModInt(pow(self.__value, exponent, ModInt.__modulus))

    def inverse(self):
        """
        Compute multiplicative inverse using Fermat's little theorem.

        Returns
        -------
        ModInt
            The inverse element satisfying x * x.inverse() = 1.

        Raises
        ------
        ZeroDivisionError
            If attempting to invert zero.
        """
        if self.__value == 0:
            raise ZeroDivisionError("Cannot invert zero")
        # exponent = modulus-2 yields inverse modulo prime
        return ModInt(pow(self.__value, ModInt.__modulus - 2, ModInt.__modulus))

    def __truediv__(self, other):
        """
        Modular division by another ModInt or integer.
        """
        if isinstance(other, ModInt):
            return self * other.inverse()
        if isinstance(other, int):
            return self * ModInt(other).inverse()
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, int):
            return ModInt(other) * self.inverse()
        return NotImplemented

    def __eq__(self, other):
        """Equality comparison modulo the global modulus."""
        if isinstance(other, ModInt):
            return self.__value == other.__value
        if isinstance(other, int):
            return self.__value == (other % ModInt.__modulus)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class DGKComparator:
    """
    Implements the DGK secure comparison protocol for integers.

    The protocol allows two parties to compare secret-shared bits
    without revealing the actual values.
    """
    def __init__(self, bit_length: Optional[int] = None, modulus: Optional[int] = None):
        """
        Initialize the comparator with optional bit length and modulus.

        Parameters
        ----------
        bit_length : int, optional
            Number of bits for decomposition. Defaults to None, set at first compare call.
        modulus : int, optional
            Prime modulus for ModInt operations; if provided, overrides default.
        """
        self.bit_length = bit_length
        if modulus is not None:
            ModInt.set_modulus(modulus)

    def share_bits(self, m: int) -> Tuple[List[ModInt], List[ModInt]]:
        """
        Secret-share each bit of m into additive shares a and b.

        Each bit is split as a random ModInt a_i and computed b_i = bit - a_i.

        Parameters
        ----------
        m : int
            Integer to secret-share.

        Returns
        -------
        a_shares, b_shares : list of ModInt
            Two lists of ModInt shares whose sum reconstructs the bit vector.
        """
        # Decompose m into binary bits
        bits = [(m >> i) & 1 for i in range(self.bit_length)]
        a_shares, b_shares = [], []
        for bit in bits:
            # Random mask share
            a = ModInt(random.randrange(ModInt.get_modulus()))
            # Complementary share yields the true bit
            b = ModInt(bit) - a
            a_shares.append(a)
            b_shares.append(b)
        return a_shares, b_shares

    def bit_decompose(self, x: int) -> List[int]:
        """
        Decompose integer x into its binary representation list.

        Parameters
        ----------
        x : int
            Integer to decompose.

        Returns
        -------
        List[int]
            Bits of x least-significant bit first.
        """
        return [(x >> i) & 1 for i in range(self.bit_length)]

    def compute_w_shares(
        self,
        a: List[ModInt],
        b: List[ModInt],
        x_bits: List[int]
    ) -> Tuple[List[ModInt], List[ModInt]]:
        """
        Compute masked difference shares for each bit.

        w_i = a_i + x_i - 2 * a_i * x_i, and similarly for b_i.

        Parameters
        ----------
        a, b : List[ModInt]
            Secret shares of m's bits.
        x_bits : List[int]
            Plaintext bits of x.

        Returns
        -------
        alpha_w, beta_w : List[ModInt]
            Masked bitwise difference shares.
        """
        alpha_w, beta_w = [], []
        for ai, bi, xi in zip(a, b, x_bits):
            alpha_w.append(ai + xi - ai * (2 * xi))
            beta_w.append(bi - bi * (2 * xi))
        return alpha_w, beta_w

    def compute_c_shares(
        self,
        a: List[ModInt],
        b: List[ModInt],
        x_bits: List[int],
        alpha_w: List[ModInt],
        beta_w: List[ModInt]
    ) -> Tuple[List[ModInt], List[ModInt]]:
        """
        Aggregate suffix sums and compute carry shares.

        Uses suffix sums of w-shares to form c-shares that
        enable comparison reconstruction.
        """
        l = len(a)
        # Initialize suffix accumulators
        suffix_alpha = [ModInt(0)] * (l + 1)
        suffix_beta = [ModInt(0)] * (l + 1)
        for i in range(l - 1, -1, -1):
            suffix_alpha[i] = alpha_w[i] + suffix_alpha[i + 1]
            suffix_beta[i] = beta_w[i] + suffix_beta[i + 1]

        alpha_c, beta_c = [], []
        for i in range(l):
            xi = ModInt(x_bits[i])
            # Compute share for the protocol's carry term
            alpha_c.append(xi - a[i] + ModInt(1) + suffix_alpha[i + 1])
            beta_c.append(ModInt(0) - b[i] + suffix_beta[i + 1])
        return alpha_c, beta_c

    def reconstruct_and_test(
        self,
        alpha_c: List[ModInt],
        beta_c: List[ModInt]
    ) -> bool:
        """
        Reconstruct masked carries and test for zero to decide comparison.

        Randomly mask non-zero carries and check if any unmasked equals zero.

        Returns
        -------
        bool
            True if m < x, False otherwise.
        """
        # Combine the two share lists
        combined = [ac + bc for ac, bc in zip(alpha_c, beta_c)]
        masked = []
        for c in combined:
            if c == 0:
                masked.append(ModInt(0))
            else:
                # Blind non-zero carries with random factor
                r = random.randrange(1, ModInt.get_modulus())
                masked.append(c * r)
        # Shuffle to hide positions, then test for any zero
        random.shuffle(masked)
        return any(m == 0 for m in masked)

    def compare(self, m: int, x: int) -> bool:
        """
        Execute the DGK protocol to compare m and x.

        Parameters
        ----------
        m, x : int
            Cleartext integers to compare.

        Returns
        -------
        bool
            True if m < x, False otherwise.
        """
        # Auto-set bit length if not provided
        if self.bit_length is None:
            self.bit_length = max(m, x).bit_length()

        # Secret-share m and decompose x
        a, b = self.share_bits(m)
        x_bits = self.bit_decompose(x)
        # Compute protocol shares
        alpha_w, beta_w = self.compute_w_shares(a, b, x_bits)
        alpha_c, beta_c = self.compute_c_shares(a, b, x_bits, alpha_w, beta_w)
        # Final reconstruction and test
        return self.reconstruct_and_test(alpha_c, beta_c)


if __name__ == "__main__":
    # Deterministic behavior for examples
    random.seed(42)
    dgk = DGKComparator()
    examples = [(5, 3), (3, 5), (7, 7)]
    for m_val, x_val in examples:
        result = dgk.compare(m_val, x_val)
        print(f"compare({m_val}, {x_val}) -> {result}")
