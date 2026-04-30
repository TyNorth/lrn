"""
latmath.core.rat

Exact rational arithmetic for LATPy (sovereign-friendly).
Pure stdlib, deterministic, fully auditable.

Normalization rules:
- den != 0
- den > 0 (sign carried by numerator)
- fraction always reduced by gcd(|num|, den)
- 0/x normalized to 0/1
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import DomainError
from .iops import gcd

__all__ = ["Rational"]


@dataclass(frozen=True, slots=True)
class Rational:
    """
    Exact rational number n/d with canonical normalization.

    Use Rational(n, d) or Rational.from_int(n).
    """
    num: int
    den: int

    def __post_init__(self) -> None:
        n = int(self.num)
        d = int(self.den)

        if d == 0:
            raise DomainError("Rational: denominator must be non-zero")

        # Normalize sign to denominator > 0
        if d < 0:
            n = -n
            d = -d

        # Normalize zero to 0/1
        if n == 0:
            object.__setattr__(self, "num", 0)
            object.__setattr__(self, "den", 1)
            return

        g = gcd(n, d)
        n //= g
        d //= g

        object.__setattr__(self, "num", n)
        object.__setattr__(self, "den", d)

    @staticmethod
    def from_int(n: int) -> "Rational":
        return Rational(int(n), 1)

    # ---------
    # Helpers
    # ---------
    def to_float(self) -> float:
        return self.num / self.den

    def __int__(self) -> int:
        # Truncate toward zero (Python default for int(float))
        if self.num >= 0:
            return self.num // self.den
        else:
            return -((-self.num) // self.den)

    def __neg__(self) -> "Rational":
        return Rational(-self.num, self.den)

    def __abs__(self) -> "Rational":
        return Rational(abs(self.num), self.den)

    # -------------
    # Comparisons
    # -------------
    def _cmp_key(self) -> tuple[int, int]:
        return (self.num, self.den)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rational):
            return self.num == other.num and self.den == other.den
        if isinstance(other, int):
            return self.den == 1 and self.num == other
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return self.num * o.den < o.num * self.den

    def __le__(self, other: object) -> bool:
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return self.num * o.den <= o.num * self.den

    def __gt__(self, other: object) -> bool:
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return self.num * o.den > o.num * self.den

    def __ge__(self, other: object) -> bool:
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return self.num * o.den >= o.num * self.den

    # -------------
    # Arithmetic
    # -------------
    def __add__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return Rational(self.num * o.den + o.num * self.den, self.den * o.den)

    def __radd__(self, other: object) -> "Rational":
        return self.__add__(other)

    def __sub__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return Rational(self.num * o.den - o.num * self.den, self.den * o.den)

    def __rsub__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return Rational(o.num * self.den - self.num * o.den, o.den * self.den)

    def __mul__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        return Rational(self.num * o.num, self.den * o.den)

    def __rmul__(self, other: object) -> "Rational":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        if o.num == 0:
            raise DomainError("Rational: division by zero")
        return Rational(self.num * o.den, self.den * o.num)

    def __rtruediv__(self, other: object) -> "Rational":
        o = _coerce_rat(other)
        if o is NotImplemented:
            return NotImplemented
        if self.num == 0:
            raise DomainError("Rational: division by zero")
        return Rational(o.num * self.den, o.den * self.num)

    def __repr__(self) -> str:
        return f"Rational({self.num}, {self.den})"


def _coerce_rat(x: object) -> Rational | object:
    if isinstance(x, Rational):
        return x
    if isinstance(x, int):
        return Rational.from_int(x)
    return NotImplemented
