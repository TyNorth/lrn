"""
latmath.core.isqrt

Exact integer square root utilities for LATPy.
Pure stdlib, deterministic, and safe for sovereign arithmetic.
"""

from __future__ import annotations

from .errors import DomainError

__all__ = ["isqrt", "is_square"]


def isqrt(n: int) -> int:
    """
    Return floor(sqrt(n)) for non-negative integers n.

    Raises DomainError if n < 0.
    """
    n = int(n)
    if n < 0:
        raise DomainError("isqrt: n must be non-negative")
    if n < 2:
        return n

    # Newton iteration in integers. Monotone decreasing after init.
    x = 1 << ((n.bit_length() + 1) >> 1)  # initial guess: 2^ceil(bitlen/2)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def is_square(n: int) -> bool:
    """Return True iff n is a perfect square (n >= 0)."""
    n = int(n)
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n
