"""
latmath.core.iops

Integer operations for LATPy. Pure stdlib, deterministic, audit-friendly.
"""

from __future__ import annotations

from .errors import DomainError

__all__ = ["gcd", "egcd", "lcm", "modinv", "mul_div", "clamp"]


def gcd(a: int, b: int) -> int:
    """Greatest common divisor. Always returns a non-negative integer."""
    a = int(a)
    b = int(b)
    a = -a if a < 0 else a
    b = -b if b < 0 else b
    while b:
        a, b = b, a % b
    return a


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended GCD.

    Returns (g, x, y) such that: a*x + b*y = g, where g = gcd(a, b) >= 0.
    """
    a = int(a)
    b = int(b)

    # Handle the degenerate case explicitly.
    if a == 0 and b == 0:
        return (0, 0, 0)

    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    g = old_r
    x = old_s
    y = old_t

    # Normalize sign so g is non-negative.
    if g < 0:
        g = -g
        x = -x
        y = -y

    return (g, x, y)


def lcm(a: int, b: int) -> int:
    """Least common multiple. Always returns a non-negative integer."""
    a = int(a)
    b = int(b)
    if a == 0 or b == 0:
        return 0
    return (abs(a) // gcd(a, b)) * abs(b)


def modinv(a: int, m: int) -> int:
    """
    Modular inverse: returns x such that (a*x) % m == 1.

    Raises DomainError if inverse does not exist or modulus is invalid.
    """
    a = int(a)
    m = int(m)
    if m == 0:
        raise DomainError("modinv: modulus m must be non-zero")
    if m < 0:
        m = -m

    a = a % m
    g, x, _ = egcd(a, m)
    if g != 1:
        raise DomainError(f"modinv: {a} has no inverse modulo {m} (gcd={g})")
    return x % m


def mul_div(a: int, b: int, c: int) -> int:
    """
    Exact integer division of a*b by c using floor division.

    Raises DomainError if c == 0.
    """
    a = int(a)
    b = int(b)
    c = int(c)
    if c == 0:
        raise DomainError("mul_div: divisor c must be non-zero")
    return (a * b) // c


def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into the inclusive range [lo, hi]."""
    x = int(x)
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        lo, hi = hi, lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
