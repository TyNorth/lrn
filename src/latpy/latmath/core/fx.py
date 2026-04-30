"""
latmath.core.fx

Fixed-point helpers for LATPy.
Values are represented as integers scaled by `scale`.
Example: scale=1000, value=1234 represents 1.234
"""

from __future__ import annotations

from .errors import DomainError

__all__ = [
    "fp_from_int", "fp_to_int",
    "fp_add", "fp_sub",
    "fp_mul", "fp_div",
]


def fp_from_int(x: int, scale: int) -> int:
    x = int(x)
    scale = int(scale)
    if scale <= 0:
        raise DomainError("fp_from_int: scale must be positive")
    return x * scale


def fp_to_int(x_fp: int, scale: int) -> int:
    """Convert fixed-point integer to int via truncation toward zero."""
    x_fp = int(x_fp)
    scale = int(scale)
    if scale <= 0:
        raise DomainError("fp_to_int: scale must be positive")
    if x_fp >= 0:
        return x_fp // scale
    return -((-x_fp) // scale)


def fp_add(a_fp: int, b_fp: int) -> int:
    return int(a_fp) + int(b_fp)


def fp_sub(a_fp: int, b_fp: int) -> int:
    return int(a_fp) - int(b_fp)


def fp_mul(a_fp: int, b_fp: int, scale: int) -> int:
    """
    Multiply two fixed-point values with the same scale.
    Returns fixed-point with the same scale.
    """
    a_fp = int(a_fp)
    b_fp = int(b_fp)
    scale = int(scale)
    if scale <= 0:
        raise DomainError("fp_mul: scale must be positive")
    return (a_fp * b_fp) // scale


def fp_div(a_fp: int, b_fp: int, scale: int) -> int:
    """
    Divide two fixed-point values with the same scale.
    Returns fixed-point with the same scale.
    """
    a_fp = int(a_fp)
    b_fp = int(b_fp)
    scale = int(scale)
    if scale <= 0:
        raise DomainError("fp_div: scale must be positive")
    if b_fp == 0:
        raise DomainError("fp_div: division by zero")
    return (a_fp * scale) // b_fp
