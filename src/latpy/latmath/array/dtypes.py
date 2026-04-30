"""
latmath.array.dtypes

Minimal dtype system for LATPy v0.0.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import DTypeError

__all__ = ["DType", "I64", "F64", "B1", "parse_dtype", "dtype_of_scalar", "promote"]


@dataclass(frozen=True, slots=True)
class DType:
    name: str
    code: str  # stdlib array typecode
    size: int # size in bytes

    def __repr__(self) -> str:
        return f"DType({self.name})"

    def __reduce__(self):
        return (parse_dtype, (self.name,))


I64 = DType("i64", "q", 8)  # signed 64-bit
F64 = DType("f64", "d", 8)  # double
B1 = DType("b1", "b", 1)   # signed char (0/1) for masks


def parse_dtype(dtype: str | DType | None) -> DType:
    if dtype is None:
        raise DTypeError("dtype is required here")
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        d = dtype.strip().lower()
        if d in ("i64", "int64", "q"):
            return I64
        if d in ("f64", "float64", "d"):
            return F64
        if d in ("b1", "bool", "b"): # Added support for b1, bool, and 'b' typecode
            return B1
    raise DTypeError(f"Unsupported dtype: {dtype!r}")


def dtype_of_scalar(x: object) -> DType:
    # treat bool as int (v0.0.1)
    if isinstance(x, (bool, int)):
        return I64
    if isinstance(x, float):
        return F64
    raise DTypeError(f"Unsupported scalar type: {type(x)!r}")


def promote(a: DType, b: DType) -> DType:
    if a == b:
        return a
    if a == F64 or b == F64:
        return F64
    raise DTypeError(f"Cannot promote dtypes {a} and {b}")
