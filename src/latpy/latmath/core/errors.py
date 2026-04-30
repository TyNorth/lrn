"""
latmath.core.errors

Stable exception hierarchy for the latpy library.
"""

__all__ = ["LATMathError", "DomainError", "ShapeError", "DTypeError"]

class LATMathError(Exception):
    """Base class for all latpy mathematics errors."""
    pass

class DomainError(LATMathError):
    """Raised when a value is outside the allowed mathematical domain (e.g. sqrt(-1))."""
    pass

class ShapeError(LATMathError):
    """Raised when array shapes are incompatible for an operation."""
    pass

class DTypeError(LATMathError):
    """Raised when data types are incompatible or unsupported."""
    pass
