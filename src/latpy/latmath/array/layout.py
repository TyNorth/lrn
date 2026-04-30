"""
latmath.array.layout

Shape/stride helpers. Strides are in ELEMENTS (not bytes).
Row-major (C-order) only for v0.0.1.
"""

from __future__ import annotations

from ..core.errors import ShapeError

__all__ = [
    "prod",
    "normalize_shape",
    "numel",
    "default_axes",
    "c_strides",
    "is_contiguous_c",
    "broadcast_shape",
]


def prod(xs: tuple[int, ...]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def normalize_shape(shape: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        shape_t = (shape,)
    else:
        shape_t = tuple(int(s) for s in shape)

    if len(shape_t) == 0:
        raise ShapeError("shape must have at least 1 dimension")
    for s in shape_t:
        if s < 0:
            raise ShapeError("shape dimensions must be >= 0")
    return shape_t


def numel(shape: tuple[int, ...]) -> int:
    return prod(shape)


def default_axes(ndim: int) -> tuple[str, ...]:
    return tuple(f"a{i}" for i in range(ndim))


def c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    # strides in elements, row-major
    ndim = len(shape)
    if ndim == 0:
        return ()
    strides = [0] * ndim
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return tuple(strides)


def is_contiguous_c(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    return strides == c_strides(shape)


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compute NumPy-style broadcasted shape for two shapes.
    Raises ShapeError if incompatible.
    """
    a = tuple(int(x) for x in a)
    b = tuple(int(x) for x in b)
    na = len(a)
    nb = len(b)
    n = na if na > nb else nb
    out = [1] * n

    for i in range(1, n + 1):
        da = a[-i] if i <= na else 1
        db = b[-i] if i <= nb else 1
        if da == db:
            out[-i] = da
        elif da == 1:
            out[-i] = db
        elif db == 1:
            out[-i] = da
        else:
            raise ShapeError(f"broadcast_shape: incompatible dims {da} vs {db}")
    return tuple(out)
