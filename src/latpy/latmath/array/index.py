"""
latmath.array.index

Indexing and assignment for NDArray v0.0.1.

Supported selectors:
- int
- slice
- tuple of (int|slice, int|slice) for 2D

Semantics:
- int reduces a dimension
- slices return views (share buffer)
- assignment supports scalar fill for selected region(s)
"""

from __future__ import annotations

from ..core.errors import DomainError, ShapeError

__all__ = ["normalize_key", "getitem", "setitem"]


def normalize_key(key, ndim: int):
    if key is None:
        return (slice(None),) * ndim
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) > ndim:
        raise ShapeError(f"Index arity mismatch: got {len(key)} indices for ndim={ndim}")
    if len(key) < ndim:
        # pad with :
        key = key + (slice(None),) * (ndim - len(key))
    for k in key:
        if not isinstance(k, (int, slice)):
            raise ShapeError(f"Unsupported index type: {type(k)!r}")
    return key


def _slice_params(s: slice, dim: int) -> tuple[int, int, int, int]:
    step = 1 if s.step is None else int(s.step)
    if step == 0:
        raise DomainError("slice step cannot be 0")
    start = 0 if s.start is None else int(s.start)
    stop = dim if s.stop is None else int(s.stop)

    # normalize negative
    if start < 0:
        start += dim
    if stop < 0:
        stop += dim

    # clamp
    if step > 0:
        if start < 0:
            start = 0
        if start > dim:
            start = dim
        if stop < 0:
            stop = 0
        if stop > dim:
            stop = dim
        length = 0 if stop <= start else (stop - start + step - 1) // step
    else:
        # negative step slicing; v0.0.1: allow, but length calc is slightly different
        if start < -1:
            start = -1
        if start >= dim:
            start = dim - 1
        if stop < -1:
            stop = -1
        if stop >= dim:
            stop = dim - 1
        length = 0 if stop >= start else (start - stop + (-step) - 1) // (-step)

    return start, stop, step, length


def getitem(A, key):
    key = normalize_key(key, A.ndim)

    # Scalar access fast path for full int indexing
    if all(isinstance(k, int) for k in key):
        off = A.offset
        for ax, k in enumerate(key):
            dim = A.shape[ax]
            idx = int(k)
            if idx < 0:
                idx += dim
            if idx < 0 or idx >= dim:
                raise ShapeError("index out of bounds")
            off += idx * A.strides[ax]
        return A._buf[off]

    # Build a view
    new_shape = []
    new_strides = []
    off = A.offset
    new_axes = []

    for ax, k in enumerate(key):
        dim = A.shape[ax]
        stride = A.strides[ax]
        axis_name = A.axes[ax]

        if isinstance(k, int):
            idx = int(k)
            if idx < 0:
                idx += dim
            if idx < 0 or idx >= dim:
                raise ShapeError("index out of bounds")
            off += idx * stride
            # axis reduced
            continue

        # slice
        start, _stop, step, length = _slice_params(k, dim)
        off += start * stride
        new_shape.append(length)
        new_strides.append(stride * step)
        new_axes.append(axis_name)

    return A._view(tuple(new_shape), tuple(new_strides), off, tuple(new_axes))


def setitem(A, key, value):
    """
    Scalar fill assignment for supported indices.
    - If key selects a scalar: assign to that element.
    - If key selects a view: fill selected region with scalar `value`.
    """
    key = normalize_key(key, A.ndim)

    # scalar element assignment
    if all(isinstance(k, int) for k in key):
        off = A.offset
        for ax, k in enumerate(key):
            dim = A.shape[ax]
            idx = int(k)
            if idx < 0:
                idx += dim
            if idx < 0 or idx >= dim:
                raise ShapeError("index out of bounds")
            off += idx * A.strides[ax]
        A._buf[off] = value
        return

    V = getitem(A, key)  # view
    if V.ndim == 0:
        # shouldn't happen with our normalization, but keep safe
        A._buf[V.offset] = value
        return

    # Fill by iterating over indices (v0.0.1 correctness > fancy speed here)
    # Kernels can optimize later.
    _fill_view(V, value)


def _fill_view(V, value):
    shape = V.shape
    strides = V.strides
    ndim = V.ndim
    base = V.offset
    buf = V._buf

    if ndim == 1:
        n = shape[0]
        st0 = strides[0]
        off = base
        for _ in range(n):
            buf[off] = value
            off += st0
        return

    if ndim == 2:
        n0, n1 = shape
        st0, st1 = strides
        off0 = base
        for _ in range(n0):
            off = off0
            for _ in range(n1):
                buf[off] = value
                off += st1
            off0 += st0
        return

    # generic ndim (rare in v0.0.1)
    idx = [0] * ndim
    while True:
        off = base
        for ax in range(ndim):
            off += idx[ax] * strides[ax]
        buf[off] = value

        # increment
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
