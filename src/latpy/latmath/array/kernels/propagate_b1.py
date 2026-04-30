"""
latmath.array.kernels.propagate_b1

Mask propagation kernels for dtype b1 (0/1 stored in array('b')).

Propagation semantics (cumulative OR along an axis):
- forward:  out[i] = out[i-1] OR in[i] (per line)
- backward: out[i] = out[i+1] OR in[i] (per line)
- both: forward then backward

Pure stdlib. Works with strided arrays (including 0-stride broadcast views).
"""

from __future__ import annotations

from ...core.errors import ShapeError, DTypeError
from ..layout import numel, is_contiguous_c

__all__ = ["propagate_b1"]


def _norm_axis(axis: int, ndim: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += ndim
    if ax < 0 or ax >= ndim:
        raise ShapeError(f"propagate: axis out of range axis={axis} ndim={ndim}")
    return ax


def propagate_b1(IN, OUT, axis: int = -1, direction: str = "forward") -> object:
    """
    Propagate mask values along `axis`.

    Parameters
    ----------
    IN : NDArray (dtype b1)
    OUT: NDArray (dtype b1), same shape as IN
    axis : int, axis along which to propagate
    direction : {"forward","backward","both"}

    Returns
    -------
    OUT (same object)

    Notes
    -----
    - Does not allocate.
    - OUT can alias IN (in-place propagation is allowed).
    """
    if IN.dtype.name != "b1" or OUT.dtype.name != "b1":
        raise DTypeError("propagate_b1: IN and OUT must be dtype b1")
    if IN.shape != OUT.shape:
        raise ShapeError("propagate_b1: shape mismatch")

    direction = str(direction).strip().lower()
    if direction not in ("forward", "backward", "both"):
        raise ShapeError("propagate_b1: direction must be 'forward', 'backward', or 'both'")

    n = numel(IN.shape)
    if n == 0:
        return OUT

    ax = _norm_axis(axis, IN.ndim)

    # Fast path: 1D
    if IN.ndim == 1:
        if direction == "forward":
            _prop_1d_forward(IN, OUT)
        elif direction == "backward":
            _prop_1d_backward(IN, OUT)
        else:
            # both (safe even if OUT==IN)
            _prop_1d_forward(IN, OUT)
            _prop_1d_backward(OUT, OUT)
        return OUT

    # 2D fast paths (common for tables)
    if IN.ndim == 2:
        if ax == 1:
            if direction == "forward":
                _prop_2d_rows_forward(IN, OUT)
            elif direction == "backward":
                _prop_2d_rows_backward(IN, OUT)
            else:
                _prop_2d_rows_forward(IN, OUT)
                _prop_2d_rows_backward(OUT, OUT)
            return OUT
        elif ax == 0:
            if direction == "forward":
                _prop_2d_cols_forward(IN, OUT)
            elif direction == "backward":
                _prop_2d_cols_backward(IN, OUT)
            else:
                _prop_2d_cols_forward(IN, OUT)
                _prop_2d_cols_backward(OUT, OUT)
            return OUT

    # Generic N-D
    if direction == "forward":
        _prop_nd(IN, OUT, ax, forward=True)
    elif direction == "backward":
        _prop_nd(IN, OUT, ax, forward=False)
    else:
        _prop_nd(IN, OUT, ax, forward=True)
        _prop_nd(OUT, OUT, ax, forward=False)
    return OUT


# -------------------------
# 1D kernels
# -------------------------

def _prop_1d_forward(IN, OUT) -> None:
    n = IN.shape[0]
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0 = IN.strides[0]
    os0 = OUT.strides[0]

    # copy + cumulative OR
    seen = 0
    i = io
    o = oo
    for _ in range(n):
        v = 1 if ib[i] else 0
        seen = 1 if (seen or v) else 0
        ob[o] = seen
        i += is0
        o += os0


def _prop_1d_backward(IN, OUT) -> None:
    n = IN.shape[0]
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0 = IN.strides[0]
    os0 = OUT.strides[0]

    seen = 0
    i = io + (n - 1) * is0
    o = oo + (n - 1) * os0
    for _ in range(n):
        v = 1 if ib[i] else 0
        seen = 1 if (seen or v) else 0
        ob[o] = seen
        i -= is0
        o -= os0


# -------------------------
# 2D kernels (axis-specific)
# -------------------------

def _prop_2d_rows_forward(IN, OUT) -> None:
    # axis=1 : propagate across columns for each row
    n0, n1 = IN.shape
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0, is1 = IN.strides
    os0, os1 = OUT.strides

    row_i = io
    row_o = oo
    for _ in range(n0):
        seen = 0
        i = row_i
        o = row_o
        for _ in range(n1):
            v = 1 if ib[i] else 0
            seen = 1 if (seen or v) else 0
            ob[o] = seen
            i += is1
            o += os1
        row_i += is0
        row_o += os0


def _prop_2d_rows_backward(IN, OUT) -> None:
    n0, n1 = IN.shape
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0, is1 = IN.strides
    os0, os1 = OUT.strides

    row_i = io
    row_o = oo
    for _ in range(n0):
        seen = 0
        i = row_i + (n1 - 1) * is1
        o = row_o + (n1 - 1) * os1
        for _ in range(n1):
            v = 1 if ib[i] else 0
            seen = 1 if (seen or v) else 0
            ob[o] = seen
            i -= is1
            o -= os1
        row_i += is0
        row_o += os0


def _prop_2d_cols_forward(IN, OUT) -> None:
    # axis=0 : propagate down rows for each column
    n0, n1 = IN.shape
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0, is1 = IN.strides
    os0, os1 = OUT.strides

    # iterate each column as a strided 1D line
    for j in range(n1):
        seen = 0
        i = io + j * is1
        o = oo + j * os1
        for _ in range(n0):
            v = 1 if ib[i] else 0
            seen = 1 if (seen or v) else 0
            ob[o] = seen
            i += is0
            o += os0


def _prop_2d_cols_backward(IN, OUT) -> None:
    n0, n1 = IN.shape
    ib = IN._buf
    ob = OUT._buf
    io = IN.offset
    oo = OUT.offset
    is0, is1 = IN.strides
    os0, os1 = OUT.strides

    for j in range(n1):
        seen = 0
        i = io + (n0 - 1) * is0 + j * is1
        o = oo + (n0 - 1) * os0 + j * os1
        for _ in range(n0):
            v = 1 if ib[i] else 0
            seen = 1 if (seen or v) else 0
            ob[o] = seen
            i -= is0
            o -= os0


# -------------------------
# Generic N-D kernel
# -------------------------

def _prop_nd(IN, OUT, ax: int, forward: bool) -> None:
    shape = IN.shape
    ndim = IN.ndim
    inner = shape[ax]

    ib = IN._buf
    ob = OUT._buf
    io0 = IN.offset
    oo0 = OUT.offset
    is_ = IN.strides
    os_ = OUT.strides

    # Iterate over all indices except ax (outer index space)
    out_nd = ndim - 1
    outer_shape = tuple(shape[i] for i in range(ndim) if i != ax)
    if out_nd == 0:
        # reduce to 1 line
        if forward:
            _prop_1d_forward(IN, OUT)
        else:
            _prop_1d_backward(IN, OUT)
        return

    idx = [0] * out_nd
    while True:
        # compute base offsets for this line
        base_i = io0
        base_o = oo0
        j = 0
        for d in range(ndim):
            if d == ax:
                continue
            k = idx[j]
            base_i += k * is_[d]
            base_o += k * os_[d]
            j += 1

        # propagate along ax
        if forward:
            seen = 0
            i = base_i
            o = base_o
            step_i = is_[ax]
            step_o = os_[ax]
            for _ in range(inner):
                v = 1 if ib[i] else 0
                seen = 1 if (seen or v) else 0
                ob[o] = seen
                i += step_i
                o += step_o
        else:
            seen = 0
            step_i = is_[ax]
            step_o = os_[ax]
            i = base_i + (inner - 1) * step_i
            o = base_o + (inner - 1) * step_o
            for _ in range(inner):
                v = 1 if ib[i] else 0
                seen = 1 if (seen or v) else 0
                ob[o] = seen
                i -= step_i
                o -= step_o

        # increment outer idx
        p = out_nd - 1
        while p >= 0:
            idx[p] += 1
            if idx[p] < outer_shape[p]:
                break
            idx[p] = 0
            p -= 1
        if p < 0:
            break


"""
from latpy.latmath.array import zeros, array
from latpy.latmath.array.dtypes import B1
from latpy.latmath.array.kernels.propagate_b1 import propagate_b1

M = array([[0,1,0,0],[0,0,0,1]], dtype="b1")
OUT = zeros(M.shape, B1, axes=M.axes)

propagate_b1(M, OUT, axis=1, direction="forward")
# OUT => [[False, True, True, True],
#         [False, False, False, True]]
"""