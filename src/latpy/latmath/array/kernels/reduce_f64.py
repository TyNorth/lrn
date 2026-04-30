"""
latmath.array.kernels.reduce_f64

Reductions for f64 with axis support.
"""

from __future__ import annotations

from ...core.errors import DomainError, ShapeError
from ..layout import is_contiguous_c, numel

__all__ = ["sum_f64", "min_f64", "max_f64"]


def sum_f64(A, axis=None):
    if axis is None:
        n = numel(A.shape)
        if n == 0: return 0.0
        if is_contiguous_c(A.shape, A.strides):
            buf, off, s = A._buf, A.offset, 0.0
            for i in range(n): s += buf[off + i]
            return s
        return _sum_all_strided(A)
    return _reduce_axis(A, axis, op="sum")


def min_f64(A, axis=None):
    if axis is None:
        n = numel(A.shape)
        if n == 0: raise DomainError("min: empty array")
        if is_contiguous_c(A.shape, A.strides):
            buf, off = A._buf, A.offset
            m = buf[off]
            for i in range(1, n):
                v = buf[off + i]
                if v < m: m = v
            return m
        return _minmax_all_strided(A, want="min")
    return _reduce_axis(A, axis, op="min")


def max_f64(A, axis=None):
    if axis is None:
        n = numel(A.shape)
        if n == 0: raise DomainError("max: empty array")
        if is_contiguous_c(A.shape, A.strides):
            buf, off = A._buf, A.offset
            m = buf[off]
            for i in range(1, n):
                v = buf[off + i]
                if v > m: m = v
            return m
        return _minmax_all_strided(A, want="max")
    return _reduce_axis(A, axis, op="max")


def _sum_all_strided(A) -> float:
    shape, strides, ndim, buf, base = A.shape, A.strides, A.ndim, A._buf, A.offset
    s, idx = 0.0, [0] * ndim
    while True:
        off = base
        for ax in range(ndim): off += idx[ax] * strides[ax]
        s += buf[off]
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]: break
            idx[ax] = 0; ax -= 1
        if ax < 0: break
    return s


def _minmax_all_strided(A, want: str) -> float:
    shape, strides, ndim, buf, base = A.shape, A.strides, A.ndim, A._buf, A.offset
    m, idx = None, [0] * ndim
    while True:
        off = base
        for ax in range(ndim): off += idx[ax] * strides[ax]
        v = buf[off]
        if m is None: m = v
        elif want == "min":
            if v < m: m = v
        else:
            if v > m: m = v
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]: break
            idx[ax] = 0; ax -= 1
        if ax < 0: break
    return m


def _reduce_axis(A, axis, op: str):
    from ..ndarray import zeros
    ndim = A.ndim
    if not isinstance(axis, int):
        raise ShapeError("axis must be an int")
    if axis < 0: axis += ndim
    if not (0 <= axis < ndim):
        raise ShapeError(f"axis {axis} out of bounds")

    new_shape = list(A.shape)
    n_reduce = new_shape.pop(axis)
    out = zeros(tuple(new_shape), dtype=A.dtype)
    
    if ndim == 2:
        n0, n1 = A.shape
        s0, s1 = A.strides
        buf, base = A._buf, A.offset
        out_buf, out_base = out._buf, out.offset
        if axis == 0:
            for j in range(n1):
                out_buf[out_base + j] = buf[base + j*s1]
            if n_reduce > 1:
                for i in range(1, n0):
                    row_off = base + i*s0
                    for j in range(n1):
                        v = buf[row_off + j*s1]
                        o_idx = out_base + j
                        if op == "sum": out_buf[o_idx] += v
                        elif op == "min":
                            if v < out_buf[o_idx]: out_buf[o_idx] = v
                        else:
                            if v > out_buf[o_idx]: out_buf[o_idx] = v
            return out
        else:
            for i in range(n0):
                row_off = base + i*s0
                val = buf[row_off]
                for j in range(1, n1):
                    v = buf[row_off + j*s1]
                    if op == "sum": val += v
                    elif op == "min":
                        if v < val: val = v
                    else:
                        if v > val: val = v
                out_buf[out_base + i] = val
            return out

    out_shape = out.shape
    out_strides = out.strides
    out_ndim = out.ndim
    buf, base = A._buf, A.offset
    out_buf, out_base = out._buf, out.offset
    stride_reduce = A.strides[axis]

    idx_out = [0] * out_ndim if out_ndim > 0 else []
    while True:
        a_off_base = base
        o_off = out_base
        for i in range(out_ndim):
            a_dim = i if i < axis else i + 1
            a_off_base += idx_out[i] * A.strides[a_dim]
            o_off += idx_out[i] * out_strides[i]
        
        val = buf[a_off_base]
        for k in range(1, n_reduce):
            v = buf[a_off_base + k * stride_reduce]
            if op == "sum": val += v
            elif op == "min":
                if v < val: val = v
            else:
                if v > val: val = v
        out_buf[o_off] = val

        if out_ndim == 0: break
        ax = out_ndim - 1
        while ax >= 0:
            idx_out[ax] += 1
            if idx_out[ax] < out_shape[ax]: break
            idx_out[ax] = 0; ax -= 1
        if ax < 0: break
    
    return out
