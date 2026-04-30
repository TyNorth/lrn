"""
latmath.array.ndarray

NDArray v0.0.1 (pure stdlib).
- Dense numeric buffers only (i64/f64)
- Row-major layout by default
- Views via (shape, strides, offset)
- Axis NAMES are first-class (labels later)
"""

from __future__ import annotations

from array import array as pyarray
from typing import Any, Iterable

from ..core.errors import ShapeError, DTypeError, DomainError
from .dtypes import DType, I64, F64, B1, parse_dtype, dtype_of_scalar, promote
from .layout import normalize_shape, c_strides, numel, default_axes, is_contiguous_c, broadcast_shape
from . import index as _index
from .kernels.elem_i64 import add_i64, mul_i64, add_scalar_i64, mul_scalar_i64, cmp_i64, cmp_scalar_i64, where_i64
from .kernels.elem_f64 import add_f64, mul_f64, add_scalar_f64, mul_scalar_f64, cmp_f64, cmp_scalar_f64, where_f64
from .kernels.reduce_i64 import sum_i64, min_i64, max_i64
from .kernels.reduce_f64 import sum_f64, min_f64, max_f64

__all__ = ["NDArray", "zeros", "array"]


class NDArray:
    __slots__ = ("_buf", "dtype", "shape", "strides", "offset", "axes")

    def __init__(self, buf, dtype: DType, shape, strides, offset: int, axes):
        self._buf = buf
        self.dtype = dtype
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.offset = int(offset)
        self.axes = tuple(axes)

    def __reduce__(self):
        # Flattened list conversion to avoid array protocol issues in older pythons
        return (array, (self.tolist(), self.dtype, self.axes))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"NDArray(shape={self.shape}, dtype={self.dtype.name}, axes={self.axes})"

    def _view(self, shape, strides, offset, axes):
        return NDArray(self._buf, self.dtype, shape, strides, offset, axes)

    def copy(self) -> "NDArray":
        # materialize into contiguous buffer
        out = zeros(self.shape, self.dtype, axes=self.axes)
        if numel(self.shape) == 0:
            return out
        if is_contiguous_c(self.shape, self.strides) and is_contiguous_c(out.shape, out.strides):
            n = numel(self.shape)
            a = self._buf
            o = out._buf
            ao = self.offset
            oo = out.offset
            for i in range(n):
                o[oo + i] = a[ao + i]
            return out

        # generic copy via index iteration (slow path ok for v0)
        # fill by using out[...] assignment elementwise
        _copy_strided(self, out)
        return out

    # ---- Indexing ----
    def __getitem__(self, key):
        # 1D boolean compression: A[mask]
        if isinstance(key, NDArray):
            return _compress_1d(self, key)
        return _index.getitem(self, key)

    def __setitem__(self, key, value):
        _index.setitem(self, key, value)

    # ---- Ops ----
    def add(self, other: "NDArray") -> "NDArray":
        return add(self, other)

    def mul(self, other: "NDArray") -> "NDArray":
        return mul(self, other)

    # ---- Comparison ----
    def __eq__(self, other) -> "NDArray": return _compare(self, other, "eq")
    def __ne__(self, other) -> "NDArray": return _compare(self, other, "ne")
    def __gt__(self, other) -> "NDArray": return _compare(self, other, "gt")
    def __ge__(self, other) -> "NDArray": return _compare(self, other, "ge")
    def __lt__(self, other) -> "NDArray": return _compare(self, other, "lt")
    def __le__(self, other) -> "NDArray": return _compare(self, other, "le")

    # ---- Reductions ----
    def sum(self, axis=None):
        return sum(self, axis=axis)

    def min(self, axis=None):
        return min(self, axis=axis)

    def max(self, axis=None):
        return max(self, axis=axis)

    # ---- Comparison ----
    def __eq__(self, other) -> "NDArray": return eq(self, other)
    def __ne__(self, other) -> "NDArray": return ne(self, other)
    def __gt__(self, other) -> "NDArray": return gt(self, other)
    def __ge__(self, other) -> "NDArray": return ge(self, other)
    def __lt__(self, other) -> "NDArray": return lt(self, other)
    def __le__(self, other) -> "NDArray": return le(self, other)

    # ---- v0.0.2 Shape/Axis Ops ----
    @property
    def T(self) -> "NDArray":
        """2D transpose shortcut."""
        if self.ndim != 2:
            raise ShapeError("T is only defined for 2D arrays in v0.0.2")
        return self.transpose(1, 0)

    def transpose(self, *perm: int) -> "NDArray":
        """
        Return a view with permuted axes.
        - If perm is empty: reverse axes
        - If perm provided: must be a permutation of range(ndim)
        """
        ndim = self.ndim
        if len(perm) == 0:
            p = tuple(range(ndim - 1, -1, -1))
        else:
            p = tuple(int(x) for x in perm)
            if len(p) != ndim:
                raise ShapeError("transpose: permutation length must equal ndim")
            if sorted(p) != list(range(ndim)):
                raise ShapeError("transpose: perm must be a permutation of axes")

        new_shape = tuple(self.shape[i] for i in p)
        new_strides = tuple(self.strides[i] for i in p)
        new_axes = tuple(self.axes[i] for i in p)
        return self._view(new_shape, new_strides, self.offset, new_axes)

    def reshape(self, new_shape) -> "NDArray":
        """
        Reshape view. v0.0.2 rule: only allowed if self is C-contiguous.
        """
        sh = normalize_shape(new_shape)
        if numel(sh) != numel(self.shape):
            raise ShapeError("reshape: total size must remain unchanged")
        if not is_contiguous_c(self.shape, self.strides):
            raise ShapeError("reshape: only supported for C-contiguous arrays in v0.0.2")

        new_strides = c_strides(sh)
        # Keep axes names if dimension count unchanged; otherwise default axes.
        if len(sh) == self.ndim:
            axes = self.axes
        else:
            axes = default_axes(len(sh))
        return self._view(sh, new_strides, self.offset, axes)

    def ravel(self) -> "NDArray":
        """Flatten to 1D view if C-contiguous; otherwise error in v0.0.2."""
        return self.reshape((numel(self.shape),))

    def flatten(self) -> "NDArray":
        """Flatten to 1D copy (materialized contiguous)."""
        return self.copy().reshape((numel(self.shape),))

    # ---- Debug conversion ----
    def tolist(self):
        # materialization for N-D
        return _tolist_rec(self)


def _tolist_rec(A: NDArray):
    if A.ndim == 1:
        n = A.shape[0]
        st = A.strides[0]
        off = A.offset
        buf = A._buf
        out = [0] * n
        is_bool = (A.dtype == B1)
        for i in range(n):
            val = buf[off + i * st]
            out[i] = bool(val) if is_bool else val
        return out
    
    # N-D slice recursion
    n0 = A.shape[0]
    out = [None] * n0
    for i in range(n0):
        out[i] = _tolist_rec(A[i])
    return out


def zeros(shape, dtype: str | DType = I64, axes: tuple[str, ...] | None = None) -> NDArray:
    dt = parse_dtype(dtype) if not isinstance(dtype, DType) else dtype
    sh = normalize_shape(shape)
    st = c_strides(sh)
    n = numel(sh)
    buf = pyarray(dt.code, [0] * n)
    ax = default_axes(len(sh)) if axes is None else tuple(axes)
    if len(ax) != len(sh):
        raise ShapeError("axes length must match ndim")
    return NDArray(buf, dt, sh, st, 0, ax)


def array(obj: Any, dtype: str | DType | None = None, axes: tuple[str, ...] | None = None) -> NDArray:
    """
    Create NDArray from nested Python lists or other NDArrays.
    Supports N-dimensional nested lists.
    """
    if isinstance(obj, NDArray):
        if dtype is None and axes is None:
            return obj
        dt = obj.dtype if dtype is None else parse_dtype(dtype)
        out = obj.copy()
        out.dtype = dt
        if axes is not None:
            out.axes = tuple(axes)
        return out
    # Support scalar conversion
    if isinstance(obj, (int, float, bool)):
        obj = [obj]

    # Flatten and infer shape/dtype
    flat = []
    shape = []
    
    def _parse(cur):
        if isinstance(cur, list):
            if not shape:
                shape.append(len(cur))
            elif len(cur) != shape[len(shape)-1]: # This simple check is only for the first level encounter
                # More robust shape check needed for ragged detection
                pass 
            
            for i, item in enumerate(cur):
                if i == 0:
                    # descent
                    _parse(item)
                else:
                    _parse(item)
        else:
            flat.append(cur)

    # Simplified shape inference for v0.0.3
    def get_shape(lst):
        if not isinstance(lst, list): return ()
        if not lst: return (0,)
        return (len(lst),) + get_shape(lst[0])

    sh = get_shape(obj)
    if not sh:
        raise ShapeError("array(): expected list or nested list")
    
    # Flatten
    flat = []
    def _flatten(lst):
        if isinstance(lst, list):
            for item in lst: _flatten(item)
        else:
            flat.append(lst)
    _flatten(obj)

    if len(flat) != numel(sh):
        raise ShapeError("array(): ragged nested list detected")

    # Inferred dtype
    inferred = I64
    for v in flat:
        inferred = promote(inferred, dtype_of_scalar(v))
    dt = inferred if dtype is None else parse_dtype(dtype)
    
    st = c_strides(sh)
    buf = pyarray(dt.code)
    if dt == I64:
        buf.extend(int(v) for v in flat)
    elif dt == B1:
        buf.extend(1 if v else 0 for v in flat)
    else:
        buf.extend(float(v) for v in flat)
    
    ax = default_axes(len(sh)) if axes is None else tuple(axes)
    if len(ax) != len(sh):
        raise ShapeError("axes length must match ndim")
    return NDArray(buf, dt, sh, st, 0, ax)


def _scalar_array(x, dt: DType) -> "NDArray":
    sh = (1,)
    st = c_strides(sh)
    buf = pyarray(dt.code, [0])
    if dt == I64:
        buf[0] = int(x)
    elif dt == B1:
        buf[0] = 1 if x else 0
    else:
        buf[0] = float(x)
    ax = ("s",)
    return NDArray(buf, dt, sh, st, 0, ax)


def add(A: NDArray, B) -> NDArray:
    # scalar handling
    if isinstance(B, (int, float)):
        dt = promote(A.dtype, dtype_of_scalar(B))
        out = zeros(A.shape, dt, axes=A.axes)
        if dt == I64:
            add_scalar_i64(A if A.dtype == I64 else A.astype(I64), int(B), out)
        else:
            add_scalar_f64(A if A.dtype == F64 else A.astype(F64), float(B), out)
        return out

    if not isinstance(B, NDArray):
        raise DTypeError("add: B must be NDArray or scalar")

    dt = promote(A.dtype, B.dtype)
    out_shape = broadcast_shape(A.shape, B.shape)

    # Choose output axes: prefer operand that already has out_ndim
    if len(A.shape) == len(out_shape):
        out_axes = A.axes
    elif len(B.shape) == len(out_shape):
        out_axes = B.axes
    else:
        out_axes = default_axes(len(out_shape))

    out = zeros(out_shape, dt, axes=out_axes)

    Ab = A.broadcast_to(out_shape)
    Bb = B.broadcast_to(out_shape)

    if dt == I64:
        add_i64(Ab if Ab.dtype == I64 else Ab.astype(I64), Bb if Bb.dtype == I64 else Bb.astype(I64), out)
    else:
        add_f64(Ab if Ab.dtype == F64 else Ab.astype(F64), Bb if Bb.dtype == F64 else Bb.astype(F64), out)
    return out


def mul(A: NDArray, B) -> NDArray:
    # scalar handling
    if isinstance(B, (int, float)):
        dt = promote(A.dtype, dtype_of_scalar(B))
        out = zeros(A.shape, dt, axes=A.axes)
        if dt == I64:
            mul_scalar_i64(A if A.dtype == I64 else A.astype(I64), int(B), out)
        else:
            mul_scalar_f64(A if A.dtype == F64 else A.astype(F64), float(B), out)
        return out

    if not isinstance(B, NDArray):
        raise DTypeError("mul: B must be NDArray or scalar")

    dt = promote(A.dtype, B.dtype)
    out_shape = broadcast_shape(A.shape, B.shape)

    if len(A.shape) == len(out_shape):
        out_axes = A.axes
    elif len(B.shape) == len(out_shape):
        out_axes = B.axes
    else:
        out_axes = default_axes(len(out_shape))

    out = zeros(out_shape, dt, axes=out_axes)

    Ab = A.broadcast_to(out_shape)
    Bb = B.broadcast_to(out_shape)

    if dt == I64:
        mul_i64(Ab if Ab.dtype == I64 else Ab.astype(I64), Bb if Bb.dtype == I64 else Bb.astype(I64), out)
    else:
        mul_f64(Ab if Ab.dtype == F64 else Ab.astype(F64), Bb if Bb.dtype == F64 else Bb.astype(F64), out)
    return out


def sum(A: NDArray, axis=None):
    return sum_i64(A, axis=axis) if A.dtype == I64 else sum_f64(A, axis=axis)


def min(A: NDArray, axis=None):
    return min_i64(A, axis=axis) if A.dtype == I64 else min_f64(A, axis=axis)


def max(A: NDArray, axis=None):
    return max_i64(A, axis=axis) if A.dtype == I64 else max_f64(A, axis=axis)


def _require_b1(A: "NDArray", opname: str) -> None:
    if not isinstance(A, NDArray) or A.dtype != B1:
        raise DTypeError(f"{opname}: operand must be NDArray with dtype b1")


def _mask_binary(M: "NDArray", N: "NDArray", op: str) -> "NDArray":
    _require_b1(M, op)
    _require_b1(N, op)
    out_shape = broadcast_shape(M.shape, N.shape)
    
    # Choose output axes
    if len(M.shape) == len(out_shape):
        out_axes = M.axes
    elif len(N.shape) == len(out_shape):
        out_axes = N.axes
    else:
        out_axes = default_axes(len(out_shape))
        
    out = zeros(out_shape, B1, axes=out_axes)

    Mb = M.broadcast_to(out_shape)
    Nb = N.broadcast_to(out_shape)

    # Fast paths: contiguous
    n = numel(out_shape)
    if n == 0:
        return out

    if is_contiguous_c(out.shape, out.strides) and is_contiguous_c(Mb.shape, Mb.strides) and is_contiguous_c(Nb.shape, Nb.strides):
        m = Mb._buf
        nb = Nb._buf
        o = out._buf
        mo = Mb.offset
        no = Nb.offset
        oo = out.offset
        if op == "and":
            for i in range(n): o[oo+i] = 1 if (m[mo+i] and nb[no+i]) else 0
        elif op == "or":
            for i in range(n): o[oo+i] = 1 if (m[mo+i] or  nb[no+i]) else 0
        elif op == "xor":
            for i in range(n): o[oo+i] = 1 if ((m[mo+i] != 0) ^ (nb[no+i] != 0)) else 0
        else:
            raise ShapeError(f"mask op: unknown op {op!r}")
        return out

    # Generic strided (covers 0-stride broadcast views)
    shape = out.shape
    ndim = out.ndim
    m_buf, n_buf, o_buf = Mb._buf, Nb._buf, out._buf
    mo, no, oo = Mb.offset, Nb.offset, out.offset
    ms, ns, os_ = Mb.strides, Nb.strides, out.strides
    idx = [0] * ndim
    while True:
        mi = mo
        ni = no
        oi = oo
        for ax in range(ndim):
            k = idx[ax]
            mi += k * ms[ax]
            ni += k * ns[ax]
            oi += k * os_[ax]
            
        mv = 1 if m_buf[mi] else 0
        nv = 1 if n_buf[ni] else 0
        if op == "and":
            o_buf[oi] = 1 if (mv and nv) else 0
        elif op == "or":
            o_buf[oi] = 1 if (mv or nv) else 0
        elif op == "xor":
            o_buf[oi] = 1 if ((mv != 0) ^ (nv != 0)) else 0
        else:
            raise ShapeError(f"mask op: unknown op {op!r}")

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
    return out


def _mask_not(M: "NDArray") -> "NDArray":
    _require_b1(M, "not")
    out = zeros(M.shape, B1, axes=M.axes)
    n = numel(M.shape)
    if n == 0:
        return out

    if is_contiguous_c(M.shape, M.strides) and is_contiguous_c(out.shape, out.strides):
        m = M._buf
        o = out._buf
        mo = M.offset
        oo = out.offset
        for i in range(n):
            o[oo+i] = 0 if m[mo+i] else 1
        return out

    # generic strided
    shape = out.shape
    ndim = out.ndim
    m_buf, o_buf = M._buf, out._buf
    mo, oo = M.offset, out.offset
    ms, os_ = M.strides, out.strides
    idx = [0] * ndim
    while True:
        mi = mo
        oi = oo
        for ax in range(ndim):
            k = idx[ax]
            mi += k * ms[ax]
            oi += k * os_[ax]
        o_buf[oi] = 0 if m_buf[mi] else 1

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
    return out


def _compress_1d(A: "NDArray", M: "NDArray") -> "NDArray":
    if A.ndim != 1:
        raise ShapeError("A[mask]: only supported for 1D arrays in v0.0.5")
    if not isinstance(M, NDArray) or M.dtype != B1:
        raise DTypeError("A[mask]: mask must be NDArray with dtype b1")

    # Broadcast mask to A.shape (may be view with 0-strides)
    Mb = M.broadcast_to(A.shape)

    n = A.shape[0]
    if n == 0:
        return zeros((0,), A.dtype, axes=A.axes)

    # Count trues
    count = 0
    if is_contiguous_c(Mb.shape, Mb.strides):
        m = Mb._buf
        mo = Mb.offset
        for i in range(n):
            if m[mo + i]:
                count += 1
    else:
        m = Mb._buf
        mo = Mb.offset
        ms = Mb.strides[0]
        mi = mo
        for _ in range(n):
            if m[mi]:
                count += 1
            mi += ms

    out = zeros((count,), A.dtype, axes=A.axes)  # keep same axis name
    if count == 0:
        return out

    # Copy selected elements
    a = A._buf
    ao = A.offset
    ast = A.strides[0]
    o = out._buf
    oo = out.offset

    if is_contiguous_c(Mb.shape, Mb.strides):
        m = Mb._buf
        mo = Mb.offset
        ai = ao
        out_i = oo
        for i in range(n):
            if m[mo + i]:
                o[out_i] = a[ai]
                out_i += 1
            ai += ast
    else:
        m = Mb._buf
        ms = Mb.strides[0]
        mi = Mb.offset
        ai = ao
        out_i = oo
        for _ in range(n):
            if m[mi]:
                o[out_i] = a[ai]
                out_i += 1
            mi += ms
            ai += ast

    return out



def _norm_axis(axis: int, ndim: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += ndim
    if ax < 0 or ax >= ndim:
        raise ShapeError(f"axis out of range: axis={axis} ndim={ndim}")
    return ax


def _reduce_out_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return tuple(shape[i] for i in range(len(shape)) if i != axis)


def _reduce_sum_b1_axis(A: "NDArray", axis: int):
    # returns int scalar if A.ndim==1 else i64 NDArray
    ax = _norm_axis(axis, A.ndim)
    if A.shape[ax] == 0:
        # sum over empty = 0
        if A.ndim == 1:
            return 0
        return zeros(_reduce_out_shape(A.shape, ax), I64, axes=tuple(a for i, a in enumerate(A.axes) if i != ax))

    if A.ndim == 1:
        # count true
        n = A.shape[0]
        buf = A._buf
        off = A.offset
        st = A.strides[0]
        cnt = 0
        if is_contiguous_c(A.shape, A.strides):
            for i in range(n):
                if buf[off + i]:
                    cnt += 1
        else:
            o = off
            for _ in range(n):
                if buf[o]:
                    cnt += 1
                o += st
        return cnt

    # ndim>=2 => output NDArray i64
    out_shape = _reduce_out_shape(A.shape, ax)
    out_axes = tuple(a for i, a in enumerate(A.axes) if i != ax)
    OUT = zeros(out_shape, I64, axes=out_axes)
    _reduce_sum_axis_generic(A, OUT, ax, mode="b1")
    return OUT


def _reduce_minmax_b1_axis(A: "NDArray", axis: int, want: str):
    ax = _norm_axis(axis, A.ndim)
    if A.shape[ax] == 0:
        raise DomainError(f"{want}: empty reduction axis")

    if A.ndim == 1:
        n = A.shape[0]
        buf = A._buf
        off = A.offset
        st = A.strides[0]
        if is_contiguous_c(A.shape, A.strides):
            # init
            v0 = 1 if buf[off] else 0
            if want == "min":
                m = v0
                for i in range(1, n):
                    v = 1 if buf[off + i] else 0
                    if v < m:
                        m = v
                return bool(m)
            else:
                m = v0
                for i in range(1, n):
                    v = 1 if buf[off + i] else 0
                    if v > m:
                        m = v
                return bool(m)
        else:
            o = off
            v0 = 1 if buf[o] else 0
            m = v0
            o += st
            for _ in range(1, n):
                v = 1 if buf[o] else 0
                if want == "min":
                    if v < m:
                        m = v
                else:
                    if v > m:
                        m = v
                o += st
            return bool(m)

    out_shape = _reduce_out_shape(A.shape, ax)
    out_axes = tuple(a for i, a in enumerate(A.axes) if i != ax)
    OUT = zeros(out_shape, B1, axes=out_axes)
    _reduce_minmax_axis_generic(A, OUT, ax, want=want, mode="b1")
    return OUT


def _reduce_sum_axis_generic(A: "NDArray", OUT: "NDArray", axis: int, mode: str):
    """
    Generic axis sum for ndim>=2.
    mode: "i64" | "f64" | "b1"
    OUT shape = A.shape with axis removed.
    """
    shape = A.shape
    ndim = A.ndim
    ax = axis
    inner = shape[ax]

    a_buf = A._buf
    a_off0 = A.offset
    a_str = A.strides

    o_buf = OUT._buf
    o_off0 = OUT.offset
    o_str = OUT.strides

    out_shape = OUT.shape
    out_nd = OUT.ndim

    # iterate over OUT indices
    idx = [0] * out_nd
    while True:
        # map idx -> full index with axis inserted
        # compute base offsets for this OUT cell
        a_base = a_off0
        o_base = o_off0
        j = 0
        for i in range(ndim):
            if i == ax:
                continue
            k = idx[j]
            a_base += k * a_str[i]
            o_base += k * o_str[j]
            j += 1

        if mode == "i64":
            s = 0
            for t in range(inner):
                s += a_buf[a_base + t * a_str[ax]]
            o_buf[o_base] = s
        elif mode == "f64":
            s = 0.0
            for t in range(inner):
                s += a_buf[a_base + t * a_str[ax]]
            o_buf[o_base] = s
        else:  # b1 count
            s = 0
            for t in range(inner):
                if a_buf[a_base + t * a_str[ax]]:
                    s += 1
            o_buf[o_base] = s

        # increment idx
        p = out_nd - 1
        while p >= 0:
            idx[p] += 1
            if idx[p] < out_shape[p]:
                break
            idx[p] = 0
            p -= 1
        if p < 0:
            break


def _reduce_minmax_axis_generic(A: "NDArray", OUT: "NDArray", axis: int, want: str, mode: str):
    """
    Generic axis min/max for ndim>=2.
    mode: "i64" | "f64" | "b1"
    OUT shape = A.shape with axis removed.
    """
    shape = A.shape
    ndim = A.ndim
    ax = axis
    inner = shape[ax]
    if inner == 0:
        raise DomainError(f"{want}: empty reduction axis")

    a_buf = A._buf
    a_off0 = A.offset
    a_str = A.strides

    o_buf = OUT._buf
    o_off0 = OUT.offset
    o_str = OUT.strides

    out_shape = OUT.shape
    out_nd = OUT.ndim

    idx = [0] * out_nd
    while True:
        a_base = a_off0
        o_base = o_off0
        j = 0
        for i in range(ndim):
            if i == ax:
                continue
            k = idx[j]
            a_base += k * a_str[i]
            o_base += k * o_str[j]
            j += 1

        # init
        v0 = a_buf[a_base]
        if mode == "b1":
            best = 1 if v0 else 0
            for t in range(1, inner):
                v = 1 if a_buf[a_base + t * a_str[ax]] else 0
                if want == "min":
                    if v < best:
                        best = v
                else:
                    if v > best:
                        best = v
            o_buf[o_base] = 1 if best else 0
        else:
            best = v0
            for t in range(1, inner):
                v = a_buf[a_base + t * a_str[ax]]
                if want == "min":
                    if v < best:
                        best = v
                else:
                    if v > best:
                        best = v
            o_buf[o_base] = best

        p = out_nd - 1
        while p >= 0:
            idx[p] += 1
            if idx[p] < out_shape[p]:
                break
            idx[p] = 0
            p -= 1
        if p < 0:
            break


def sum(A: "NDArray", axis=None):
    if axis is None:
        # axis=None scalar reductions
        if A.dtype == I64:
            return sum_i64(A)
        if A.dtype == F64:
            return sum_f64(A)
        if A.dtype == B1:
            # count True
            return _reduce_sum_b1_axis(A, 0) if A.ndim == 1 else _reduce_sum_b1_axis(A, A.ndim - 1).sum(axis=0)  # fallback
        raise DTypeError("sum: unsupported dtype")
    # axis reduction
    if A.ndim == 1:
        ax = _norm_axis(axis, 1)
        if A.dtype == I64:
            # scalar
            n = A.shape[0]
            if n == 0:
                return 0
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                s = 0
                for i in range(n):
                    s += buf[off + i]
                return s
            else:
                s = 0
                o = off
                for _ in range(n):
                    s += buf[o]
                    o += st
                return s
        if A.dtype == F64:
            n = A.shape[0]
            if n == 0:
                return 0.0
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                s = 0.0
                for i in range(n):
                    s += buf[off + i]
                return s
            else:
                s = 0.0
                o = off
                for _ in range(n):
                    s += buf[o]
                    o += st
                return s
        if A.dtype == B1:
            return _reduce_sum_b1_axis(A, ax)
        raise DTypeError("sum: unsupported dtype")

    ax = _norm_axis(axis, A.ndim)
    out_shape = _reduce_out_shape(A.shape, ax)
    out_axes = tuple(a for i, a in enumerate(A.axes) if i != ax)

    if A.dtype == I64:
        OUT = zeros(out_shape, I64, axes=out_axes)
        _reduce_sum_axis_generic(A, OUT, ax, mode="i64")
        return OUT
    if A.dtype == F64:
        OUT = zeros(out_shape, F64, axes=out_axes)
        _reduce_sum_axis_generic(A, OUT, ax, mode="f64")
        return OUT
    if A.dtype == B1:
        return _reduce_sum_b1_axis(A, ax)

    raise DTypeError("sum: unsupported dtype")


def min(A: "NDArray", axis=None):
    if axis is None:
        if A.dtype == I64:
            return min_i64(A)
        if A.dtype == F64:
            return min_f64(A)
        if A.dtype == B1:
            # scalar min over mask: False if any False else True; empty -> DomainError
            return _reduce_minmax_b1_axis(A, 0, "min") if A.ndim == 1 else _reduce_minmax_b1_axis(A, A.ndim - 1, "min").min(axis=0)
        raise DTypeError("min: unsupported dtype")

    if A.ndim == 1:
        ax = _norm_axis(axis, 1)
        if A.shape[0] == 0:
            raise DomainError("min: empty array")
        if A.dtype == I64:
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                m = buf[off]
                for i in range(1, A.shape[0]):
                    v = buf[off + i]
                    if v < m:
                        m = v
                return m
            else:
                o = off
                m = buf[o]
                o += st
                for _ in range(1, A.shape[0]):
                    v = buf[o]
                    if v < m:
                        m = v
                    o += st
                return m
        if A.dtype == F64:
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                m = buf[off]
                for i in range(1, A.shape[0]):
                    v = buf[off + i]
                    if v < m:
                        m = v
                return m
            else:
                o = off
                m = buf[o]
                o += st
                for _ in range(1, A.shape[0]):
                    v = buf[o]
                    if v < m:
                        m = v
                    o += st
                return m
        if A.dtype == B1:
            return _reduce_minmax_b1_axis(A, ax, "min")
        raise DTypeError("min: unsupported dtype")

    ax = _norm_axis(axis, A.ndim)
    out_shape = _reduce_out_shape(A.shape, ax)
    out_axes = tuple(a for i, a in enumerate(A.axes) if i != ax)

    if A.dtype == I64:
        OUT = zeros(out_shape, I64, axes=out_axes)
        _reduce_minmax_axis_generic(A, OUT, ax, want="min", mode="i64")
        return OUT
    if A.dtype == F64:
        OUT = zeros(out_shape, F64, axes=out_axes)
        _reduce_minmax_axis_generic(A, OUT, ax, want="min", mode="f64")
        return OUT
    if A.dtype == B1:
        return _reduce_minmax_b1_axis(A, ax, "min")

    raise DTypeError("min: unsupported dtype")


def max(A: "NDArray", axis=None):
    if axis is None:
        if A.dtype == I64:
            return max_i64(A)
        if A.dtype == F64:
            return max_f64(A)
        if A.dtype == B1:
            return _reduce_minmax_b1_axis(A, 0, "max") if A.ndim == 1 else _reduce_minmax_b1_axis(A, A.ndim - 1, "max").max(axis=0)
        raise DTypeError("max: unsupported dtype")

    if A.ndim == 1:
        ax = _norm_axis(axis, 1)
        if A.shape[0] == 0:
            raise DomainError("max: empty array")
        if A.dtype == I64:
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                m = buf[off]
                for i in range(1, A.shape[0]):
                    v = buf[off + i]
                    if v > m:
                        m = v
                return m
            else:
                o = off
                m = buf[o]
                o += st
                for _ in range(1, A.shape[0]):
                    v = buf[o]
                    if v > m:
                        m = v
                    o += st
                return m
        if A.dtype == F64:
            buf = A._buf; off = A.offset; st = A.strides[0]
            if is_contiguous_c(A.shape, A.strides):
                m = buf[off]
                for i in range(1, A.shape[0]):
                    v = buf[off + i]
                    if v > m:
                        m = v
                return m
            else:
                o = off
                m = buf[o]
                o += st
                for _ in range(1, A.shape[0]):
                    v = buf[o]
                    if v > m:
                        m = v
                    o += st
                return m
        if A.dtype == B1:
            return _reduce_minmax_b1_axis(A, ax, "max")
        raise DTypeError("max: unsupported dtype")

    ax = _norm_axis(axis, A.ndim)
    out_shape = _reduce_out_shape(A.shape, ax)
    out_axes = tuple(a for i, a in enumerate(A.axes) if i != ax)

    if A.dtype == I64:
        OUT = zeros(out_shape, I64, axes=out_axes)
        _reduce_minmax_axis_generic(A, OUT, ax, want="max", mode="i64")
        return OUT
    if A.dtype == F64:
        OUT = zeros(out_shape, F64, axes=out_axes)
        _reduce_minmax_axis_generic(A, OUT, ax, want="max", mode="f64")
        return OUT
    if A.dtype == B1:
        return _reduce_minmax_b1_axis(A, ax, "max")

    raise DTypeError("max: unsupported dtype")


# (D) Update NDArray methods to accept axis
def _ndarray_sum(self: "NDArray", axis=None):
    return sum(self, axis=axis)

def _ndarray_min(self: "NDArray", axis=None):
    return min(self, axis=axis)

def _ndarray_max(self: "NDArray", axis=None):
    return max(self, axis=axis)

setattr(NDArray, "sum", _ndarray_sum)
setattr(NDArray, "min", _ndarray_min)
setattr(NDArray, "max", _ndarray_max)


def where(M: NDArray, A, B) -> NDArray:
    """NumPy-style where(M, A, B)."""
    if M.dtype != B1:
        raise DTypeError("where: first argument must be a boolean mask (B1)")
    
    # Coerce scalars
    A_is_scalar = isinstance(A, (int, float, bool))
    B_is_scalar = isinstance(B, (int, float, bool))

    if A_is_scalar and B_is_scalar:
        dt = promote(dtype_of_scalar(A), dtype_of_scalar(B))
        Ab = _scalar_array(A, dt).broadcast_to(M.shape)
        Bb = _scalar_array(B, dt).broadcast_to(M.shape)
    elif A_is_scalar:
        dt = promote(dtype_of_scalar(A), B.dtype)
        Ab = _scalar_array(A, dt).broadcast_to(broadcast_shape(M.shape, B.shape))
        Bb = B.broadcast_to(Ab.shape)
    elif B_is_scalar:
        dt = promote(A.dtype, dtype_of_scalar(B))
        Bb = _scalar_array(B, dt).broadcast_to(broadcast_shape(M.shape, A.shape))
        Ab = A.broadcast_to(Bb.shape)
    else:
        dt = promote(A.dtype, B.dtype)
        s = broadcast_shape(M.shape, broadcast_shape(A.shape, B.shape))
        Ab = A.broadcast_to(s)
        Bb = B.broadcast_to(s)
    
    out_shape = Ab.shape
    Mb = M.broadcast_to(out_shape)
    out = zeros(out_shape, dt, axes=Ab.axes)
    
    if dt == I64:
        where_i64(Mb, Ab if Ab.dtype == I64 else Ab.astype(I64), 
                  Bb if Bb.dtype == I64 else Bb.astype(I64), out)
    else:
        where_f64(Mb, Ab if Ab.dtype == F64 else Ab.astype(F64), 
                  Bb if Bb.dtype == F64 else Bb.astype(F64), out)
    return out


def _compare(A: NDArray, B, op: str) -> NDArray:
    if isinstance(B, (int, float, bool)):
        dt = promote(A.dtype, dtype_of_scalar(B))
        out = zeros(A.shape, B1, axes=A.axes)
        if dt == I64:
            cmp_scalar_i64(A if A.dtype == I64 else A.astype(I64), int(B), out, op)
        else:
            cmp_scalar_f64(A if A.dtype == F64 else A.astype(F64), float(B), out, op)
        return out
    
    if not isinstance(B, NDArray):
        raise DTypeError(f"Comparison: unsupported type {type(B)!r}")
    
    dt = promote(A.dtype, B.dtype)
    s = broadcast_shape(A.shape, B.shape)
    Ab = A.broadcast_to(s)
    Bb = B.broadcast_to(s)
    
    out = zeros(s, B1, axes=Ab.axes)
    if dt == I64:
        cmp_i64(Ab if Ab.dtype == I64 else Ab.astype(I64), 
                Bb if Bb.dtype == I64 else Bb.astype(I64), out, op)
    else:
        cmp_f64(Ab if Ab.dtype == F64 else Ab.astype(F64), 
                Bb if Bb.dtype == F64 else Bb.astype(F64), out, op)
    return out


def eq(A: NDArray, B): return _compare(A, B, "eq")
def ne(A: NDArray, B): return _compare(A, B, "ne")
def lt(A: NDArray, B): return _compare(A, B, "lt")
def le(A: NDArray, B): return _compare(A, B, "le")
def gt(A: NDArray, B): return _compare(A, B, "gt")
def ge(A: NDArray, B): return _compare(A, B, "ge")


def _copy_strided(A: NDArray, OUT: NDArray) -> None:
    # elementwise copy via indices
    shape = A.shape
    strides = A.strides
    ndim = A.ndim
    buf = A._buf
    base = A.offset
    out_buf = OUT._buf
    out_base = OUT.offset

    out_strides = OUT.strides

    idx = [0] * ndim
    while True:
        a_off = base
        o_off = out_base
        for ax in range(ndim):
            a_off += idx[ax] * strides[ax]
            o_off += idx[ax] * out_strides[ax]
        out_buf[o_off] = buf[a_off]

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break


def _astype_copy(A: NDArray, dt: DType) -> NDArray:
    out = zeros(A.shape, dt, axes=A.axes)
    # copy + cast
    shape = A.shape
    strides = A.strides
    ndim = A.ndim
    buf = A._buf
    base = A.offset
    out_buf = out._buf
    out_base = out.offset
    out_strides = out.strides

    idx = [0] * ndim
    while True:
        a_off = base
        o_off = out_base
        for ax in range(ndim):
            a_off += idx[ax] * strides[ax]
            o_off += idx[ax] * out_strides[ax]
        if dt == I64:
            out_buf[o_off] = int(buf[a_off])
        else:
            out_buf[o_off] = float(buf[a_off])

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
    return out


def _ndarray_astype(self: NDArray, dtype: str | DType) -> NDArray:
    dt = parse_dtype(dtype) if not isinstance(dtype, DType) else dtype
    if dt == self.dtype:
        return self.copy()
    return _astype_copy(self, dt)


# attach as method (keeps file count low for v0.0.1)
setattr(NDArray, "astype", _ndarray_astype)


def _ndarray_broadcast_to(self: "NDArray", shape) -> "NDArray":
    """
    Return a broadcasted VIEW of self with target `shape` (NumPy rules).
    Uses 0-stride expansion. No copy.

    v0.0.3: requires target shape compatible with self.shape.
    """
    target = normalize_shape(shape)
    in_shape = self.shape
    in_strides = self.strides
    in_axes = self.axes

    if in_shape == target:
        return self

    nin = len(in_shape)
    nt = len(target)
    if nin > nt:
        raise ShapeError("broadcast_to: target has fewer dims than input")

    # Align from the right
    pad = nt - nin
    # Prefix new axes names for padded dims
    prefix_axes = tuple(f"b{i}" for i in range(pad))
    new_axes = prefix_axes + in_axes

    # Build padded shapes/strides
    padded_shape = (1,) * pad + in_shape
    padded_strides = (0,) * pad + in_strides

    out_strides = [0] * nt
    for i in range(nt):
        d_in = padded_shape[i]
        d_out = target[i]
        st_in = padded_strides[i]
        if d_in == d_out:
            out_strides[i] = st_in
        elif d_in == 1 and d_out >= 1:
            out_strides[i] = 0
        else:
            raise ShapeError(f"broadcast_to: incompatible dim {d_in} -> {d_out}")

    return self._view(target, tuple(out_strides), self.offset, new_axes)


setattr(NDArray, "broadcast_to", _ndarray_broadcast_to)


def _ndarray_invert(self: "NDArray") -> "NDArray":
    return _mask_not(self)


def _ndarray_and(self: "NDArray", other: "NDArray") -> "NDArray":
    return _mask_binary(self, other, "and")


def _ndarray_or(self: "NDArray", other: "NDArray") -> "NDArray":
    return _mask_binary(self, other, "or")


def _ndarray_xor(self: "NDArray", other: "NDArray") -> "NDArray":
    return _mask_binary(self, other, "xor")


setattr(NDArray, "__invert__", _ndarray_invert)
setattr(NDArray, "__and__", _ndarray_and)
setattr(NDArray, "__or__", _ndarray_or)
setattr(NDArray, "__xor__", _ndarray_xor)
