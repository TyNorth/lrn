"""
latmath.array.kernels.elem_i64

Elementwise kernels for i64.
"""

from __future__ import annotations

from ...core.errors import ShapeError
from ..layout import is_contiguous_c, numel

__all__ = ["add_i64", "mul_i64", "add_scalar_i64", "mul_scalar_i64", "cmp_i64", "cmp_scalar_i64", "where_i64"]


def add_i64(A, B, OUT):
    if A.shape != B.shape or A.shape != OUT.shape:
        raise ShapeError("add: shape mismatch")
    n = numel(A.shape)
    if n == 0:
        return OUT

    # Fast contiguous path
    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(B.shape, B.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a = A._buf
        b = B._buf
        o = OUT._buf
        ao = A.offset
        bo = B.offset
        oo = OUT.offset
        for i in range(n):
            o[oo + i] = a[ao + i] + b[bo + i]
        return OUT

    # Strided path (2D optimized, generic fallback)
    _binary_strided(A, B, OUT, op="add")
    return OUT


def mul_i64(A, B, OUT):
    if A.shape != B.shape or A.shape != OUT.shape:
        raise ShapeError("mul: shape mismatch")
    n = numel(A.shape)
    if n == 0:
        return OUT

    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(B.shape, B.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a = A._buf
        b = B._buf
        o = OUT._buf
        ao = A.offset
        bo = B.offset
        oo = OUT.offset
        for i in range(n):
            o[oo + i] = a[ao + i] * b[bo + i]
        return OUT

    _binary_strided(A, B, OUT, op="mul")
    return OUT


def _binary_strided(A, B, OUT, op: str):
    shape = A.shape
    ndim = A.ndim
    a_buf, b_buf, o_buf = A._buf, B._buf, OUT._buf
    ao, bo, oo = A.offset, B.offset, OUT.offset
    as_, bs_, os_ = A.strides, B.strides, OUT.strides

    if ndim == 1:
        n = shape[0]
        a_i, b_i, o_i = ao, bo, oo
        a_st, b_st, o_st = as_[0], bs_[0], os_[0]
        if op == "add":
            for _ in range(n):
                o_buf[o_i] = a_buf[a_i] + b_buf[b_i]
                a_i += a_st; b_i += b_st; o_i += o_st
        else:
            for _ in range(n):
                o_buf[o_i] = a_buf[a_i] * b_buf[b_i]
                a_i += a_st; b_i += b_st; o_i += o_st
        return

    if ndim == 2:
        n0, n1 = shape
        a0, a1 = as_
        b0, b1 = bs_
        o0, o1 = os_
        a_row, b_row, o_row = ao, bo, oo
        if op == "add":
            for _ in range(n0):
                a_i, b_i, o_i = a_row, b_row, o_row
                for _ in range(n1):
                    o_buf[o_i] = a_buf[a_i] + b_buf[b_i]
                    a_i += a1; b_i += b1; o_i += o1
                a_row += a0; b_row += b0; o_row += o0
        else:
            for _ in range(n0):
                a_i, b_i, o_i = a_row, b_row, o_row
                for _ in range(n1):
                    o_buf[o_i] = a_buf[a_i] * b_buf[b_i]
                    a_i += a1; b_i += b1; o_i += o1
                a_row += a0; b_row += b0; o_row += o0
        return

    # Generic N-D
    idx = [0] * ndim
    while True:
        a_i = ao
        b_i = bo
        o_i = oo
        for ax in range(ndim):
            a_i += idx[ax] * as_[ax]
            b_i += idx[ax] * bs_[ax]
            o_i += idx[ax] * os_[ax]

        if op == "add":
            o_buf[o_i] = a_buf[a_i] + b_buf[b_i]
        else:
            o_buf[o_i] = a_buf[a_i] * b_buf[b_i]

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break


def add_scalar_i64(A, s: int, OUT):
    if A.shape != OUT.shape:
        raise ShapeError("add_scalar: shape mismatch")
    n = numel(A.shape)
    if n == 0:
        return OUT
    s = int(s)

    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a = A._buf
        o = OUT._buf
        ao = A.offset
        oo = OUT.offset
        for i in range(n):
            o[oo + i] = a[ao + i] + s
        return OUT

    # Strided
    shape = A.shape
    ndim = A.ndim
    a_buf, o_buf = A._buf, OUT._buf
    ao, oo = A.offset, OUT.offset
    as_, os_ = A.strides, OUT.strides

    if ndim == 1:
        n0 = shape[0]
        a_i, o_i = ao, oo
        a_st, o_st = as_[0], os_[0]
        for _ in range(n0):
            o_buf[o_i] = a_buf[a_i] + s
            a_i += a_st; o_i += o_st
        return OUT

    if ndim == 2:
        n0, n1 = shape
        a0, a1 = as_
        o0, o1 = os_
        a_row, o_row = ao, oo
        for _ in range(n0):
            a_i, o_i = a_row, o_row
            for _ in range(n1):
                o_buf[o_i] = a_buf[a_i] + s
                a_i += a1; o_i += o1
            a_row += a0; o_row += o0
        return OUT

    idx = [0] * ndim
    while True:
        a_i = ao
        o_i = oo
        for ax in range(ndim):
            a_i += idx[ax] * as_[ax]
            o_i += idx[ax] * os_[ax]
        o_buf[o_i] = a_buf[a_i] + s

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
    return OUT


def mul_scalar_i64(A, s: int, OUT):
    if A.shape != OUT.shape:
        raise ShapeError("mul_scalar: shape mismatch")
    n = numel(A.shape)
    if n == 0:
        return OUT
    s = int(s)

    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a = A._buf
        o = OUT._buf
        ao = A.offset
        oo = OUT.offset
        for i in range(n):
            o[oo + i] = a[ao + i] * s
        return OUT

    # Strided
    shape = A.shape
    ndim = A.ndim
    a_buf, o_buf = A._buf, OUT._buf
    ao, oo = A.offset, OUT.offset
    as_, os_ = A.strides, OUT.strides

    if ndim == 1:
        n0 = shape[0]
        a_i, o_i = ao, oo
        a_st, o_st = as_[0], os_[0]
        for _ in range(n0):
            o_buf[o_i] = a_buf[a_i] * s
            a_i += a_st; o_i += o_st
        return OUT

    if ndim == 2:
        n0, n1 = shape
        a0, a1 = as_
        o0, o1 = os_
        a_row, o_row = ao, oo
        for _ in range(n0):
            a_i, o_i = a_row, o_row
            for _ in range(n1):
                o_buf[o_i] = a_buf[a_i] * s
                a_i += a1; o_i += o1
            a_row += a0; o_row += o0
        return OUT

    idx = [0] * ndim
    while True:
        a_i = ao
        o_i = oo
        for ax in range(ndim):
            a_i += idx[ax] * as_[ax]
            o_i += idx[ax] * os_[ax]
        o_buf[o_i] = a_buf[a_i] * s

        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]:
                break
            idx[ax] = 0
            ax -= 1
        if ax < 0:
            break
    return OUT


def cmp_i64(A, B, OUT, op: str):
    if A.shape != B.shape or A.shape != OUT.shape:
        raise ShapeError("cmp: shape mismatch")
    n = numel(A.shape)
    if n == 0: return OUT
    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(B.shape, B.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a, b, o = A._buf, B._buf, OUT._buf
        ao, bo, oo = A.offset, B.offset, OUT.offset
        if op == "eq":
            for i in range(n): o[oo + i] = 1 if a[ao + i] == b[bo + i] else 0
        elif op == "ne":
            for i in range(n): o[oo + i] = 1 if a[ao + i] != b[bo + i] else 0
        elif op == "gt":
            for i in range(n): o[oo + i] = 1 if a[ao + i] > b[bo + i] else 0
        elif op == "ge":
            for i in range(n): o[oo + i] = 1 if a[ao + i] >= b[bo + i] else 0
        elif op == "lt":
            for i in range(n): o[oo + i] = 1 if a[ao + i] < b[bo + i] else 0
        elif op == "le":
            for i in range(n): o[oo + i] = 1 if a[ao + i] <= b[bo + i] else 0
        return OUT
    return _cmp_strided(A, B, OUT, op)


def cmp_scalar_i64(A, val: int, OUT, op: str):
    if A.shape != OUT.shape:
        raise ShapeError("cmp: shape mismatch")
    n = numel(A.shape)
    if n == 0: return OUT
    val = int(val)
    if is_contiguous_c(A.shape, A.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        a, o = A._buf, OUT._buf
        ao, oo = A.offset, OUT.offset
        if op == "eq":
            for i in range(n): o[oo + i] = 1 if a[ao + i] == val else 0
        elif op == "ne":
            for i in range(n): o[oo + i] = 1 if a[ao + i] != val else 0
        elif op == "gt":
            for i in range(n): o[oo + i] = 1 if a[ao + i] > val else 0
        elif op == "ge":
            for i in range(n): o[oo + i] = 1 if a[ao + i] >= val else 0
        elif op == "lt":
            for i in range(n): o[oo + i] = 1 if a[ao + i] < val else 0
        elif op == "le":
            for i in range(n): o[oo + i] = 1 if a[ao + i] <= val else 0
        return OUT
    return _cmp_scalar_strided(A, val, OUT, op)


def where_i64(M, A, B, OUT):
    if M.shape != A.shape or A.shape != B.shape or B.shape != OUT.shape:
        raise ShapeError("where: shape mismatch")
    n = numel(M.shape)
    if n == 0: return OUT
    if is_contiguous_c(M.shape, M.strides) and is_contiguous_c(A.shape, A.strides) and \
       is_contiguous_c(B.shape, B.strides) and is_contiguous_c(OUT.shape, OUT.strides):
        m, a, b, o = M._buf, A._buf, B._buf, OUT._buf
        mo, ao, bo, oo = M.offset, A.offset, B.offset, OUT.offset
        for i in range(n):
            o[oo+i] = a[ao+i] if m[mo+i] else b[bo+i]
        return OUT
    return _where_strided(M, A, B, OUT)


def _cmp_strided(A, B, OUT, op: str):
    shape, ndim = A.shape, A.ndim
    a_buf, b_buf, o_buf = A._buf, B._buf, OUT._buf
    ao, bo, oo = A.offset, B.offset, OUT.offset
    as_, bs_, os_ = A.strides, B.strides, OUT.strides
    idx = [0] * ndim
    while True:
        a_i = ao; b_i = bo; o_i = oo
        for ax in range(ndim):
            k = idx[ax]
            a_i += k * as_[ax]
            b_i += k * bs_[ax]
            o_i += k * os_[ax]
        va, vb = a_buf[a_i], b_buf[b_i]
        if op == "eq": res = 1 if va == vb else 0
        elif op == "ne": res = 1 if va != vb else 0
        elif op == "gt": res = 1 if va > vb else 0
        elif op == "ge": res = 1 if va >= vb else 0
        elif op == "lt": res = 1 if va < vb else 0
        elif op == "le": res = 1 if va <= vb else 0
        o_buf[o_i] = res
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]: break
            idx[ax] = 0; ax -= 1
        if ax < 0: break


def _cmp_scalar_strided(A, val: int, OUT, op: str):
    shape, ndim = A.shape, A.ndim
    a_buf, o_buf = A._buf, OUT._buf
    ao, oo = A.offset, OUT.offset
    as_, os_ = A.strides, OUT.strides
    idx = [0] * ndim
    while True:
        a_i = ao; o_i = oo
        for ax in range(ndim):
            k = idx[ax]
            a_i += k * as_[ax]
            o_i += k * os_[ax]
        va = a_buf[a_i]
        if op == "eq": res = 1 if va == val else 0
        elif op == "ne": res = 1 if va != val else 0
        elif op == "gt": res = 1 if va > val else 0
        elif op == "ge": res = 1 if va >= val else 0
        elif op == "lt": res = 1 if va < val else 0
        elif op == "le": res = 1 if va <= val else 0
        o_buf[o_i] = res
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]: break
            idx[ax] = 0; ax -= 1
        if ax < 0: break


def _where_strided(M, A, B, OUT):
    shape, ndim = M.shape, M.ndim
    m_buf, a_buf, b_buf, o_buf = M._buf, A._buf, B._buf, OUT._buf
    mo, ao, bo, oo = M.offset, A.offset, B.offset, OUT.offset
    ms, as_, bs, os_ = M.strides, A.strides, B.strides, OUT.strides
    idx = [0] * ndim
    while True:
        mi = mo; ai = ao; bi = bo; oi = oo
        for ax in range(ndim):
            k = idx[ax]
            mi += k * ms[ax]
            ai += k * as_[ax]
            bi += k * bs[ax]
            oi += k * os_[ax]
        o_buf[oi] = a_buf[ai] if m_buf[mi] else b_buf[bi]
        ax = ndim - 1
        while ax >= 0:
            idx[ax] += 1
            if idx[ax] < shape[ax]: break
            idx[ax] = 0; ax -= 1
        if ax < 0: break
