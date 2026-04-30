"""
latmath.array.kernels.propagate_sparse

Algorithmic sparse propagation for b1 masks:
- work on sparse positions (or intervals)
- avoid touching dense buffers unless materializing

Intervals are half-open [start, stop).
"""

from __future__ import annotations

from ...core.errors import ShapeError, DTypeError

__all__ = [
    "propagate_positions_1d",
    "merge_intervals",
]


def propagate_positions_1d(pos: list[int], n: int, mode: str = "forward") -> list[tuple[int, int]]:
    """
    Given sorted true indices `pos` in a 1D line of length n,
    return sparse propagated result as intervals [start, stop).

    mode in {"forward","backward","span","full_if_any"}.
    """
    n = int(n)
    if n < 0:
        raise ShapeError("propagate_positions_1d: n must be >= 0")

    mode = str(mode).strip().lower()
    if mode not in ("forward", "backward", "span", "full_if_any"):
        raise ShapeError(f"propagate_positions_1d: unknown mode {mode!r}")

    if not pos:
        return []

    # Validate and assume sorted; keep checks cheap and deterministic.
    first = int(pos[0])
    last = int(pos[-1])
    if first < 0 or last < 0 or first >= n or last >= n:
        raise ShapeError("propagate_positions_1d: position out of bounds")

    if mode == "forward":
        return [(first, n)]
    if mode == "backward":
        return [(0, last + 1)]
    if mode == "span":
        return [(first, last + 1)]
    # full_if_any
    return [(0, n)]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping / adjacent half-open intervals.
    Requires: each interval has start <= stop.
    """
    if not intervals:
        return []
    itv = sorted((int(a), int(b)) for a, b in intervals)
    out: list[tuple[int, int]] = []
    a0, b0 = itv[0]
    if a0 > b0:
        raise ShapeError("merge_intervals: invalid interval")
    for a, b in itv[1:]:
        if a > b:
            raise ShapeError("merge_intervals: invalid interval")
        if a <= b0:  # overlap/adjacent (since half-open, treat a==b0 as adjacent merge)
            if b > b0:
                b0 = b
        else:
            out.append((a0, b0))
            a0, b0 = a, b
    out.append((a0, b0))
    return out

def materialize_intervals_1d(out_buf, out_off: int, out_stride: int, n: int, intervals: list[tuple[int,int]]) -> None:
    """
    Write intervals into a dense b1 output line (0/1).
    out_buf is array('b') or bytearray-like.
    Clears to 0 then writes 1 in intervals.
    """
    n = int(n)
    out_off = int(out_off)
    out_stride = int(out_stride)

    # clear
    oi = out_off
    for _ in range(n):
        out_buf[oi] = 0
        oi += out_stride

    # fill intervals
    for a, b in intervals:
        a = int(a); b = int(b)
        if a < 0: a = 0
        if b > n: b = n
        if b <= a:
            continue
        oi = out_off + a * out_stride
        for _ in range(b - a):
            out_buf[oi] = 1
            oi += out_stride