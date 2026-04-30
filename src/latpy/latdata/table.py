"""
latdata.table

Table = 2D named grid: NDArray + Axis(rows) + Axis(cols)

Design goals:
- Spreadsheet-native: rows/cols selected by names
- Selections read like math: T[rows, cols]
- Aliases are first-class (Axis.alias)
- Composable: selection -> compute (delegates to NDArray)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, List, Optional, Sequence, Tuple, Union

from ..latmath.core.errors import ShapeError, DTypeError
from ..latmath.array import NDArray, zeros, array as nd_array, where as nd_where
from ..latmath.array.dtypes import I64, F64, B1, parse_dtype
from .axis import Axis, Label

__all__ = ["Table"]


@dataclass
class Table:
    data: NDArray
    rows: Axis
    cols: Axis

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ShapeError("Table: data must be 2D NDArray")
        if self.data.shape[0] != len(self.rows):
            raise ShapeError("Table: row labels length must match data.shape[0]")
        if self.data.shape[1] != len(self.cols):
            raise ShapeError("Table: col labels length must match data.shape[1]")

    @classmethod
    def from_list(
        cls,
        values_2d: List[List[Any]],
        row_labels: Optional[List[Label]] = None,
        col_labels: Optional[List[Label]] = None,
        dtype: str | None = None,
        name_rows: str = "rows",
        name_cols: str = "cols",
    ) -> "Table":
        A = nd_array(values_2d, dtype=dtype)
        r = row_labels if row_labels is not None else [f"r{i}" for i in range(A.shape[0])]
        c = col_labels if col_labels is not None else [f"c{j}" for j in range(A.shape[1])]
        rows = Axis(name_rows, list(r))
        cols = Axis(name_cols, list(c))
        return cls(A, rows, cols)

    def __repr__(self) -> str:
        return f"Table(shape={self.data.shape}, rows={self.rows.name}, cols={self.cols.name})"

    def tolist(self):
        return self.data.tolist()

    # -------- Selection --------
    def __getitem__(self, key):
        """
        T[rows, cols] selection.

        Allowed selectors for rows/cols:
        - slice (positional)
        - int (positional)
        - label
        - list of labels
        - (start_label, end_label) inclusive range
        - callable(label)->bool predicate
        - alias name (string registered via Axis.alias)

        Returns:
        - scalar if both selectors resolve to ints
        - 1D NDArray if one selector resolves to int and the other to multiple
        - Table if both resolve to multiple
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ShapeError("Table.__getitem__: expected T[rows, cols]")
            rsel, csel = key
        else:
            # Column selection shortcut: T["col"] == T[:, "col"]
            rsel, csel = slice(None), key

        r_kind, r_key, r_labs = self.rows.resolve(rsel)
        c_kind, c_key, c_labs = self.cols.resolve(csel)

        # scalar scalar
        if r_kind == "int" and c_kind == "int":
            return self.data[r_key, c_key]

        # view path when both selectors are slices or ints (no list gather)
        if r_kind in ("int", "slice") and c_kind in ("int", "slice"):
            sub = self.data[r_key, c_key]
            # If both are slices -> Table view
            if isinstance(sub, NDArray) and sub.ndim == 2:
                return Table(sub, Axis(self.rows.name, list(r_labs)), Axis(self.cols.name, list(c_labs)))
            # If reduced to 1D
            if isinstance(sub, NDArray) and sub.ndim == 1:
                return sub
            return sub  # scalar fallback

        # gather/copy path (lists involved)
        return self._take(r_kind, r_key, r_labs, c_kind, c_key, c_labs)

    def _take(self, r_kind, r_key, r_labs, c_kind, c_key, c_labs):
        # Determine row index list
        if r_kind == "int":
            r_idxs = [r_key]
        elif r_kind == "slice":
            start, stop, step = r_key.indices(self.data.shape[0])
            r_idxs = list(range(start, stop, step))
        else:
            r_idxs = list(r_key)

        # Determine col index list
        if c_kind == "int":
            c_idxs = [c_key]
        elif c_kind == "slice":
            start, stop, step = c_key.indices(self.data.shape[1])
            c_idxs = list(range(start, stop, step))
        else:
            c_idxs = list(c_key)

        nr = len(r_idxs)
        nc = len(c_idxs)

        # scalar row -> return 1D NDArray copy
        if nr == 1 and nc >= 1 and (r_kind == "int" or (r_kind == "slice" and nr == 1)):
            out = zeros((nc,), self.data.dtype, axes=(self.cols.name,))
            _gather_row_1d(self.data, r_idxs[0], c_idxs, out)
            return out

        # scalar col -> return 1D NDArray copy
        if nc == 1 and nr >= 1 and (c_kind == "int" or (c_kind == "slice" and nc == 1)):
            out = zeros((nr,), self.data.dtype, axes=(self.rows.name,))
            _gather_col_1d(self.data, r_idxs, c_idxs[0], out)
            return out

        # 2D table copy
        out = zeros((nr, nc), self.data.dtype, axes=(self.rows.name, self.cols.name))
        _gather_2d(self.data, r_idxs, c_idxs, out)
        return Table(out, Axis(self.rows.name, list(r_labs)), Axis(self.cols.name, list(c_labs)))

    # -------- Compute (thin wrappers) --------
    def where(self, mask, a, b):
        """
        Table.where(mask, a, b)

        mask can be:
        - Table with b1 data
        - callable(row_label, col_label)->bool (slow; for v0.1.0 only)
        - NDArray b1 with broadcastable shape (2D)

        a/b can be:
        - scalars
        - Table
        - NDArray
        """
        M = self._mask_to_ndarray(mask)
        A = a.data if isinstance(a, Table) else a
        B = b.data if isinstance(b, Table) else b
        out = nd_where(M, A, B)
        # output axes/labels follow self for now
        return Table(out, Axis(self.rows.name, list(self.rows.labels)), Axis(self.cols.name, list(self.cols.labels)))

    def sum(self, axis: str = "rows"):
        """
        axis="rows" -> sum down rows => returns 1D NDArray over cols
        axis="cols" -> sum across cols => returns 1D NDArray over rows
        """
        ax = axis.strip().lower()
        if ax in ("rows", "row", "r", "0"):
            return self.data.sum(axis=0)
        if ax in ("cols", "col", "c", "1"):
            return self.data.sum(axis=1)
        raise ShapeError("Table.sum: axis must be 'rows' or 'cols'")

    def _mask_to_ndarray(self, mask):
        if isinstance(mask, Table):
            if mask.data.dtype != B1:
                raise DTypeError("Table.where: mask Table must have dtype b1")
            return mask.data
        if isinstance(mask, NDArray):
            if mask.dtype != B1:
                raise DTypeError("Table.where: mask NDArray must have dtype b1")
            return mask
        if callable(mask):
            # Build a b1 mask table from callable(row_label, col_label)->bool
            M = zeros(self.data.shape, B1, axes=self.data.axes)
            buf = M._buf
            oo = M.offset
            # M is contiguous by construction
            n0, n1 = M.shape
            k = 0
            for i, rl in enumerate(self.rows.labels):
                for j, cl in enumerate(self.cols.labels):
                    buf[oo + k] = 1 if bool(mask(rl, cl)) else 0
                    k += 1
            return M
        raise DTypeError("Table.where: unsupported mask type")
        

def _gather_2d(src: NDArray, r_idxs: List[int], c_idxs: List[int], out: NDArray) -> None:
    s_buf = src._buf
    o_buf = out._buf
    so = src.offset
    oo = out.offset
    rs0, rs1 = src.strides
    # out is contiguous by zeros()
    k = 0
    for ri in r_idxs:
        base = so + ri * rs0
        for ci in c_idxs:
            o_buf[oo + k] = s_buf[base + ci * rs1]
            k += 1


def _gather_row_1d(src: NDArray, r_idx: int, c_idxs: List[int], out: NDArray) -> None:
    s_buf = src._buf
    o_buf = out._buf
    so = src.offset
    oo = out.offset
    rs0, rs1 = src.strides
    base = so + r_idx * rs0
    for j, ci in enumerate(c_idxs):
        o_buf[oo + j] = s_buf[base + ci * rs1]


def _gather_col_1d(src: NDArray, r_idxs: List[int], c_idx: int, out: NDArray) -> None:
    s_buf = src._buf
    o_buf = out._buf
    so = src.offset
    oo = out.offset
    rs0, rs1 = src.strides
    for i, ri in enumerate(r_idxs):
        o_buf[oo + i] = s_buf[so + ri * rs0 + c_idx * rs1]
