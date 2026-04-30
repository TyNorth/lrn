"""
latdata.axis

Axis = ordered labels + alias sets.

Design goals:
- Names first (labels over indices)
- Aliases are first-class
- Selectors are composable (label, list, slice, callable, alias)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Sequence, Tuple, Union

from ..latmath.core.errors import DomainError, ShapeError, DTypeError

Label = Hashable
Selector = Union[
    slice,                 # :, slice(...)
    int,                   # direct index
    Label,                 # single label
    Sequence[Label],       # list/tuple of labels
    Tuple[Label, Label],   # inclusive label range (start_label, end_label)
    Callable[[Label], bool],  # predicate on label
]

__all__ = ["Axis"]


@dataclass
class Axis:
    name: str
    labels: List[Label]

    def __post_init__(self) -> None:
        self._reindex()

        # alias name -> selector spec (labels/list/range/callable)
        self._aliases: Dict[str, Any] = {}

    def _reindex(self) -> None:
        # stable first occurrence wins
        self._index: Dict[Label, int] = {}
        for i, lab in enumerate(self.labels):
            if lab not in self._index:
                self._index[lab] = i

    def __len__(self) -> int:
        return len(self.labels)

    def has(self, label: Label) -> bool:
        return label in self._index

    def pos(self, label: Label) -> int:
        if label not in self._index:
            raise ShapeError(f"Axis[{self.name}]: unknown label {label!r}")
        return self._index[label]

    def alias(self, name: str, sel: Any) -> None:
        n = str(name)
        if not n:
            raise DomainError("Axis.alias: name must be non-empty")
        self._aliases[n] = sel

    def resolve(self, sel: Any):
        """
        Resolve a selector into one of:
        - ("int", idx:int, labels_out:[label])
        - ("slice", slice_obj, labels_out:[...])
        - ("list", idx_list:[int], labels_out:[...])

        Notes:
        - tuple(label,label) means inclusive range by axis order
        - label string can also reference an alias name
        """
        if sel is None:
            sel = slice(None)

        # Alias expansion (only for str keys)
        if isinstance(sel, str) and sel in self._aliases:
            sel = self._aliases[sel]

        # Pass through ints
        if isinstance(sel, int):
            idx = int(sel)
            if idx < 0:
                idx += len(self.labels)
            if idx < 0 or idx >= len(self.labels):
                raise ShapeError(f"Axis[{self.name}]: index out of bounds")
            return ("int", idx, [self.labels[idx]])

        # Pass through slices (positional slicing)
        if isinstance(sel, slice):
            # normalize slice to concrete indices for labels_out
            start, stop, step = sel.indices(len(self.labels))
            labs = [self.labels[i] for i in range(start, stop, step)]
            return ("slice", slice(start, stop, step), labs)

        # Predicate callable on labels
        if callable(sel):
            idxs: List[int] = []
            labs: List[Label] = []
            for i, lab in enumerate(self.labels):
                if bool(sel(lab)):
                    idxs.append(i)
                    labs.append(lab)
            return ("list", idxs, labs)

        # Inclusive label range: (start_label, end_label)
        if isinstance(sel, tuple) and len(sel) == 2 and not isinstance(sel[0], (int, slice)) and not isinstance(sel[1], (int, slice)):
            a, b = sel[0], sel[1]
            ia = self.pos(a)
            ib = self.pos(b)
            if ia <= ib:
                labs = self.labels[ia : ib + 1]
                return ("slice", slice(ia, ib + 1, 1), labs)
            else:
                # allow reversed ranges: include descending with step -1
                labs = [self.labels[i] for i in range(ia, ib - 1, -1)]
                return ("slice", slice(ia, ib - 1, -1), labs)

        # Single label
        if not isinstance(sel, (list, tuple)):
            idx = self.pos(sel)
            return ("int", idx, [sel])

        # List/tuple of labels
        idxs = []
        labs = []
        for lab in sel:
            idxs.append(self.pos(lab))
            labs.append(lab)
        return ("list", idxs, labs)
