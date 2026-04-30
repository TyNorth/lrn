from typing import Dict, Tuple, Optional
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.node import Node
from lrn.spring import Spring


UPSILON = 12
TAU_W = [20736, 1728, 144, 12, 1]
K_BASE = 1024
RETENTION = 4
INFLUENCE = 6
PROPAGATION_STEPS = 5
WINDOW_SIZE = 5
E_THRESHOLD = 48

FLAG_PINNED = 1 << 0
FLAG_DAMPENED = 1 << 1
FLAG_EPHEMERAL = 1 << 2
FLAG_META = 1 << 3
FLAG_STOP = 1 << 4
FLAG_CONFINED = 1 << 5
FLAG_ANCHOR = 1 << 6
FLAG_FROZEN = 1 << 7
FLAG_SENSOR = 1 << 8


class LatticeNN:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.springs: Dict[Tuple[str, str], Spring] = {}
        self.trigrams: Dict[Tuple, int] = {}
        self.k_base = K_BASE
        self._adj_dirty = True
        self._name_to_idx = {}
        self._idx_to_name = []
        self._act = []
        self._act_next = []
        self._flags = []

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def add_node(self, name: str) -> Node:
        if name not in self.nodes:
            self.nodes[name] = Node(name=name)
            self._adj_dirty = True
        return self.nodes[name]

    def get_or_create(self, name: str) -> Node:
        return self.add_node(name)

    def add_spring(self, a: str, b: str, stiffness: int = 1,
                   tau: int = 4) -> Spring:
        self.add_node(a)
        self.add_node(b)
        key = self._key(a, b)
        if key in self.springs:
            sp = self.springs[key]
            sp.stiffness += stiffness
            sp.tau = min(sp.tau, tau)
            sp.exposure_count += 1
        else:
            sp = Spring(stiffness=stiffness, tau=tau)
            self.springs[key] = sp
        self._adj_dirty = True
        return sp

    def add_or_update_spring(self, a: str, b: str, stiffness: int,
                              tau: int = 4, mode: str = "add") -> Spring:
        self.add_node(a)
        self.add_node(b)
        key = self._key(a, b)
        if key in self.springs:
            sp = self.springs[key]
            if mode == "add":
                sp.stiffness += stiffness
            elif mode == "pos_max":
                if stiffness > 0:
                    sp.stiffness = max(sp.stiffness, stiffness)
            elif mode == "neg_override":
                sp.stiffness = stiffness
            elif mode == "set":
                sp.stiffness = stiffness
            sp.tau = min(sp.tau, tau)
            sp.exposure_count += 1
        else:
            sp = Spring(stiffness=stiffness, tau=tau, exposure_count=1)
            self.springs[key] = sp
        self._adj_dirty = True
        return sp

    def get_neighbors(self, name: str) -> list:
        neighbors = []
        for (a, b), sp in self.springs.items():
            if a == name:
                neighbors.append((b, sp))
            elif b == name:
                neighbors.append((a, sp))
        return neighbors

    def _degree_of(self, name: str) -> int:
        return len(self.get_neighbors(name))

    def reset(self):
        for name, node in self.nodes.items():
            if not node.pinned:
                node.activation = 0
        ephemeral = [n for n, node in self.nodes.items() if node.is_ephemeral]
        for n in ephemeral:
            del self.nodes[n]
        self._adj_dirty = True

    def get_node(self, name: str) -> Optional[Node]:
        return self.nodes.get(name)

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, name: str):
        return name in self.nodes