from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Node:
    name: str
    x: int = 0
    y: int = 0
    z: int = 0
    activation: int = 0
    pinned: bool = False
    dampened: bool = False
    is_ephemeral: bool = False
    role_counts: Dict[int, int] = field(default_factory=dict)
    modality: str = "T"
    reality_s: int = 1024
    unresolved_tension: int = 0
    tau: int = 4
    flavors: Dict[str, int] = field(default_factory=dict)

    @property
    def dominant_role(self) -> int:
        if not self.role_counts:
            return 2
        return max(self.role_counts, key=self.role_counts.get)


ROLE_NAMES = {
    0: "STARTER",
    1: "ACTOR",
    2: "LINKER",
    3: "SETTLER",
    4: "CLOSER"
}


def assign_roles(tokens: list) -> list:
    n = len(tokens)
    roles = []
    for i in range(n):
        frac = i / max(1, n - 1)
        if frac < 0.15:
            role = 0
        elif frac < 0.35:
            role = 1
        elif frac < 0.65:
            role = 2
        elif frac < 0.85:
            role = 3
        else:
            role = 4
        roles.append(role)
    return roles