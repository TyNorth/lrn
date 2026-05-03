from dataclasses import dataclass, field


@dataclass
class Spring:
    stiffness: int = 1
    tau: int = 4
    precedence: int = 50
    exposure_count: int = 0
    modality: str = "T"
    directional: bool = False
    origin: str = ""
    myelin: int = 0
    distinction: str = ""
    relation_type: str = ""
    channels: dict = None
    saturated: bool = False
    saturation_count: int = 0

    def __post_init__(self):
        if self.channels is None:
            self.channels = {}

    @property
    def energy(self) -> int:
        if self.stiffness == 0:
            return 1000
        if self.stiffness > 0:
            return max(5, 50 - (self.stiffness * 5))
        return 50 + abs(self.stiffness) * 15


SPRING_PROMOTION_THRESHOLDS = {
    4: 15,
    3: 40,
    2: 80,
    1: 120,
}

# Stiffness ceilings per tau level (geometric limits)
#
# Derived from the kissing number K=12 — the maximum number of non-overlapping
# unit spheres that can simultaneously touch a central sphere in 3D space.
# This is a hard topological constraint on how many neighbors a node can
# meaningfully maintain before connections start crowding each other out.
#
# Each spring occupies angular space on the node's surface (solid angle).
# Stronger springs (lower tau) are "thicker" bonds that consume more surface area.
# The ceiling is the point where a spring has consumed enough of the node's
# finite surface area that further reinforcement would geometrically crowd
# out other connections — the lattice must stop paying attention to it.
#
# Formula: ceiling(τ) = K × 2^(4-τ)
#   τ=4: 12 × 2^0 = 12   (contextual — thin, many can coexist)
#   τ=3: 12 × 2^1 = 24   (categorical — moderate thickness)
#   τ=2: 12 × 2^2 = 48   (causal — thick, fewer fit)
#   τ=1: 12 × 2^3 = 96   (definitional — very thick, core identity)
#   τ=0: 12 × 2^4 = 192  (constitutive — permanent, rare)
#
# This creates a 16× range from loosest to tightest, matching the biological
# reality that permanent connections are rare but dominant. Once a spring
# hits its ceiling, it's "geometrically saturated" — the knot can't get
# any tighter given the surface area available.
KISSING_NUMBER = 12

STIFFNESS_CEILINGS = {
    4: 12,     # contextual: K × 2^0 — thin, many coexist
    3: 24,     # categorical: K × 2^1 — moderate
    2: 48,     # causal: K × 2^2 — thick
    1: 96,     # definitional: K × 2^3 — very thick
    0: 192,    # constitutive: K × 2^4 — permanent
}

# After this many saturations, spring is "learned" and skipped during REM.
# 3 saturations = the connection has been reinforced across 3 distinct
# wake-buffer contexts, confirming it's not a fluke co-occurrence.
SATURATION_LEARNED_THRESHOLD = 3

# Diminishing returns multiplier for saturated springs.
# Once learned, springs get 10% of normal reinforcement — enough to
# maintain the connection but not enough to crowd out weaker springs.
SATURATED_RETURN_RATE = 10  # 10% of normal reinforcement