from dataclasses import dataclass


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