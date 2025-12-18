from enum import Enum, auto

class LocationType(Enum):
    PICKUP = auto()
    BIG_MACHINE = auto()
    SMALL_MACHINE = auto()
    STORAGE = auto()
    PACKAGING = auto()