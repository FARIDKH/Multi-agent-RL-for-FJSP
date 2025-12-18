from enum import Enum, auto

class AgentType(Enum):
    PICKUP_STATION = auto()
    AGV = auto()
    SMALL_MACHINE = auto()
    BIG_MACHINE = auto()
    PACKAGING = auto()