
from typing import Dict, Tuple
from enums.LocationType import LocationType

LOCATION_POSITIONS: Dict[LocationType, Tuple[int, int]] = {
    LocationType.PICKUP: (0, 0),
    LocationType.BIG_MACHINE: (0, 3),
    LocationType.SMALL_MACHINE: (2, 3),
    LocationType.STORAGE: (3, 0),
    LocationType.PACKAGING: (3, 5),
}

# Processing times
PROCESSING_TIMES = {
    'small_machine': 60,
    'big_machine': 120,
    'packaging': 30,
}

# Configuration
CONFIG = {
    'num_trays': 1000,
    'tray_capacity': 5,
    'num_packaging_blue': 2,
    'num_packaging_red': 1,
    'num_packaging_green': 1,
    'grid_rows': 4,
    'grid_cols': 6,
    'agv_speed': 1,  
    'step_size': 10,  # RL step size in simulation time units
}