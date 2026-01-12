from gymnasium import spaces
import numpy as np
from constants import CONFIG
from enums.PackagingColor import PackagingColor


class ObservationSpaces:
    """
    Defines observation spaces for each agent type.
    All spaces are gymnasium.spaces compatible.
    """
    
    @staticmethod
    def pickup_station() -> spaces.Dict:
        """
        Pickup Station observes:
        - Queue of pending orders (simplified: next N orders' product types)
        - Available trays count
        - Trays currently at pickup (what's loaded)
        """
        return spaces.Dict({
            # Current order view
            'order_size': spaces.Discrete(21),              # 0-20 products in order
            'products_remaining': spaces.Discrete(21),      # How many left to load

            # Next product to load
            'next_product_type': spaces.Discrete(4),        # 0=None, 1=SMALL, 2=MEDIUM, 3=BIG
            'next_product_color': spaces.Discrete(4),       # 0=None, 1=RED, 2=BLUE, 3=GREEN

            # Current tray state
            'current_tray_type': spaces.Discrete(4),        # 0=Empty, 1=SMALL, 2=MEDIUM, 3=BIG
            'current_tray_color': spaces.Discrete(4),       # 0=Empty, 1=RED, 2=BLUE, 3=GREEN
            'current_tray_count': spaces.Discrete(6),       # 0-5 products on tray

            # Action mask
            'action_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
        })
    
    @staticmethod
    def agv() -> spaces.Dict:
        """
        AGV observes:
        - Current position (grid cell)
        - Carrying tray (yes/no and its contents summary)
        - Tray type (to know which machine to use)
        - Status of all stations (busy/free/queue length)
        - Ready trays at each location (trays waiting for pickup)
        """
        return spaces.Dict({
            'position': spaces.MultiDiscrete([CONFIG['grid_rows'], CONFIG['grid_cols']]),
            'carrying_tray': spaces.Discrete(2),  # 0=no, 1=yes
            'tray_product_count': spaces.Discrete(CONFIG['tray_capacity'] + 1),
            'tray_type': spaces.Discrete(4),  # 0=none, 1=SMALL, 2=MEDIUM, 3=BIG
            'tray_needs_processing': spaces.Discrete(2),  # Does tray need machine?
            'tray_needs_packaging': spaces.Discrete(2),   # Does tray need packaging?
            # Ready trays at pickup (trays AGV can pick up)
            'pickup_ready_trays': spaces.Discrete(10),
            'small_machine_busy': spaces.Discrete(2),
            'big_machine_busy': spaces.Discrete(2),
            # Ready trays at machines (processed, waiting for pickup)
            'small_machine_ready': spaces.Discrete(10),
            'big_machine_ready': spaces.Discrete(10),
            'storage_tray_count': spaces.Discrete(100),
            'action_mask': spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8),
        })
    
    @staticmethod
    def small_machine() -> spaces.Dict:
        """
        Small Machine observes:
        - Is busy (processing)
        - Queue of trays waiting
        - Current processing progress
        """
        return spaces.Dict({
            'is_busy': spaces.Discrete(2),
            'processing_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'queue_length': spaces.Discrete(10),
            'action_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
        })

    @staticmethod
    def big_machine() -> spaces.Dict:
        """Same structure as small machine."""
        return spaces.Dict({
            'is_busy': spaces.Discrete(2),
            'processing_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'queue_length': spaces.Discrete(10),
            'action_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
        })
    
    @staticmethod
    def packaging(color: PackagingColor) -> spaces.Dict:
        """
        Packaging station observes:
        - Is busy
        - Queue of products waiting for THIS color
        - Processing progress
        """
        return spaces.Dict({
            'is_busy': spaces.Discrete(2),
            'processing_progress': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'queue_length': spaces.Discrete(20),
            'action_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
        })
