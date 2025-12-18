from gymnasium import spaces
from constants import CONFIG

class ActionSpaces:
    """
    Defines action spaces for each agent type.
    """
    
    @staticmethod
    def pickup_station() -> spaces.Discrete:
        """
        Actions:
        0 = IDLE (do nothing)
        1 = LOAD_NEXT_PRODUCT (load next product from current order to tray)
        2 = SIGNAL_TRAY_READY (mark tray as ready for AGV pickup)
        """
        return spaces.Discrete(3)
    
    @staticmethod
    def agv() -> spaces.Discrete:
        """
        Actions:
        0 = IDLE
        1 = MOVE_TO_PICKUP
        2 = MOVE_TO_SMALL_MACHINE
        3 = MOVE_TO_BIG_MACHINE
        4 = MOVE_TO_STORAGE
        5 = MOVE_TO_PACKAGING
        6 = PICKUP_TRAY (at current location)
        7 = DROP_TRAY (at current location)
        """
        return spaces.Discrete(8)
    
    @staticmethod
    def small_machine() -> spaces.Discrete:
        """
        Actions:
        0 = IDLE
        1 = START_PROCESSING (if has tray in queue and not busy)
        2 = SIGNAL_COMPLETE (mark processed tray as ready for pickup)
        """
        return spaces.Discrete(3)
    
    @staticmethod
    def big_machine() -> spaces.Discrete:
        """Same as small machine."""
        return spaces.Discrete(3)
    
    @staticmethod
    def packaging() -> spaces.Discrete:
        """
        Actions:
        0 = IDLE
        1 = START_PACKAGING (if has product and not busy)
        2 = SIGNAL_COMPLETE
        """
        return spaces.Discrete(3)