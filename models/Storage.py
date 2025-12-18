from typing import List, Optional
from enums.LocationType import LocationType
from models.Tray import Tray


class Storage:
    """
    Storage area for trays waiting for processing or packaging.
    Not an RL agent - just a buffer managed by AGV decisions.
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.trays: List[Tray] = []
    
    def add_tray(self, tray: Tray) -> bool:
        """Add tray to storage. Returns False if full."""
        if len(self.trays) < self.capacity:
            tray.current_location = LocationType.STORAGE
            self.trays.append(tray)
            return True
        return False
    
    def get_tray(self, filter_fn=None) -> Optional[Tray]:
        """
        Get a tray from storage.
        Optional filter function to get specific type of tray.
        """
        if filter_fn:
            for i, tray in enumerate(self.trays):
                if filter_fn(tray):
                    return self.trays.pop(i)
            return None
        elif self.trays:
            return self.trays.pop(0)
        return None
    
    def count(self) -> int:
        return len(self.trays)
