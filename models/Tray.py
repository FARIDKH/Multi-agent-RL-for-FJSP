from dataclasses import dataclass, field
from typing import List, Optional
from enums.LocationType import LocationType
from models.Product import Product


@dataclass
class Tray:
    """A tray that holds products during transportation."""
    id: int
    capacity: int = 5
    products: List[Product] = field(default_factory=list)
    current_location: Optional[LocationType] = None
    
    def is_full(self) -> bool:
        return len(self.products) >= self.capacity
    
    def is_empty(self) -> bool:
        return len(self.products) == 0
    
    @property
    def tray_color(self) -> Optional[str]:
        """Determine tray color based on products it holds."""
        if not self.products:
            return None
        return self.products[0].packaging_color.name
    
    @property
    def tray_type(self) -> Optional[str]:
        """Determine tray type based on products it holds."""
        if not self.products:
            return None
        return self.products[0].product_type.name
    
    