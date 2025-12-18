
from dataclasses import dataclass
from typing import List, Optional
from models.Product import Product




@dataclass
class Order:
    """An order containing multiple products."""
    id: int
    products: List[Product]
    arrival_time: float
    
    # State tracking
    is_complete: bool = False
    completion_time: Optional[float] = None
    
    def check_completion(self) -> bool:
        """Check if all products in order are packaged."""
        # All products packaged means order complete
        pass