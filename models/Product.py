from dataclasses import dataclass
from typing import Optional
from enums.ProductType import ProductType
from enums.PackagingColor import PackagingColor
from enums.LocationType import LocationType

@dataclass
class Product:
    """A single product to be manufactured."""
    id: int
    product_type: ProductType
    packaging_color: PackagingColor
    order_id: int
    
    # State tracking
    is_processed: bool = False  # Has been through machine
    is_packaged: bool = False   # Has been through packaging
    current_location: Optional[LocationType] = None
    tray_id: Optional[int] = None