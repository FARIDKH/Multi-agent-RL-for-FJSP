
from  typing import Any, Dict, List, Optional
import simpy

import FJSPSimulation
from utils.ActionSpaces import ActionSpaces
from utils.ObservationSpaces import ObservationSpaces
from .BaseAgent import BaseAgent

from enums.AgentType import AgentType
from enums.LocationType import LocationType

from models.Order import Order
from models.Tray import Tray
from gymnasium import spaces

import numpy as np

class PickupStationAgent(BaseAgent):

    """
    Pickup Station Agent
    
    Responsibilities:
    - Receive incoming orders
    - Load products onto trays
    - Signal when trays are ready for AGV
    
    SimPy Integration:
    - No processing time (loading is instantaneous in our model)
    - Maintains queue of orders and available trays
    """
    
    def __init__(self, env: simpy.Environment, simulation: FJSPSimulation):
        super().__init__("pickup_station", AgentType.PICKUP_STATION, env)
        self.simulation = simulation
        
        # State
        self.order_queue: List[Order] = []
        self.current_order: Optional[Order] = None
        self.current_order_product_idx: int = 0
        self.trays_at_station: List[Tray] = []
        self.current_tray: Optional[Tray] = None
        self.ready_trays: List[Tray] = []  # Trays ready for AGV pickup
    
    def get_observation_space(self) -> spaces.Space:
        return ObservationSpaces.pickup_station()
    
    def get_action_space(self) -> spaces.Space:
        return ActionSpaces.pickup_station()
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        
        """Return current observation for Pickup Station."""
        # Current order details
        order_size = 0
        products_remaining = 0
        next_product_type = 0
        next_product_color = 0
        
        if self.current_order:
            order_size = len(self.current_order.products)
            products_remaining = order_size - self.current_order_product_idx
            if products_remaining > 0:
                next_product = self.current_order.products[self.current_order_product_idx]
                next_product_type = next_product.product_type
                next_product_color = next_product.packaging_color
        
        # Current tray details
        current_tray_type = 0
        current_tray_color = 0
        current_tray_count = 0
        
        if self.current_tray:
            current_tray_count = len(self.current_tray.products)
            if current_tray_count > 0:
                first_product = self.current_tray.products[0]
                current_tray_type = first_product.product_type
                current_tray_color = first_product.packaging_color
        
        observation = {
            'order_size': np.array(order_size, dtype=np.int32),
            'products_remaining': np.array(products_remaining, dtype=np.int32),
            'next_product_type': np.array(next_product_type, dtype=np.int32),
            'next_product_color': np.array(next_product_color, dtype=np.int32),
            'current_tray_type': np.array(current_tray_type, dtype=np.int32),
            'current_tray_color': np.array(current_tray_color, dtype=np.int32),
            'current_tray_count': np.array(current_tray_count, dtype=np.int32),
        }
        
        return observation
    
        

    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Action 0: IDLE - do nothing
        Action 1: LOAD_NEXT_PRODUCT - take next product from order, put on tray
        Action 2: SIGNAL_TRAY_READY - mark current tray as ready for AGV
        
        Returns result dict for reward calculation.
        """
        result = {
            'action': action,
            'success': False,
            'product_loaded': False,
            'tray_completed': False,
            'idle_with_orders': False,
        }
        
        if action == 0:  # IDLE
            result['idle_with_orders'] = len(self.order_queue) > 0 or self.current_order is not None
            result['success'] = True
            
        elif action == 1:  # LOAD_NEXT_PRODUCT
            # Check if we have an order and a tray
            if not self.current_order:
                if self.order_queue:
                    self.current_order = self.order_queue.pop(0)
                    self.current_order_product_idx = 0
                else:
                    return result  # No order to load from
            # Load product onto tray
            if not self.current_tray:
                # Get new empty tray if available
                if self.trays_at_station:
                    self.current_tray = self.trays_at_station.pop(0)
                else:
                    return result  # No tray available

            # Update product's location and tray_id
            product = self.current_order.products[self.current_order_product_idx]
            product.current_location = LocationType.PICKUP  # On tray at pickup station


            # Check if tray is full or order products all loaded
            if not self.current_tray.is_full():
                self.current_tray.products.append(product)
                self.current_order_product_idx += 1
                result['product_loaded'] = True
                result['success'] = True
                
                # Check if order is complete
                if self.current_order_product_idx >= len(self.current_order.products):
                    self.current_order = None
                    self.current_order_product_idx = 0
                
                # Check if tray is now full
                if self.current_tray.is_full():
                    result['tray_completed'] = True
            else:
                # Tray is full, cannot load more 
                return result
        elif action == 2:  # SIGNAL_TRAY_READY
            # Move current tray to ready_trays list
            # Get new empty tray if available
            if self.current_tray and len(self.current_tray.products) > 0:
                self.ready_trays.append(self.current_tray)
                self.current_tray = None
                result['success'] = True

        
        return result
    
    def add_order(self, order: Order):
        """Called by simulation to add new order to queue."""
        self.order_queue.append(order)
    
    def get_ready_tray(self) -> Optional[Tray]:
        """Called by AGV to pick up a ready tray."""
        if self.ready_trays:
            return self.ready_trays.pop(0)
        return None