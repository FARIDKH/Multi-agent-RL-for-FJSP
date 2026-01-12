
from typing import Any, Dict, List, Optional
import simpy

import FJSPSimulation
from utils.ActionSpaces import ActionSpaces
from utils.ObservationSpaces import ObservationSpaces
from utils.Logger import get_logger
from .BaseAgent import BaseAgent

from enums.AgentType import AgentType
from enums.LocationType import LocationType
from enums.ProductType import ProductType

from models.Order import Order
from models.Tray import Tray
from gymnasium import spaces
from constants import CONFIG

import numpy as np

logger = get_logger()


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
        next_product_type: int = 0
        next_product_color: int = 0

        if self.current_order:
            order_size = len(self.current_order.products)
            products_remaining = order_size - self.current_order_product_idx
            if products_remaining > 0:
                next_product = self.current_order.products[self.current_order_product_idx]
                next_product_type = int(next_product.product_type.value)
                next_product_color = int(next_product.packaging_color.value)

        # Current tray details
        current_tray_type = 0
        current_tray_color = 0
        current_tray_count = 0

        if self.current_tray:
            current_tray_count = len(self.current_tray.products)
            if current_tray_count > 0:
                first_product = self.current_tray.products[0]
                current_tray_type = int(first_product.product_type.value)
                current_tray_color = int(first_product.packaging_color.value)

        observation = {
            'order_size': np.array(order_size, dtype=np.int32),
            'products_remaining': np.array(products_remaining, dtype=np.int32),
            'next_product_type': np.array(next_product_type, dtype=np.int32),
            'next_product_color': np.array(next_product_color, dtype=np.int32),
            'current_tray_type': np.array(current_tray_type, dtype=np.int32),
            'current_tray_color': np.array(current_tray_color, dtype=np.int32),
            'current_tray_count': np.array(current_tray_count, dtype=np.int32),
            'action_mask': self.get_action_mask().astype(np.int8),
        }

        return observation

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for valid actions based on current state.
        1 = Valid, 0 = Invalid.

        Action Space (3):
        0: IDLE - always valid
        1: LOAD_NEXT_PRODUCT - valid if order and tray available, tray not full
        2: SIGNAL_TRAY_READY - valid if current tray has products
        """
        mask = np.zeros(3, dtype=np.int32)

        # 0. IDLE is always valid
        mask[0] = 1

        # 1. LOAD_NEXT_PRODUCT
        # Valid if: (current_order OR order_queue not empty) AND
        #           (current_tray OR trays_at_station not empty) AND
        #           tray not full AND products remaining
        has_order = self.current_order is not None or len(self.order_queue) > 0
        has_tray = self.current_tray is not None or len(self.trays_at_station) > 0

        # Check if current tray is full
        tray_not_full = True
        if self.current_tray:
            tray_not_full = len(self.current_tray.products) < CONFIG['tray_capacity']

        # Check if there are products remaining in the order
        products_remaining = False
        if self.current_order:
            products_remaining = self.current_order_product_idx < len(self.current_order.products)
        elif len(self.order_queue) > 0:
            products_remaining = True  # New order will have products

        if has_order and has_tray and tray_not_full and products_remaining:
            mask[1] = 1

        # 2. SIGNAL_TRAY_READY
        # Valid if current tray exists and has at least one product
        if self.current_tray and len(self.current_tray.products) > 0:
            mask[2] = 1

        return mask



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
            if not self.current_order:
                if self.order_queue:
                    self.current_order = self.order_queue.pop(0)
                    self.current_order_product_idx = 0
                else:
                    return result  # No order to load from

            if not self.current_tray:
                if self.trays_at_station:
                    self.current_tray = self.trays_at_station.pop(0)
                    self.current_tray.order_id = self.current_order.id
                else:
                    return result  # No tray available

            # Update product's location
            product = self.current_order.products[self.current_order_product_idx]
            product.current_location = LocationType.PICKUP

            # Check if tray is full or order products all loaded
            if not self.current_tray.is_full():
                if self.current_order.id != self.current_tray.order_id:
                    self.ready_trays.append(self.current_tray)
                    self.current_tray = None
                    result['tray_completed'] = True
                    return result

                self.current_tray.products.append(product)
                self.current_order_product_idx += 1
                logger.product_loaded(self.current_order.id, product.id, self.current_tray.id)

                result['product_loaded'] = True
                result['success'] = True

                # Check if order is complete
                if self.current_order_product_idx >= len(self.current_order.products):
                    self.current_order = None
                    self.current_order_product_idx = 0
                    self.ready_trays.append(self.current_tray)
                    logger.tray_ready(self.current_tray.id, 'PICKUP', len(self.current_tray.products))
                    self.current_tray = None
                    result['tray_completed'] = True
                    return result

                # Check if tray is now full
                if self.current_tray.is_full():
                    self.ready_trays.append(self.current_tray)
                    logger.tray_ready(self.current_tray.id, 'PICKUP', len(self.current_tray.products))
                    self.current_tray = None
                    result['tray_completed'] = True
                    return result
            else:
                # Tray is full, cannot load more
                self.ready_trays.append(self.current_tray)
                logger.tray_ready(self.current_tray.id, 'PICKUP', len(self.current_tray.products))
                self.current_tray = None
                result['tray_completed'] = True
                return result

        elif action == 2:  # SIGNAL_TRAY_READY
            if self.current_tray and len(self.current_tray.products) > 0:
                self.ready_trays.append(self.current_tray)
                logger.tray_ready(self.current_tray.id, 'PICKUP', len(self.current_tray.products))
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

    def add_empty_tray(self, tray: Tray):
        """Called when AGV returns an empty tray."""
        tray.products = []
        tray.order_id = None
        self.trays_at_station.append(tray)
