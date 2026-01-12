
from typing import List, Dict, Any, Optional
import simpy
import FJSPSimulation
from enums.AgentType import AgentType
from enums.PackagingColor import PackagingColor
from agents.BaseAgent import BaseAgent
from models import Tray
from models.Product import Product
from utils.ObservationSpaces import ObservationSpaces
from utils.ActionSpaces import ActionSpaces
from utils.Logger import get_logger
from gymnasium import spaces
from constants import PROCESSING_TIMES
import numpy as np

logger = get_logger()


class PackagingAgent(BaseAgent):
    """
    Packaging Agent (one per color station)

    Responsibilities:
    - Package products of matching color
    - Multiple stations per color (2 blue, 1 red, 1 green)

    SimPy Integration:
    - Packaging uses env.timeout() for duration
    - Each station is independent SimPy resource
    """

    def __init__(self, agent_id: str, color: PackagingColor,
                 env: simpy.Environment, simulation: FJSPSimulation):
        super().__init__(agent_id, AgentType.PACKAGING, env)
        self.simulation = simulation
        self.color = color
        self.processing_time = PROCESSING_TIMES['packaging']

        # State
        self.product_queue: List[Product] = []
        self.current_product: Optional[Product] = None
        self.processing_progress: float = 0.0
        self.products_completed: int = 0
        # SimPy resource
        self.resource = simpy.Resource(env, capacity=20)

    def get_observation_space(self) -> spaces.Space:
        return ObservationSpaces.packaging(self.color)

    def get_action_space(self) -> spaces.Space:
        return ActionSpaces.packaging()

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build observation with busy status, progress, queue for this color."""
        observation = {
            "is_busy": np.array(int(self.is_busy), dtype=np.int8),
            'processing_progress': np.array(self.processing_progress, dtype=np.float32),
            'queue_length': np.array(len(self.product_queue), dtype=np.int8),
            'action_mask': self.get_action_mask().astype(np.int8),
        }
        return observation

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for valid actions based on current state.
        1 = Valid, 0 = Invalid.

        Action Space (3):
        0: IDLE - always valid
        1: START_PACKAGING - valid if queue not empty AND not busy AND has capacity
        2: SIGNAL_COMPLETE - valid if not busy AND current_product exists
        """
        mask = np.zeros(3, dtype=np.int32)

        # 0. IDLE is always valid
        mask[0] = 1

        # 1. START_PACKAGING
        # Valid if: product_queue not empty AND not busy AND has resource capacity
        if len(self.product_queue) > 0 and not self.is_busy and self.has_capacity():
            mask[1] = 1

        # 2. SIGNAL_COMPLETE
        # Valid if: not busy AND current_product exists
        if not self.is_busy and self.current_product is not None:
            mask[2] = 1

        return mask

    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Actions:
        0 = IDLE
        1 = START_PACKAGING - begin packaging next product
        2 = SIGNAL_COMPLETE - mark product as packaged, remove from system
        """
        result = {
            'action': action,
            'success': False,
            'started_packaging': False,
            'completed_packaging': False,
            'idle_with_queue': False,
            'products_completed_this_step': 0,
        }

        if action == 0:  # IDLE
            result['idle_with_queue'] = len(self.product_queue) > 0 and not self.is_busy
            result['success'] = True

        elif action == 1:  # START_PACKAGING
            i = 1
            for product in self.product_queue:
                logger.packaging_start(self.agent_id, product.id, order_id=product.order_id)
                self.env.process(self._simpy_packaging_process(product))
                result['started_packaging'] = True
                self.processing_progress = (i / len(self.product_queue)) * 100
                result['success'] = True

        elif action == 2:  # SIGNAL_COMPLETE
            if not self.is_busy and self.current_product:
                result["products_completed_this_step"] = self.products_completed
                result["completed_packaging"] = True

        return result

    def add_tray(self, tray: Tray):
        """Called when a processed products arrives for packaging."""
        for product in tray.products:
            if product.packaging_color == self.color:
                self.product_queue.append(product)

    def _simpy_packaging_process(self, product: Product):
        """SimPy process for packaging."""
        with self.resource.request() as req:
            yield req

            self.is_busy = True
            self.current_product = product
            self.product_queue.remove(product)
            yield self.env.timeout(self.processing_time)

            product.is_packaged = True
            self.products_completed += 1
            self.simulation.total_products_packaged += 1  # Update global counter
            logger.packaging_complete(self.agent_id, product.id, order_id=product.order_id)
            self.is_busy = False

    def has_capacity(self) -> bool:
        """Check if the packaging station has capacity to process more products."""
        if self.resource.count < self.resource.capacity:
            return True
        return False
