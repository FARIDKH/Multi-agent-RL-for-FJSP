
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import simpy
import FJSPSimulation
from enums.AgentType import AgentType
from enums.PackagingColor import PackagingColor
from agents.BaseAgent import BaseAgent
from models.Product import Product
from utils.ObservationSpaces import ObservationSpaces
from utils.ActionSpaces import ActionSpaces
from gymnasium import spaces
from constants import PROCESSING_TIMES 
import numpy as np    

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
        self.resource = simpy.Resource(env, capacity=1)
    
    def get_observation_space(self) -> spaces.Space:
        return ObservationSpaces.packaging(self.color)
    
    def get_action_space(self) -> spaces.Space:
        return ActionSpaces.packaging()
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build observation with busy status, progress, queue for this color."""
        pass
    
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
            # Check if not busy and queue not empty
            # Start SimPy packaging process
            pass
            
        elif action == 2:  # SIGNAL_COMPLETE
            # Check if packaging complete
            # Mark product as packaged
            # Check if order is now complete
            pass
        
        return result
    
    def add_product(self, product: Product):
        """Called when a processed product arrives for packaging."""
        if product.packaging_color == self.color:
            self.product_queue.append(product)
    
    def _simpy_packaging_process(self, product: Product):
        """SimPy process for packaging."""
        with self.resource.request() as req:
            yield req
            
            self.is_busy = True
            self.current_product = product
            
            yield self.env.timeout(self.processing_time)
            
            product.is_packaged = True
            self.products_completed += 1
            self.is_busy = False


