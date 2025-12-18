
from typing import Any, Dict, List, Optional, Set
import simpy

import FJSPSimulation
from .BaseAgent import BaseAgent
from enums.AgentType import AgentType
from enums.ProductType import ProductType
from models.Tray import Tray
from gymnasium import spaces
from utils.ObservationSpaces import ObservationSpaces
from utils.ActionSpaces import ActionSpaces
import numpy as np

class MachineAgent(BaseAgent):
    """
    Machine Agent (Base for Small and Big machines)
    
    Responsibilities:
    - Process products (change is_processed flag)
    - Manage processing queue
    - Signal completion
    
    SimPy Integration:
    - Processing uses env.timeout() for duration
    - Resource modeling for single-machine constraint
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 env: simpy.Environment, simulation: FJSPSimulation,
                 processing_time: int, compatible_types: Set[ProductType]):
        super().__init__(agent_id, agent_type, env)
        self.simulation = simulation
        self.processing_time = processing_time
        self.compatible_types = compatible_types
        self.is_busy = False
        
        # State
        self.tray_queue: List[Tray] = []
        self.current_tray: Optional[Tray] = None
        self.processing_progress: float = 0.0  # 0.0 to 1.0
        self.processing_start_time: Optional[float] = None
        self.ready_trays: List[Tray] = []  # Processed, waiting for pickup
        
        # SimPy resource (ensures only one tray processed at a time)
        self.resource = simpy.Resource(env, capacity=1)
    
    def get_observation_space(self) -> spaces.Space:
        if self.agent_type == AgentType.SMALL_MACHINE:
            return ObservationSpaces.small_machine()
        return ObservationSpaces.big_machine()
    
    def get_action_space(self) -> spaces.Space:
        if self.agent_type == AgentType.SMALL_MACHINE:
            return ActionSpaces.small_machine()
        return ActionSpaces.big_machine()
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build observation with busy status, progress, queue info."""

        observation = {
            'is_busy': np.array(int(self.is_busy), dtype=np.int8),
            'processing_progress': np.array(self.processing_progress, dtype=np.float32),
            'queue_length': np.array(len(self.tray_queue), dtype=np.int8),
            # 'queue_product_types': np.zeros(5, dtype=np.int8),  # Up to 5 trays in queue
        }

        return observation
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Actions:
        0 = IDLE
        1 = START_PROCESSING - begin processing next tray in queue
        2 = SIGNAL_COMPLETE - move processed tray to ready_trays
        """
        result = {
            'action': action,
            'success': False,
            'started_processing': False,
            'completed_processing': False,
            'idle_with_queue': False,
        }
        
        if action == 0:  # IDLE
            result['idle_with_queue'] = len(self.tray_queue) > 0 and not self.is_busy
            result['success'] = True
            
        elif action == 1:  # START_PROCESSING
            # Check if not busy and queue not empty
            # Start SimPy processing process
            # Filter products in tray that this machine can process

            if self.tray_queue and not self.is_busy:
                tray_to_process = self.tray_queue.pop(0)
                self.env.process(self._simpy_processing_process(tray_to_process))
                result['started_processing'] = True
                result['success'] = True

            
            
        elif action == 2:  # SIGNAL_COMPLETE
            # Check if processing complete
            # Move tray to ready_trays

            if not self.is_busy and self.current_tray:
                self.ready_trays.append(self.current_tray)
                self.current_tray = None
                result['completed_processing'] = True
                result['success'] = True

        
        return result
    
    def add_tray(self, tray: Tray):
        """Called when AGV drops tray at this machine."""
        self.tray_queue.append(tray)
    
    def get_ready_tray(self) -> Optional[Tray]:
        """Called by AGV to pickup processed tray."""
        if self.ready_trays:
            return self.ready_trays.pop(0)
        return None
    
    def _simpy_processing_process(self, tray: Tray):
        """
        SimPy process for machine processing.
        Processes all compatible products in the tray.
        """
        with self.resource.request() as req:
            yield req
            
            self.is_busy = True
            self.current_tray = tray
            self.processing_start_time = self.env.now
            
            # Process each compatible product
            for product in tray.products:
                if product.product_type in self.compatible_types and not product.is_processed:
                    yield self.env.timeout(self.processing_time)
                    product.is_processed = True
            
            self.is_busy = False
            self.processing_progress = 1.0