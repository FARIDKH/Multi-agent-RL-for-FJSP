

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import simpy
from enums.AgentType import AgentType
from gymnasium import spaces


class BaseAgent(ABC):
    """Abstract base class for all agents in the simulation."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, env: simpy.Environment):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.env = env  # SimPy environment reference
        
        # State tracking
        self.is_busy: bool = False
        self.current_action: Optional[int] = None
        self.last_action_result: Dict[str, Any] = {}
    
    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return current observation for this agent."""
        pass
    
    @abstractmethod
    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute the given action.
        Returns dict with action results for reward calculation.
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Return the observation space for this agent."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Return the action space for this agent."""
        pass