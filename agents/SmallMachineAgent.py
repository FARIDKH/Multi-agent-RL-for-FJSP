from typing import TYPE_CHECKING
import simpy

import FJSPSimulation
from .MachineAgent import MachineAgent
from enums.AgentType import AgentType
from enums.ProductType import ProductType
from constants import PROCESSING_TIMES


class SmallMachineAgent(MachineAgent):
    """Small Machine - processes SMALL and MEDIUM products."""
    
    def __init__(self, env: simpy.Environment, simulation: FJSPSimulation):
        super().__init__(
            agent_id="small_machine",
            agent_type=AgentType.SMALL_MACHINE,
            env=env,
            simulation=simulation,
            processing_time=PROCESSING_TIMES['small_machine'],
            compatible_types={ProductType.SMALL, ProductType.MEDIUM}
        )