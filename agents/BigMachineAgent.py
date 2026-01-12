from typing import TYPE_CHECKING
import simpy

from .MachineAgent import MachineAgent, FJSPSimulation
from enums.AgentType import AgentType
from enums.ProductType import ProductType
from constants import PROCESSING_TIMES


class BigMachineAgent(MachineAgent):
    """Big Machine - processes BIG and MEDIUM products."""
    
    def __init__(self, env: simpy.Environment, simulation: FJSPSimulation):
        super().__init__(
            agent_id="big_machine",
            agent_type=AgentType.BIG_MACHINE,
            env=env,
            simulation=simulation,
            processing_time=PROCESSING_TIMES['big_machine'],
            compatible_types={ProductType.BIG, ProductType.MEDIUM}
        )