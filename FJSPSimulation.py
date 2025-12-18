
from typing import List, Dict, Tuple
from agents.AGVAgent import AGVAgent
from agents.BaseAgent import BaseAgent
from constants import CONFIG
import simpy

from models.Order import Order
from models.Tray import Tray
from models.Product import Product

from models.Storage import Storage
from utils.RewardModel import RewardModel as RewardCalculator

from agents.SmallMachineAgent import SmallMachineAgent
from agents.BigMachineAgent import BigMachineAgent
from agents.PackagingAgent import PackagingAgent
from agents.PickupStationAgent import PickupStationAgent

from enums.PackagingColor import PackagingColor
from enums.ProductType import ProductType

import numpy as np
from gymnasium import spaces


class FJSPSimulation:
    """
    Main simulation orchestrator.
    
    Manages:
    - SimPy environment
    - All agents
    - Order generation
    - State tracking
    - Step-based RL interface
    
    This class bridges SimPy (discrete-event) with RL (step-based).
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        
        # SimPy environment
        self.env = simpy.Environment()
        
        # Initialize components
        self._init_agents()
        self._init_storage()
        self._init_trays()
        
        # Tracking
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.current_step: int = 0
        self.total_products_packaged: int = 0
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()


    def _init_agents(self):
        """Initialize all agents in the simulation."""
        
        self.agv = AGVAgent(self.env, self)
        self.small_machine = SmallMachineAgent(self.env, self)
        self.big_machine = BigMachineAgent(self.env, self)
        self.packaging_stations: Dict[str, PackagingAgent] = {
            'packaging_blue_1': PackagingAgent('packaging_blue_1', PackagingColor.BLUE, self.env, self),
            'packaging_blue_2': PackagingAgent('packaging_blue_2', PackagingColor.BLUE, self.env, self),
            'packaging_red': PackagingAgent('packaging_red', PackagingColor.RED, self.env, self),
            'packaging_green': PackagingAgent('packaging_green', PackagingColor.GREEN, self.env, self),
        }
        self.pickup_station = PickupStationAgent(self.env, self)

        self.agents: Dict[str, BaseAgent] = {
            'pickup_station': self.pickup_station,
            'agv': self.agv,
            'small_machine': self.small_machine,
            'big_machine': self.big_machine,
            **self.packaging_stations
        }


    def _init_storage(self):
        """Initialize storage area."""
        self.storage = Storage(capacity=self.config.get('storage_capacity', 100))

    def _init_trays(self):
        """Initialize tray pool."""
        self.available_trays: List[Tray] = [
            Tray(id=i, capacity=self.config.get('tray_capacity', 5))
            for i in range(self.config['num_trays'])
        ]
        # Give some trays to pickup station initially
        for _ in range(10):
            if self.available_trays:
                self.pickup_station.trays_at_station.append(self.available_trays.pop())


    def generate_order(self, num_products: int = None) -> Order:
        """
        Generate a random order.
        Called to add work to the simulation.
        """
        if num_products is None:
            num_products = np.random.randint(1, 10)
        
        order_id = len(self.orders)
        products = []
        
        for i in range(num_products):
            product = Product(
                id=len(self.orders) * 100 + i,  # Unique ID
                product_type=np.random.choice(list(ProductType)),
                packaging_color=np.random.choice(list(PackagingColor)),
                order_id=order_id,
            )
            products.append(product)
        
        order = Order(
            id=order_id,
            products=products,
            arrival_time=self.env.now
        )
        
        self.orders.append(order)
        self.pickup_station.add_order(order)
        
        return order
    

    def get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get observations for all agents.
        Returns: {agent_id: observation_dict}
        """
        return {
            agent_id: agent.get_observation()
            for agent_id, agent in self.agents.items()
        }
    
    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, Dict],      # observations
        Dict[str, float],     # rewards
        Dict[str, bool],      # terminations
        Dict[str, bool],      # truncations
        Dict[str, Dict]       # infos
    ]:
        """
        Execute one RL step:
        1. Apply actions to all agents
        2. Advance SimPy by step_size
        3. Calculate rewards
        4. Return new observations
        
        This is the main RL interface method.
        """
        # Track state before step for reward calculation
        orders_before = len(self.completed_orders)
        products_before = self.total_products_packaged
        
        # 1. Execute actions for each agent
        action_results = {}
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                action_results[agent_id] = self.agents[agent_id].execute_action(action)
        
        # 2. Advance SimPy simulation by step_size
        # This processes all pending SimPy events (timeouts, etc.)
        target_time = self.env.now + self.config['step_size']
        self.env.run(until=target_time)
        
        # 3. Check for completed orders
        self._check_order_completions()
        
        # 4. Calculate rewards
        orders_completed = len(self.completed_orders) - orders_before
        products_packaged = self.total_products_packaged - products_before
        time_elapsed = self.config['step_size']
        
        global_reward = self.reward_calculator.calculate_global_reward(
            orders_completed, products_packaged, time_elapsed
        )
        
        local_rewards = {
            agent_id: self.reward_calculator.calculate_local_reward(
                self.agents[agent_id].agent_type,
                actions.get(agent_id, 0),
                action_results.get(agent_id, {})
            )
            for agent_id in self.agents
        }
        
        rewards = self.reward_calculator.combine_rewards(
            global_reward, local_rewards, len(self.agents)
        )
        
        # 5. Get new observations
        observations = self.get_observations()
        
        # 6. Check termination conditions
        # Terminate if all orders complete and no pending orders
        all_done = (
            len(self.completed_orders) == len(self.orders) and
            len(self.orders) > 0 and
            len(self.pickup_station.order_queue) == 0
        )
        
        terminations = {agent_id: all_done for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}
        
        # 7. Build info dicts
        infos = {
            agent_id: {
                'action_result': action_results.get(agent_id, {}),
                'sim_time': self.env.now,
                'orders_completed': len(self.completed_orders),
                'total_products_packaged': self.total_products_packaged,
            }
            for agent_id in self.agents
        }
        
        self.current_step += 1
        
        return observations, rewards, terminations, truncations, infos
    

    def _check_order_completions(self):
        """Check and update order completion status."""
        for order in self.orders:
            if not order.is_complete:
                if all(p.is_packaged for p in order.products):
                    order.is_complete = True
                    order.completion_time = self.env.now
                    self.completed_orders.append(order)

    def reset(self, seed: int = None) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Reset simulation to initial state.
        Returns initial observations and info dicts.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset SimPy environment
        self.env = simpy.Environment()
        
        # Reinitialize everything
        self._init_agents()
        self._init_storage()
        self._init_trays()
        
        # Reset tracking
        self.orders = []
        self.completed_orders = []
        self.current_step = 0
        self.total_products_packaged = 0
        
        # Generate initial orders
        for _ in range(5):  # Start with 5 orders
            self.generate_order()
        
        observations = self.get_observations()
        infos = {agent_id: {} for agent_id in self.agents}
        
        return observations, infos
    
    def get_agent_ids(self) -> List[str]:
        """Return list of all agent IDs."""
        return list(self.agents.keys())
    
    def observation_space(self, agent_id: str) -> spaces.Space:
        """Return observation space for given agent."""
        return self.agents[agent_id].get_observation_space()
    
    def action_space(self, agent_id: str) -> spaces.Space:
        """Return action space for given agent."""
        return self.agents[agent_id].get_action_space()