
from   typing import Dict, Tuple
from   FJSPSimulation import FJSPSimulation
from   gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv
from enums.LocationType import LocationType 

class FJSPParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel API wrapper for the FJSP simulation.
    
    This wraps FJSPSimulation to provide the standard PettingZoo interface
    that works with the training loop:
    
    while parallel_env.agents:
        actions = {agent: ... for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    """
    
    metadata = {
        "name": "fjsp_v1",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }
    
    def __init__(self, config: Dict = None, render_mode: str = None):
        self.simulation = FJSPSimulation(config)
        self.render_mode = render_mode
        
        # PettingZoo required attributes
        self.possible_agents = self.simulation.get_agent_ids()
        self.agents = self.possible_agents.copy()  # Currently active agents
    
    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for agent."""
        return self.simulation.observation_space(agent)
    
    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for agent."""
        return self.simulation.action_space(agent)
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """
        Reset environment.
        Returns: (observations, infos)

        Options:
            num_orders: int - Number of orders to generate (default: 30)
        """
        self.agents = self.possible_agents.copy()
        num_orders = options.get('num_orders') if options else None
        observations, infos = self.simulation.reset(seed, num_orders=num_orders)
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute step with actions for all agents.
        Returns: (observations, rewards, terminations, truncations, infos)
        """
        observations, rewards, terminations, truncations, infos = self.simulation.step(actions)
        
        # Remove terminated agents from active list
        self.agents = [
            agent for agent in self.agents
            if not terminations.get(agent, False) and not truncations.get(agent, False)
        ]
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment (for visualization)."""
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_grid()
    
    def _render_text(self):
        sim = self.simulation
        print(f"\n{'='*60}")
        print(f"Step {sim.current_step} | SimTime: {sim.env.now}")
        print(f"{'='*60}")
        print(f"Orders: {len(sim.completed_orders)}/{len(sim.orders)} complete")
        print(f"Products packaged: {sim.total_products_packaged}")
        print(f"\nAGV: pos={sim.agv.position}, carrying={'Yes' if sim.agv.carrying_tray else 'No'}")
        print(f"Pickup: orders_queue={len(sim.pickup_station.order_queue)}, ready_trays={len(sim.pickup_station.ready_trays)}")
        print(f"Small Machine: busy={sim.small_machine.is_busy}, queue={len(sim.small_machine.tray_queue)}")
        print(f"Big Machine: busy={sim.big_machine.is_busy}, queue={len(sim.big_machine.tray_queue)}")
        print(f"Storage: {len(sim.storage.trays)} trays")
        for color, station in sim.packaging_stations.items():
            print(f"Packaging {color.name}: busy={station.is_busy}, queue={len(station.tray_queue)}")
    
    def _render_grid(self) -> np.ndarray:
        """Return RGB array of grid world."""
        grid = np.ones((4, 6, 3), dtype=np.uint8) * 255
        
        # Mark locations
        locations = {
            LocationType.PICKUP: (0, 0, [0, 255, 0]),      # Green
            LocationType.SMALL_MACHINE: (2, 3, [0, 0, 255]),  # Blue
            LocationType.BIG_MACHINE: (0, 3, [255, 0, 0]),    # Red
            LocationType.STORAGE: (3, 0, [128, 128, 128]),    # Gray
            LocationType.PACKAGING: (3, 5, [255, 255, 0]),    # Yellow
        }
        
        for loc, (r, c, color) in locations.items():
            grid[r, c] = color
        
        # Mark AGV position
        agv_pos = self.simulation.agv.position
        grid[agv_pos[0], agv_pos[1]] = [0, 0, 0]  # Black
        
        return grid
    
    def close(self):
        """Clean up resources."""
        pass
    
    def state(self) -> np.ndarray:
        """Return global state for centralized critic."""
        all_obs = []
        for agent_id in self.possible_agents:
            obs = self.simulation.agents[agent_id].get_observation()
            for key in sorted(obs.keys()):
                all_obs.append(obs[key].flatten())
        
        # Add global info
        global_info = np.array([
            len(self.simulation.orders),
            len(self.simulation.completed_orders),
            self.simulation.total_products_packaged,
            self.simulation.env.now,
        ], dtype=np.float32)
        all_obs.append(global_info)
        
        return np.concatenate(all_obs)

