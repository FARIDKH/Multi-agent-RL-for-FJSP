

from dataclasses import dataclass
from typing import Dict, Any
from enums.AgentType import AgentType

@dataclass
class RewardModel:
    """Calculates rewards for CTDE MARL."""
    
    # Reward weights (tunable hyperparameters)
    ORDER_COMPLETE_REWARD: float = 100.0
    THROUGHPUT_BONUS: float = 10.0
    TIME_PENALTY: float = -0.1
    
    # Local rewards
    PICKUP_LOAD_REWARD: float = 1.0
    PICKUP_TRAY_COMPLETE: float = 5.0
    PICKUP_IDLE_PENALTY: float = -1.0
    
    AGV_DELIVERY_REWARD: float = 2.0
    AGV_MOVE_PENALTY: float = -0.1
    AGV_PACKAGING_DELIVERY: float = 10.0
    AGV_INVALID_ACTION: float = -5.0
    
    MACHINE_COMPLETE_REWARD: float = 5.0
    MACHINE_START_REWARD: float = 1.0
    MACHINE_IDLE_PENALTY: float = -2.0
    
    PACKAGING_COMPLETE_REWARD: float = 20.0
    PACKAGING_START_REWARD: float = 2.0
    PACKAGING_IDLE_PENALTY: float = -1.0
    
    def calculate_global_reward(self, 
                                orders_completed: int,
                                products_packaged: int,
                                time_elapsed: float) -> float:
        """Calculate shared global reward component."""

        reward = self.ORDER_COMPLETE_REWARD * orders_completed
        reward += self.THROUGHPUT_BONUS * products_packaged
        reward += self.TIME_PENALTY * time_elapsed

        return reward
    
    def calculate_local_reward(self,
                               agent_type: AgentType,
                               action_taken: int,
                               action_result: Dict[str, Any]) -> float:
        """Calculate individual agent's local reward."""
        reward = 0.0
        match agent_type:
            case AgentType.PICKUP_STATION:
                if action_result.get('product_loaded', False):
                    reward += self.PICKUP_LOAD_REWARD
                if action_result.get('tray_completed', False):
                    reward += self.PICKUP_TRAY_COMPLETE
                # Only penalize idle when there are orders waiting
                if action_taken == 0 and action_result.get('idle_with_orders', False):
                    reward += self.PICKUP_IDLE_PENALTY

            case AgentType.AGV:
                # Reward for successful pickup
                if action_result.get('pickup_success', False):
                    reward += self.AGV_DELIVERY_REWARD
                # Reward for successful drop
                if action_result.get('drop_success', False):
                    reward += self.AGV_DELIVERY_REWARD
                # Extra reward for delivering to packaging
                if action_result.get('delivered_to_packaging', False):
                    reward += self.AGV_PACKAGING_DELIVERY
                # Small penalty for movement (encourages efficiency)
                if action_result.get('moved', False):
                    reward += self.AGV_MOVE_PENALTY
                # Penalty for invalid actions
                if action_result.get('invalid_action', False):
                    reward += self.AGV_INVALID_ACTION

            case AgentType.SMALL_MACHINE | AgentType.BIG_MACHINE:
                if action_result.get('started_processing', False):
                    reward += self.MACHINE_START_REWARD
                if action_result.get('completed_processing', False):
                    reward += self.MACHINE_COMPLETE_REWARD
                # Only penalize idle when there's work waiting
                if action_taken == 0 and action_result.get('idle_with_queue', False):
                    reward += self.MACHINE_IDLE_PENALTY

            case AgentType.PACKAGING:
                if action_result.get('started_packaging', False):
                    reward += self.PACKAGING_START_REWARD
                if action_result.get('completed_packaging', False):
                    reward += self.PACKAGING_COMPLETE_REWARD
                # Only penalize idle when there's work waiting
                if action_taken == 0 and action_result.get('idle_with_queue', False):
                    reward += self.PACKAGING_IDLE_PENALTY

        return reward
    
    def combine_rewards(self,
                        global_reward: float,
                        local_rewards: Dict[str, float],
                        num_agents: int) -> Dict[str, float]:
        """
        Combine global and local rewards for each agent.
        Returns: {agent_id: final_reward}
        """
        final_rewards = {}
        for agent_id, local_reward in local_rewards.items():
            final_rewards[agent_id] = global_reward / num_agents + local_reward
        return final_rewards
