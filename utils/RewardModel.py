

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
                if action_taken == 1:  # LOAD_NEXT_PRODUCT
                    reward += self.PICKUP_LOAD_REWARD
                elif action_taken == 2 and action_result.get('tray_complete', False):
                    reward += self.PICKUP_TRAY_COMPLETE
                elif action_taken == 0:  # IDLE
                    reward += self.PICKUP_IDLE_PENALTY
            case AgentType.AGV:
                if action_taken in [1, 2, 3, 4, 5]:  #
                    reward += self.AGV_MOVE_PENALTY
                elif action_taken == 6 and action_result.get('delivered', True):
                    reward += self.AGV_DELIVERY_REWARD
                elif action_taken == 7 and action_result.get('packaged', True):
                    reward += self.AGV_PACKAGING_DELIVERY
                else:
                    reward += self.AGV_INVALID_ACTION
            case AgentType.SMALL_MACHINE | AgentType.BIG_MACHINE:
                if action_taken == 1:  # START_PROCESSING
                    reward += self.MACHINE_START_REWARD
                elif action_taken == 2 and action_result.get('processing_complete', False):
                    reward += self.MACHINE_COMPLETE_REWARD
                elif action_taken == 0:  # IDLE
                    reward += self.MACHINE_IDLE_PENALTY
            case AgentType.PACKAGING:
                if action_taken == 1:  # START_PACKAGING
                    reward += self.PACKAGING_START_REWARD
                elif action_taken == 2 and action_result.get('packaging_complete', False):
                    reward += self.PACKAGING_COMPLETE_REWARD
                elif action_taken == 0:  # IDLE
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
