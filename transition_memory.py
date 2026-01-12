from typing import List, Dict, Tuple
import numpy as np
import torch


class MultiAgentTransitionMemory:
    """
    Stores transitions for all agents.
    Adapts TransitionMemoryAdvantage for multi-agent setting.
    """
    
    def __init__(self, agent_ids: List[str], gamma: float, lamb: float, use_gae: bool = True):
        self.agent_ids = agent_ids
        self.gamma = gamma
        self.lamb = lamb
        self.use_gae = use_gae
        
        # Per-agent storage
        self.obs_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.action_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.reward_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.logprob_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.value_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.return_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        self.adv_lst: Dict[str, List] = {agent: [] for agent in agent_ids}
        
        # Track trajectory start per agent
        self.traj_start: Dict[str, int] = {agent: 0 for agent in agent_ids}
    
    def put(self,
            observations: Dict[str, np.ndarray],
            actions: Dict[str, int],
            rewards: Dict[str, float],
            logprobs: Dict[str, torch.Tensor],
            values: Dict[str, float]):
        """Store one timestep for all agents."""
        for agent_id in self.agent_ids:
            if agent_id in observations:
                self.obs_lst[agent_id].append(observations[agent_id])
                self.action_lst[agent_id].append(actions[agent_id])
                self.reward_lst[agent_id].append(rewards[agent_id])
                self.logprob_lst[agent_id].append(logprobs[agent_id])
                self.value_lst[agent_id].append(values[agent_id])
    
    def finish_trajectory(self, next_values: Dict[str, float]):
        """
        Compute returns and advantages for completed trajectory.
        
        Parameters
        ----------
        next_values : Dict[str, float]
            Value of final state per agent (0.0 if terminal, V(s) if truncated)
        """
        for agent_id in self.agent_ids:
            traj_start = self.traj_start[agent_id]
            reward_traj = self.reward_lst[agent_id][traj_start:]
            value_traj = self.value_lst[agent_id][traj_start:]
            next_value = next_values.get(agent_id, 0.0)
            
            if len(reward_traj) == 0:
                continue
            
            # Use detached scalars for return/advantage computation to avoid holding graphs
            value_traj_detached = [
                v.detach().item() if torch.is_tensor(v) else float(v)
                for v in value_traj
            ]
            
            # Compute returns
            return_traj = self._compute_returns(reward_traj, next_value)
            self.return_lst[agent_id].extend(return_traj)
            
            # Compute advantages
            if self.use_gae:
                adv_traj = self._compute_gae(reward_traj, value_traj_detached, next_value)
            else:
                adv_traj = self._compute_advantages(return_traj, value_traj_detached)
            self.adv_lst[agent_id].extend(adv_traj)
            
            # Update trajectory start
            self.traj_start[agent_id] = len(self.reward_lst[agent_id])
    
    def _compute_returns(self, rewards: List[float], next_value: float) -> List[float]:
        """Compute discounted returns."""
        returns = []
        ret = next_value
        for reward in reversed(rewards):
            ret = reward + self.gamma * ret
            returns.append(ret)
        return returns[::-1]
    
    def _compute_advantages(self, returns: List[float], values: List[float]) -> List[float]:
        """Compute simple advantages: A = R - V."""
        return [ret - val for ret, val in zip(returns, values)]
    
    def _compute_gae(self, rewards: List[float], values: List[float], next_value: float) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        gae = 0
        advantages = []
        for reward, value in zip(reversed(rewards), reversed(values)):
            td_error = reward + self.gamma * next_value - value
            gae = td_error + self.gamma * self.lamb * gae
            advantages.append(gae)
            next_value = value
        return advantages[::-1]
    
    def get(self, agent_id: str) -> Tuple[List, List, List, List, List, List, List]:
        """Get stored data for specific agent."""
        return (
            self.obs_lst[agent_id],
            self.action_lst[agent_id],
            self.reward_lst[agent_id],
            self.logprob_lst[agent_id],
            self.return_lst[agent_id],
            self.value_lst[agent_id],
            self.adv_lst[agent_id]
        )
    
    def clear(self):
        """Clear all stored transitions."""
        for agent_id in self.agent_ids:
            self.obs_lst[agent_id] = []
            self.action_lst[agent_id] = []
            self.reward_lst[agent_id] = []
            self.logprob_lst[agent_id] = []
            self.value_lst[agent_id] = []
            self.return_lst[agent_id] = []
            self.adv_lst[agent_id] = []
            self.traj_start[agent_id] = 0

    def has_data(self) -> bool:
        """Check if memory has any data."""
        return any(len(self.obs_lst[agent]) > 0 for agent in self.agent_ids)
