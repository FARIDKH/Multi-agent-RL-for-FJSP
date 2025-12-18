from collections import defaultdict
from typing import Any, Dict, List
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

from networks import ActorNetwork, CentralizedCriticNetwork, CriticNetwork
from transition_memory import MultiAgentTransitionMemory


class MultiAgentA2C:
    """
    Multi-Agent Actor-Critic for PettingZoo Parallel API.
    
    CTDE: Centralized Training, Decentralized Execution
    - Each agent has its own Actor network
    - One shared Critic network sees global state (all observations concatenated)
    """
    
    def __init__(self, 
                 env,
                 batch_size: int = 500,
                 gamma: float = 0.99,
                 lamb: float = 0.99,
                 lr_actor: float = 0.005,
                 lr_critic: float = 0.001,
                 use_gae: bool = True):
        """
        Parameters
        ----------
        env : PettingZoo Parallel Environment
            Must have: possible_agents, agents, observation_space(), action_space(), reset(), step()
        batch_size : int
            Number of timesteps per update
        gamma : float
            Discount factor
        lamb : float
            GAE lambda
        lr_actor : float
            Actor learning rate
        lr_critic : float
            Critic learning rate
        use_gae : bool
            Use Generalized Advantage Estimation
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamb = lamb
        
        # Get agent IDs
        self.possible_agents = env.possible_agents
        
        # Calculate observation and action dimensions per agent
        self.obs_dims: Dict[str, int] = {}
        self.act_dims: Dict[str, int] = {}
        
        for agent_id in self.possible_agents:
            obs_space = env.observation_space(agent_id)
            act_space = env.action_space(agent_id)
            
            # Handle Dict observation space
            self.obs_dims[agent_id] = self._get_obs_dim(obs_space)
            self.act_dims[agent_id] = act_space.n
        
        # Global state dimension for centralized critic
        self.global_obs_dim = sum(self.obs_dims.values())
        
        # Create Actor networks (one per agent)
        self.actor_nets: Dict[str, ActorNetwork] = {}
        self.optim_actors: Dict[str, optim.Adam] = {}
        
        for agent_id in self.possible_agents:
            self.actor_nets[agent_id] = ActorNetwork(
                self.obs_dims[agent_id],
                self.act_dims[agent_id]
            )
            self.optim_actors[agent_id] = optim.Adam(
                self.actor_nets[agent_id].parameters(),
                lr=lr_actor
            )
        
        # Create Centralized Critic (single network, sees global state)
        self.critic_net = CentralizedCriticNetwork(self.global_obs_dim)
        self.optim_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
    
        # Transition memory
        self.memory = MultiAgentTransitionMemory(
            self.possible_agents, gamma, lamb, use_gae
        )
    
    def _get_obs_dim(self, obs_space) -> int:
        """Calculate flattened observation dimension."""
        if hasattr(obs_space, 'spaces'):  # Dict space
            total = 0
            for space in obs_space.spaces.values():
                if hasattr(space, 'n'):  # Discrete
                    total += 1
                elif hasattr(space, 'shape'):
                    total += int(np.prod(space.shape))
                else:
                    total += 1
            return total
        elif hasattr(obs_space, 'shape'):
            return int(np.prod(obs_space.shape))
        else:
            return obs_space.n
    
    def _flatten_obs(self, obs: Dict) -> np.ndarray:
        """Flatten a Dict observation to 1D numpy array."""
        if not isinstance(obs, dict):
            return np.array(obs).flatten().astype(np.float32)
        
        values = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, np.ndarray):
                values.extend(val.flatten())
            else:
                values.append(float(val))
        return np.array(values, dtype=np.float32)
    
    def _get_global_state(self, observations: Dict[str, Any], active_agents: List[str]) -> np.ndarray:
        """
        Build global state by concatenating all agent observations.
        For inactive agents, use zeros.
        """
        global_state = []
        for agent_id in self.possible_agents:
            if agent_id in observations:
                obs_flat = self._flatten_obs(observations[agent_id])
            else:
                # Inactive agent - fill with zeros
                obs_flat = np.zeros(self.obs_dims[agent_id], dtype=np.float32)
            global_state.append(obs_flat)
        return np.concatenate(global_state)
    
    def predict(self, 
                observations: Dict[str, Any],
                active_agents: List[str],
                train_returns: bool = False):
        """
        Get actions for all active agents.
        
        Parameters
        ----------
        observations : Dict[str, obs]
            Current observations per agent
        active_agents : List[str]
            List of currently active agent IDs
        train_returns : bool
            If True, return logprobs and values for training
        
        Returns
        -------
        actions : Dict[str, int]
        logprobs : Dict[str, Tensor] (only if train_returns=True)
        values : Dict[str, float] (only if train_returns=True)
        """
        actions = {}
        logprobs = {}
        values = {}
        
        # Get global state for critic
        global_state = self._get_global_state(observations, active_agents)
        global_state_tensor = torch.FloatTensor(global_state)
        
        # Get value from centralized critic (shared by all agents)
        global_value = self.critic_net(global_state_tensor).item()
        
        # Get action from each active agent's actor
        for agent_id in active_agents:
            if agent_id not in observations:
                continue
            
            obs_flat = self._flatten_obs(observations[agent_id])
            obs_tensor = torch.FloatTensor(obs_flat)
            
            # Forward through actor
            probs = self.actor_nets[agent_id](obs_tensor)
            policy = Categorical(probs=probs)
            action = policy.sample()
            
            actions[agent_id] = action.item()
            
            if train_returns:
                logprobs[agent_id] = policy.log_prob(action)
                values[agent_id] = global_value  # All agents share same critic value
        
        if train_returns:
            return actions, logprobs, values
        return actions
    
    def learn(self, total_timesteps: int):
        """
        Train the multi-agent system using PettingZoo Parallel API pattern.
        
        Uses: while env.agents: ... pattern
        
        Parameters
        ----------
        total_timesteps : int
            Total environment steps for training
        """
        observations, _ = self.env.reset()
        
        # Tracking
        overall_rewards = []
        episode_rewards = defaultdict(float)
        episode_counter = 0
        timestep_counter = 0
        global_timestep = 0
        
        while global_timestep < total_timesteps:
            
            # === EPISODE LOOP ===
            while self.env.agents:
                
                active_agents = self.env.agents
                
                # Predict actions for all active agents
                actions, logprobs, values = self.predict(
                    observations, active_agents, train_returns=True
                )
                
                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Store transitions
                self.memory.put(observations, actions, rewards, logprobs, values)
                
                # Track episode rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                timestep_counter += 1
                global_timestep += 1
                
                # Update observations
                observations = next_observations
                
                # Update when batch is full
                if timestep_counter >= self.batch_size:
                    
                    # Bootstrap values if episode not finished
                    if self.env.agents:
                        global_state = self._get_global_state(observations, self.env.agents)
                        bootstrap_value = self.critic_net(torch.FloatTensor(global_state)).item()
                        last_values = {agent: bootstrap_value for agent in self.possible_agents}
                    else:
                        last_values = {agent: 0.0 for agent in self.possible_agents}
                    
                    self.memory.finish_trajectory(last_values)
                    
                    # Perform update
                    if self.memory.has_data():
                        self._update()
                    
                    # Clear memory
                    self.memory.clear()
                    timestep_counter = 0
                
                # Check if we've reached total timesteps
                if global_timestep >= total_timesteps:
                    break
            
            # === EPISODE ENDED ===
            episode_counter += 1
            total_ep_reward = sum(episode_rewards.values())
            overall_rewards.append(total_ep_reward)
            
            # Finish trajectory with terminal values
            last_values = {agent: 0.0 for agent in self.possible_agents}
            self.memory.finish_trajectory(last_values)
            
            # Log progress
            if episode_counter % 10 == 0:
                avg_reward = np.mean(overall_rewards[-10:]) if len(overall_rewards) >= 10 else np.mean(overall_rewards)
                print(f"Episode {episode_counter} | Timestep {global_timestep} | Avg Reward: {avg_reward:.2f}")
            
            # Reset for next episode
            if global_timestep < total_timesteps:
                observations, _ = self.env.reset()
                episode_rewards = defaultdict(float)
        
        print(f"\nTraining complete! Episodes: {episode_counter}, Timesteps: {global_timestep}")
        return overall_rewards
    
    def _update(self):
        """Perform one update step for all networks."""
        
        # Update each agent's actor
        for agent_id in self.possible_agents:
            _, _, _, logprob_lst, return_lst, value_lst, adv_lst = self.memory.get(agent_id)
            
            if len(logprob_lst) == 0:
                continue
            
            # Actor loss
            actor_loss = self.calc_actor_loss(logprob_lst, adv_lst)
            
            # Backprop for this actor
            self.optim_actors[agent_id].zero_grad()
            actor_loss.backward()
            self.optim_actors[agent_id].step()
        
        # Critic loss (aggregate all agents' returns and values)
        all_values = []
        all_returns = []
        for agent_id in self.possible_agents:
            _, _, _, _, return_lst, value_lst, _ = self.memory.get(agent_id)
            all_values.extend(value_lst)
            all_returns.extend(return_lst)
        
        if len(all_values) > 0:
            critic_loss = self.calc_critic_loss(all_values, all_returns)
            
            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()
    
    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss."""
        return F.mse_loss(torch.FloatTensor(value_lst), torch.FloatTensor(return_lst))
    
    @staticmethod
    def calc_actor_loss(logprob_lst, adv_lst):
        """Calculate actor loss."""
        # Normalize advantages (helps stability)
        adv_tensor = torch.FloatTensor(adv_lst)
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        return -(adv_tensor * torch.stack(logprob_lst)).mean()