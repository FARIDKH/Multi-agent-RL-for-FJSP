from collections import defaultdict
from typing import Any, Dict, List, Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

from networks import ActorNetwork, CentralizedCriticNetwork, CriticNetwork
from transition_memory import MultiAgentTransitionMemory
from visualization import GridVisualizer


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
                 use_gae: bool = True,
                 visualize: bool = False,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
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
        visualize : bool
            Enable grid visualization
        entropy_coef : float
            Entropy bonus coefficient for exploration
        max_grad_norm : float
            Maximum gradient norm for clipping
        """
        self.env = env
        self.visualize = visualize
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamb = lamb
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
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
        
        # Track episode end timesteps for plotting
        self.episode_end_timesteps: List[int] = []

        # Loss tracking
        self.actor_loss_history: Dict[str, List[float]] = {agent: [] for agent in self.possible_agents}
        self.critic_loss_history: List[float] = []

        # Visualization
        self.grid_viz: Optional[GridVisualizer] = None
    
    def _get_obs_dim(self, obs_space) -> int:
        """Calculate flattened observation dimension."""
        if hasattr(obs_space, 'spaces'):  # Dict space
            total = 0
            for key, space in obs_space.spaces.items():
                if key == 'action_mask': # <--- SKIP MASK
                    continue
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
            if key == "action_mask":  # <--- SKIP MASK
                continue
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
                train_returns: bool = False,
                deterministic: bool = False):
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
        deterministic : bool
            If True, use greedy action selection (argmax) instead of sampling

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
        global_value_tensor = self.critic_net(global_state_tensor)

        # Get action from each active agent's actor
        for agent_id in active_agents:
            if agent_id not in observations:
                continue

            # 1. Prepare Input (mask is excluded by _flatten_obs)
            obs_flat = self._flatten_obs(observations[agent_id])
            obs_tensor = torch.FloatTensor(obs_flat)

            # 2. Forward pass to get raw probabilities
            probs = self.actor_nets[agent_id](obs_tensor)

            # 3. Apply Action Masking
            if 'action_mask' in observations[agent_id]:
                mask = observations[agent_id]['action_mask']
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
                
                # Zero out probabilities for invalid actions
                probs = probs * mask_tensor
                
                # Renormalize probabilities
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                else:
                    # Fallback: if network predicts 0 prob for all valid actions,
                    # distribute uniformly among valid actions
                    probs = mask_tensor / mask_tensor.sum()

            # 4. Action Selection
            if deterministic:
                # Greedy: pick action with highest probability among valid ones
                action = torch.argmax(probs)
            else:
                # Stochastic: sample from masked distribution
                policy = Categorical(probs=probs)
                action = policy.sample()

            actions[agent_id] = action.item()

            if train_returns:
                # Re-create policy with masked probs for correct log_prob calculation
                policy = Categorical(probs=probs)
                logprobs[agent_id] = policy.log_prob(action)
                values[agent_id] = global_value_tensor

        if train_returns:
            return actions, logprobs, values
        return actions
    
    def learn(self, total_timesteps: int, num_orders: int = 25):
        """
        Train the multi-agent system using PettingZoo Parallel API pattern.

        Uses: while env.agents: ... pattern

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for training
        num_orders : int
            Number of orders to generate per episode (default: 25 for training)
        """
        observations, _ = self.env.reset(options={'num_orders': num_orders})

        # Initialize visualization if enabled
        if self.visualize:
            self.grid_viz = GridVisualizer(enabled=True)

        # Tracking
        overall_rewards = []
        episode_rewards = defaultdict(float)
        episode_counter = 0
        timestep_counter = 0
        global_timestep = 0
        self.episode_end_timesteps = []
        
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

                # Update visualization
                if self.grid_viz is not None:
                    sim = self.env.unwrapped.simulation
                    self.grid_viz.update(
                        agv_position=sim.agv.position,
                        actions=actions,
                        step=global_timestep,
                        episode=episode_counter,
                        agv_carrying=sim.agv.carrying_tray is not None,
                        simulation=sim
                    )

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
            self.episode_end_timesteps.append(global_timestep)

            # Get order progress before reset
            sim = self.env.unwrapped.simulation
            progress = sim.get_order_progress()

            # Finish trajectory with terminal values
            last_values = {agent: 0.0 for agent in self.possible_agents}
            self.memory.finish_trajectory(last_values)

            # Log episode summary (to log file via logger)
            from utils.Logger import get_logger
            logger = get_logger()
            logger.episode_summary(
                episode=episode_counter,
                steps=sim.current_step,
                orders_completed=progress['completed_orders'],
                total_orders=progress['total_orders'],
                products_packaged=progress['products_packaged'],
                total_products=progress['total_products'],
                reward=total_ep_reward
            )
            # Also print to console
            print(f"Episode {episode_counter:3d} | Steps: {sim.current_step:4d} | "
                  f"Orders: {progress['completed_orders']}/{progress['total_orders']} | "
                  f"Products: {progress['products_packaged']}/{progress['total_products']} | "
                  f"Reward: {total_ep_reward:.1f}")

            # Reset for next episode
            if global_timestep < total_timesteps:
                observations, _ = self.env.reset(options={'num_orders': num_orders})
                episode_rewards = defaultdict(float)

        # Keep visualization open after training
        if self.grid_viz is not None:
            self.grid_viz.keep_open()

        print(f"\nTraining complete! Episodes: {episode_counter}, Timesteps: {global_timestep}")
        return overall_rewards

    def _get_heuristic_actions(self, sim) -> Dict[str, int]:
        """
        Rule-based policy for testing when RL policy hasn't converged.

        AGV Logic:
        1. If carrying a tray that needs processing:
           - If machine is available, go there and drop
           - If machine is busy, go to storage
        2. If carrying a tray that needs packaging, go to packaging
        3. If not carrying:
           - Priority 1: Pick up ready trays at machines (processed, need packaging)
           - Priority 2: Pick up trays from storage that need processing (if machine now available)
           - Priority 3: Pick up ready trays at pickup
        """
        actions = {}

        # Pickup Station: Always try to load products if there are orders
        if sim.pickup_station.order_queue or sim.pickup_station.current_order:
            actions['pickup_station'] = 1  # LOAD_NEXT_PRODUCT
        else:
            actions['pickup_station'] = 0  # IDLE

        # AGV: Follow a state machine with storage buffer
        agv = sim.agv

        if agv.is_moving:
            actions['agv'] = 0  # Already moving, wait
        elif agv.carrying_tray:
            tray = agv.carrying_tray
            if tray.needs_processing:
                tray_type = tray.tray_type
                # Determine target machine
                if tray_type in ['SMALL', 'MEDIUM']:
                    target_machine = sim.small_machine
                    target_pos = (2, 3)
                    move_action = 2  # MOVE_TO_SMALL_MACHINE
                else:  # BIG
                    target_machine = sim.big_machine
                    target_pos = (0, 3)
                    move_action = 3  # MOVE_TO_BIG_MACHINE

                # Check if machine is available (not busy and has queue space)
                machine_available = not target_machine.is_busy and len(target_machine.tray_queue) < 3

                if machine_available:
                    if agv.position == target_pos:
                        actions['agv'] = 7  # DROP at machine
                    else:
                        actions['agv'] = move_action
                else:
                    # Machine busy, go to storage
                    if agv.position == (1, 5):  # At storage
                        actions['agv'] = 7  # DROP at storage
                    else:
                        actions['agv'] = 4  # MOVE_TO_STORAGE

            elif tray.needs_packaging:
                # Processed, need packaging
                if agv.position == (3, 5):  # At packaging
                    actions['agv'] = 7  # DROP
                else:
                    actions['agv'] = 5  # MOVE_TO_PACKAGING
            else:
                actions['agv'] = 0  # IDLE (shouldn't happen)
        else:
            # Not carrying - look for trays to pick up
            # Priority 1: Ready trays at machines (processed, ready for packaging)
            if sim.small_machine.ready_trays:
                if agv.position == (2, 3):
                    actions['agv'] = 6  # PICKUP
                else:
                    actions['agv'] = 2  # MOVE_TO_SMALL_MACHINE
            elif sim.big_machine.ready_trays:
                if agv.position == (0, 3):
                    actions['agv'] = 6  # PICKUP
                else:
                    actions['agv'] = 3  # MOVE_TO_BIG_MACHINE
            # Priority 2: Trays in storage that need processing (if machine now available)
            elif sim.storage.trays:
                # Check if any tray in storage can go to an available machine
                for storage_tray in sim.storage.trays:
                    if storage_tray.needs_processing:
                        tray_type = storage_tray.tray_type
                        if tray_type in ['SMALL', 'MEDIUM']:
                            if not sim.small_machine.is_busy and len(sim.small_machine.tray_queue) < 3:
                                if agv.position == (1, 5):
                                    actions['agv'] = 6  # PICKUP from storage
                                else:
                                    actions['agv'] = 4  # MOVE_TO_STORAGE
                                break
                        else:  # BIG
                            if not sim.big_machine.is_busy and len(sim.big_machine.tray_queue) < 3:
                                if agv.position == (1, 5):
                                    actions['agv'] = 6  # PICKUP from storage
                                else:
                                    actions['agv'] = 4  # MOVE_TO_STORAGE
                                break
                else:
                    # No storage trays ready for machines, check pickup
                    if sim.pickup_station.ready_trays:
                        if agv.position == (0, 0):
                            actions['agv'] = 6  # PICKUP
                        else:
                            actions['agv'] = 1  # MOVE_TO_PICKUP
                    else:
                        actions['agv'] = 0  # Wait
            # Priority 3: Ready trays at pickup
            elif sim.pickup_station.ready_trays:
                if agv.position == (0, 0):
                    actions['agv'] = 6  # PICKUP
                else:
                    actions['agv'] = 1  # MOVE_TO_PICKUP
            # Nothing ready, wait at appropriate location
            elif sim.small_machine.is_busy or sim.small_machine.current_tray:
                if agv.position == (2, 3):
                    actions['agv'] = 0  # Wait at small machine
                else:
                    actions['agv'] = 2  # Go to small machine
            elif sim.big_machine.is_busy or sim.big_machine.current_tray:
                if agv.position == (0, 3):
                    actions['agv'] = 0  # Wait at big machine
                else:
                    actions['agv'] = 3  # Go to big machine
            elif sim.pickup_station.order_queue or sim.pickup_station.current_order:
                if agv.position == (0, 0):
                    actions['agv'] = 0  # Wait at pickup
                else:
                    actions['agv'] = 1  # Go to pickup
            else:
                actions['agv'] = 0  # Nothing to do

        # Machines: Start processing if queue, signal complete if done
        for machine_name, machine in [('small_machine', sim.small_machine), ('big_machine', sim.big_machine)]:
            if machine.tray_queue and not machine.is_busy:
                actions[machine_name] = 1  # START_PROCESSING
            elif machine.current_tray and not machine.is_busy:
                actions[machine_name] = 2  # SIGNAL_COMPLETE
            else:
                actions[machine_name] = 0  # IDLE

        # Packaging: Start if queue
        for name, station in sim.packaging_stations.items():
            if station.product_queue and not station.is_busy:
                actions[name] = 1  # START_PACKAGING
            else:
                actions[name] = 0  # IDLE

        return actions

    def test(self, num_orders: int = 5, visualize: bool = True, max_steps: int = 500, use_heuristic: bool = False):
        """
        Test the trained model on a set of orders.

        Parameters
        ----------
        num_orders : int
            Number of orders to test on (default: 5)
        visualize : bool
            Whether to enable visualization during test
        max_steps : int
            Maximum steps before truncation
        use_heuristic : bool
            If True, use rule-based heuristic instead of learned policy

        Returns
        -------
        dict with test results
        """
        print(f"\n{'='*60}")
        print(f"TEST PHASE: Running with {num_orders} orders")
        print(f"{'='*60}")
        if use_heuristic:
            print("Using HEURISTIC policy (rule-based)")

        # Close any existing visualization from training
        if self.grid_viz is not None:
            import matplotlib.pyplot as plt
            plt.close('all')
            self.grid_viz = None

        # Initialize visualization if requested
        if visualize:
            self.grid_viz = GridVisualizer(enabled=True)

        observations, _ = self.env.reset(options={'num_orders': num_orders})
        sim = self.env.unwrapped.simulation

        episode_rewards = defaultdict(float)
        step_count = 0
        orders_completed = 0
        products_packaged = 0

        while self.env.agents and step_count < max_steps:
            active_agents = self.env.agents

            # Get actions
            if use_heuristic:
                actions = self._get_heuristic_actions(sim)
            else:
                actions = self.predict(observations, active_agents, train_returns=False, deterministic=True)
                
               

            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

            # Update visualization
            if self.grid_viz is not None:
                sim = self.env.unwrapped.simulation
                self.grid_viz.update(
                    agv_position=sim.agv.position,
                    actions=actions,
                    step=step_count,
                    episode=0,
                    agv_carrying=sim.agv.carrying_tray is not None,
                    simulation=sim
                )

            # Track rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            # Update tracking from infos
            for agent_id, info in infos.items():
                orders_completed = info.get('orders_completed', orders_completed)
                products_packaged = info.get('total_products_packaged', products_packaged)

            step_count += 1
            observations = next_observations

            # Progress logging
            if step_count % 50 == 0:
                print(f"  Test Step {step_count} | Orders: {orders_completed}/{num_orders} | Products: {products_packaged}")

        # Test complete
        total_reward = sum(episode_rewards.values())
        print(f"\n{'='*60}")
        print(f"TEST COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps: {step_count}")
        print(f"  Orders completed: {orders_completed}/{num_orders}")
        print(f"  Products packaged: {products_packaged}")
        print(f"  Total reward: {total_reward:.2f}")

        # Keep visualization open
        if self.grid_viz is not None:
            self.grid_viz.keep_open()

        return {
            'steps': step_count,
            'orders_completed': orders_completed,
            'total_orders': num_orders,
            'products_packaged': products_packaged,
            'total_reward': total_reward,
            'rewards_by_agent': dict(episode_rewards)
        }

    def _update(self):
        """Perform one update step for all networks."""

        # Update each agent's actor
        for agent_id in self.possible_agents:
            obs_lst, _, _, logprob_lst, return_lst, value_lst, adv_lst = self.memory.get(agent_id)

            if len(logprob_lst) == 0:
                continue

            # Calculate entropy bonus for exploration
            entropy_bonus = self._calculate_entropy(agent_id, obs_lst)

            # Actor loss with entropy bonus
            actor_loss = self.calc_actor_loss(logprob_lst, adv_lst)
            actor_loss = actor_loss - self.entropy_coef * entropy_bonus
            self.actor_loss_history[agent_id].append(actor_loss.item())

            # Backprop for this actor with gradient clipping
            self.optim_actors[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_nets[agent_id].parameters(),
                self.max_grad_norm
            )
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
            self.critic_loss_history.append(critic_loss.item())

            self.optim_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic_net.parameters(),
                self.max_grad_norm
            )
            self.optim_critic.step()

    def _calculate_entropy(self, agent_id: str, obs_lst: List) -> torch.Tensor:
        """Calculate entropy of policy for exploration bonus."""
        if len(obs_lst) == 0:
            return torch.tensor(0.0)

        entropies = []
        for obs in obs_lst:
            # Flatten dict observation to tensor
            obs_flat = self._flatten_obs(obs)
            obs_tensor = torch.FloatTensor(obs_flat)
            probs = self.actor_nets[agent_id](obs_tensor)
            # Entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum()
            entropies.append(entropy)

        return torch.stack(entropies).mean()
    
    @staticmethod
    def calc_critic_loss(value_lst, return_lst):
        """Calculate critic loss."""
        # Ensure values keep gradients while targets are plain tensors
        values_tensor = torch.stack([
            v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
            for v in value_lst
        ]).view(-1)
        returns_tensor = torch.tensor(return_lst, dtype=torch.float32, device=values_tensor.device)
        return F.mse_loss(values_tensor, returns_tensor)
    
    @staticmethod
    def calc_actor_loss(logprob_lst, adv_lst):
        """Calculate actor loss."""
        # Normalize advantages (helps stability)
        adv_tensor = torch.FloatTensor(adv_lst)
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return -(adv_tensor * torch.stack(logprob_lst)).mean()

    def plot_losses(self, save_path: str = None, show: bool = True):
        """
        Plot actor and critic loss curves.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        show : bool
            Whether to display the plot.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot critic loss
        axes[0].plot(self.critic_loss_history, label='Critic Loss', color='blue')
        axes[0].set_xlabel('Update Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Critic Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot actor losses for each agent
        for agent_id, losses in self.actor_loss_history.items():
            if len(losses) > 0:
                axes[1].plot(losses, label=f'{agent_id}')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Actor Losses by Agent')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")

        if show:
            plt.show()

        return fig
