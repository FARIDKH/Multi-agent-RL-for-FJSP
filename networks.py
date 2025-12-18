from torch import nn


class CriticNetwork(nn.Module):
    """Neural Network used to learn the state-value function."""

    def __init__(self, num_observations):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)

class CentralizedCriticNetwork(nn.Module):
    """
    Centralized Critic for CTDE.
    Input: Global state (concatenation of all agents' observations)
    Output: Single value estimate
    """
    
    def __init__(self, global_state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.net(x)