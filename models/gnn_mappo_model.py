import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from models.gnn_layer import GNNLayer

class GNNMAPPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.gnn1 = GNNLayer(4, 64)
        self.gnn2 = GNNLayer(64, 128)

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        x, neigh = self.gnn1(obs["self"], obs["neighbors"], obs["mask"])
        x, _ = self.gnn2(x, neigh, obs["mask"])

        self._last_x = x
        return self.actor(x), state

    def value_function(self):
        return self.critic(self._last_x).squeeze(-1)