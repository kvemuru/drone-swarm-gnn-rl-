import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from models.gnn_layer import GNNLayer

class GNNMAPPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.comm_radius = custom_config.get("comm_radius", 30.0)
        n_agents = custom_config.get("n_agents", 4)
        global_state_dim = n_agents * 4

        self.gnn1 = GNNLayer(4, 64)
        self.gnn2 = GNNLayer(64, 128)

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _compute_edge_features(self_feat, neigh_feats, comm_radius):
        self_pos = self_feat[:, :2].unsqueeze(1)
        neigh_pos = neigh_feats[:, :, :2]
        rel_pos = neigh_pos - self_pos
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        dist_ratio = dist / comm_radius
        return torch.cat([rel_pos, dist, dist_ratio], dim=-1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        self._global_state = obs.get("global_state")

        edge_feats = self._compute_edge_features(obs["self"], obs["neighbors"], self.comm_radius)

        x, neigh = self.gnn1(obs["self"], obs["neighbors"], obs["mask"], edge_feats)
        x, _ = self.gnn2(x, neigh, obs["mask"], edge_feats)

        return self.actor(x), state

    def value_function(self):
        return self.critic(self._global_state).squeeze(-1)
