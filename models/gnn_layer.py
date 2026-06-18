import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2, 1)

    def forward(self, self_feat, neigh_feats, mask):
        self_emb = self.fc_self(self_feat)
        neigh_emb = self.fc_neigh(neigh_feats)

        self_exp = self_emb.unsqueeze(1).repeat(1, neigh_feats.shape[1], 1)
        scores = self.attn(torch.cat([self_exp, neigh_emb], dim=-1)).squeeze(-1)

        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)

        agg = torch.sum(weights.unsqueeze(-1) * neigh_emb, dim=1)
        return F.relu(self_emb + agg)