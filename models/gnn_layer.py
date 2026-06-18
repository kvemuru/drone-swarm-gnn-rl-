import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2 + edge_dim, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, self_feat, neigh_feats, mask, edge_feats=None):
        self_emb = self.fc_self(self_feat)
        neigh_emb = self.fc_neigh(neigh_feats)

        self_exp = self_emb.unsqueeze(1).repeat(1, neigh_feats.shape[1], 1)

        attn_input = [self_exp, neigh_emb]
        if edge_feats is not None:
            attn_input.append(edge_feats)

        scores = self.attn(torch.cat(attn_input, dim=-1)).squeeze(-1)

        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)

        agg = torch.sum(weights.unsqueeze(-1) * neigh_emb, dim=1)
        out = self_emb + agg
        out = self.norm(out)
        return F.relu(out), neigh_emb
