import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PGAT import ParallelGraphAttentionLayer
from TTransformer import TTransformer

class STBlock(nn.Module):
    def __init__(self, embed_size, hidden_dim, num_nodes, num_heads, heads, time_num, dropout=0.6):
        super(STBlock, self).__init__()
        self.PGAT = ParallelGraphAttentionLayer(embed_size, hidden_dim, embed_size, num_nodes, num_heads)
        self.TTransformer = TTransformer(embed_size, heads, time_num)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_1, adj_2):
        x1 = self.norm1(self.PGAT(x, adj_1, adj_2))
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        return x2

class Layers(nn.Module):
    def __init__(self, embed_size, hidden_dim, num_nodes, spatial_heads, temporal_heads, time_num, num_layers, dropout=0.6):
        super(Layers, self).__init__()
        self.layers = nn.ModuleList([
            STBlock(embed_size, hidden_dim, num_nodes, spatial_heads, temporal_heads, time_num, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_1, adj_2):
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, adj_1, adj_2)
        return out
