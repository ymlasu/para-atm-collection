
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_nodes, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2.0))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a2.data, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.downsample = nn.Conv1d(in_features, out_features, 1)

        self.bias = nn.Parameter(torch.zeros(num_nodes, out_features))

    def forward(self, input, adj):
        batch_size = input.size(0)
        h = torch.bmm(input, self.W.expand(batch_size, self.in_features, self.out_features))
        f_1 = torch.bmm(h, self.a1.expand(batch_size, self.out_features, 1))
        f_2 = torch.bmm(h, self.a2.expand(batch_size, self.out_features, 1))
        e = self.leakyrelu(f_1 + f_2.transpose(2,1))

        attention = torch.mul(adj, e)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h) + self.bias.expand(batch_size, self.num_nodes, self.out_features)

        if input.shape[-1] != h_prime.shape[-1]:
            input_transformed = self.downsample(input.permute(0, 2, 1)).permute(0, 2, 1)
            h_prime = h_prime + input_transformed
        else:
            h_prime = h_prime + input


        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_nodes, num_heads, dropout=0.6, alpha=0.2):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList([GraphAttentionLayer(in_features, hidden_dim, num_nodes, dropout, alpha) for _ in range(num_heads)])
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, out_features, num_nodes, dropout, alpha)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attention_heads], dim=2)
        x = F.elu(self.out_att(x, adj))
        return x

class ParallelGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_nodes, num_heads):
        super(ParallelGraphAttentionLayer, self).__init__()
        self.out_features = out_features
        self.gat_layer1 = MultiHeadGraphAttention(in_features, hidden_dim, out_features, num_nodes, num_heads)
        self.gat_layer2 = MultiHeadGraphAttention(in_features, hidden_dim, out_features, num_nodes, num_heads)

        self.fs = nn.Linear(out_features, out_features)
        self.fg = nn.Linear(out_features, out_features)

    def forward(self, input, adj_1, adj_2):
        B, N, T, C = input.shape
        C_ = self.out_features
        X_d = torch.zeros(B, N, T, C_, device=input.device)
        X_f = torch.zeros(B, N, T, C_, device=input.device)

        for t in range(T):
            o_1 = self.gat_layer1(input[:, :, t, :], adj_1)
            o_2 = self.gat_layer2(input[:, :, t, :], adj_2)
            X_d[:, :, t, :] = o_1
            X_f[:, :, t, :] = o_2

        g = torch.sigmoid(self.fs(X_d) + self.fg(X_f))
        out = g * X_d + (1 - g) * X_f

        return out
