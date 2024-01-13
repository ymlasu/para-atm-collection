
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Layers import Layers

def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    gumbel = gumbel.cuda()
    return gumbel

def gumbel_softmax_sample(logits, temperature):
    y = logits + gumbel_sample(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.max(y, dim=1)[1]
        y = F.one_hot(y_hard, num_classes=logits.size(-1)).float()
    return y

class NetworkGenerator(nn.Module):
    def __init__(self, num_nodes):
        super(NetworkGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.matrix = nn.Parameter(torch.randn(num_nodes, num_nodes, 2))
        self.tau = 10
        self.weight = nn.Parameter(torch.ones(1, num_nodes, num_nodes))

    def forward(self, adj, batch_size, a):
        adj_mx = self.matrix.view(-1, 2)
        adj_mx = F.gumbel_softmax(adj_mx, tau=self.tau, hard=True)[:, 0]
        adjs = adj_mx.view(1, self.num_nodes, self.num_nodes)
        adjs = self.weight * adjs
        return adjs

    def drop_temp(self, a=0.9999):
        self.tau = self.tau * a

class MultiNetworkGenerator(nn.Module):
    """
    Graph Generator, similar to https://dl.acm.org/doi/abs/10.1007/s10489-021-02648-0
    """
    def __init__(self, num_nodes, nheads=6):
        super(MultiNetworkGenerator, self).__init__()
        self.nheads = nheads
        self.num_nodes = num_nodes
        self.generators = nn.ModuleList([NetworkGenerator(num_nodes) for _ in range(nheads)])

    def forward(self, adj, batch_size, a):
        adjs = sum([generator(adj, batch_size, a) for generator in self.generators]) / self.nheads
        adjs = a * adjs.repeat(batch_size, 1, 1) + (1-a) * adj
        return adjs

    def drop_temp(self, a=0.9999):
        for generator in self.generators:
            generator.drop_temp(a)

class PIGAT(nn.Module):
    def __init__(self, in_features, embed_size, hidden_dim, num_nodes, spatial_heads, temporal_heads, graph_heads, num_time, num_time_out, num_layers):
        super(PIGAT, self).__init__()

        self.network_generator_1 = MultiNetworkGenerator(num_nodes, graph_heads)
        self.network_generator_2 = MultiNetworkGenerator(num_nodes, graph_heads)
        self.conv1 = nn.Conv2d(in_features, embed_size, 1)
        self.Layers = Layers(embed_size, hidden_dim, num_nodes, spatial_heads, temporal_heads, num_time, num_layers)
        self.conv2 = nn.Conv2d(num_time, num_time_out, 1)
        self.conv3 = nn.Conv2d(embed_size, 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x, adj_1, adj_2, a):

        adj_1 = self.network_generator_1(adj_1, x.shape[0], a)
        adj_2 = self.network_generator_2(adj_2, x.shape[0], a)
        if self.training:
            self.network_generator_1.drop_temp()
            self.network_generator_2.drop_temp()
        norm_loss = self.calLoss(x, adj_1) + self.calLoss(x, adj_2)

        input = self.conv1(x).permute(0, 3, 2, 1)
        output = self.Layers(input, adj_1, adj_2).permute(0, 2, 1, 3)
        out = self.relu(self.conv2(output)).permute(0, 3, 2, 1)
        out = self.conv3(out).permute(0, 1, 3, 2)
        return out, norm_loss

    def calLoss(self, x, A, a=1, b=1):
        n = A.shape[1]
        device = x.device
        eps = 1e-15

        A1 = A @ torch.ones(x.shape[0], n, 1, device=device)
        zero_vec = eps * torch.ones_like(A1)
        A1 = torch.where(A1 > 0, A1, zero_vec)
        FA1 = torch.sum(torch.ones((x.shape[0], 1, n), device=device) @ torch.log(A1 + eps))

        FA2 = torch.sum(A * A)
        FA = (-b) * FA1 / n + a * FA2 / (n * n)
        FA = FA / A.shape[0]

        return FA
