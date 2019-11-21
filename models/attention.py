import torch
from torch import nn


class MultiLayerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, num_layers=5, dropout=0.5):
        super(MultiLayerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.n_layers = num_layers
        self.layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads, dropout) for i in range(num_layers)])

    def clone(self, tensor, n):
        return [tensor.clone() for i in range(n)]

    def forward(self, source, target=None):
        """
        :param source: [source len, batch size, embed dim]
        :param target: [target len, batch_size, embed dim]
        :return attn:
        """
        q, k, v = self.clone(source, 3)
        for layer in self.layers[:-1]:
            result, weights = layer(q, k, v)
            q, k, v = self.clone(result, 3)
        if target is not None:
            q = target.clone()
        attn, weights = self.layers[-1](q, k, v)
        return attn
