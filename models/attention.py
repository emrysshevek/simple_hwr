import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, embed_dim, n_layers=1, dropout=0.01):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout)

    def forward(self, encoder_output, decoder_hidden):
        attn_output, attn_weights = self.attn(decoder_hidden, encoder_output, encoder_output)
        attn_output = attn_output.squeeze()
        return attn_output
