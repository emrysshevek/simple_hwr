import torch
from torch import nn


class RNNDecoder(nn.Module):

    def __init__(self, input_dim, context_dim, hidden_dim, output_dim, n_layers, dropout=0.01):
        super(RNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size=input_dim+context_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout)
        self.linear_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, context, hidden=None):
        rnn_input = torch.cat([input, context], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.linear_proj(output)
        return output, hidden
