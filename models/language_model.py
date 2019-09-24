import torch
from torch import nn


class DeepFusion(nn.Module):

    def __init__(self, char_freq, vocab_size, n_layers):
        super(DeepFusion, self).__init__()
        self.char_freq = char_freq
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.gate = nn.Linear(vocab_size, vocab_size)
        self.dnn = nn.Sequential(
            *[nn.Linear(2*vocab_size, 2*vocab_size if i < n_layers - 1 else vocab_size) for i in range(n_layers)]
        )

    def forward(self, x):
        batch_size, vocab_size = x.shape
        gated_lm = self.gate(self.char_freq).view(1, -1).repeat(batch_size, 1)
        output = self.dnn(torch.cat((x, gated_lm), dim=1))
        return output
