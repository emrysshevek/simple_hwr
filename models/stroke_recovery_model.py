import torch
from torch import nn
from models.CoordConv import CoordConv
from models.encoder import CRNNEncoder
from models.attention import MultiLayerSelfAttention


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
        super(StrokeRecoveryModel, self).__init__()
        self.encoder = CRNNEncoder(vocab_size, device, first_conv_op, first_conv_opts)
        self.attn = MultiLayerSelfAttention(vocab_size)

    def get_cnn(self):
        return self.encoder.cnn

    def forward(self, input):
        rnn_output = self.encoder(input)
        attn_output = self.attn(rnn_output)
        return attn_output