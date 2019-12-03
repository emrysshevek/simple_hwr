import torch
from torch import nn

from models.CoordConv import CoordConv
from models.basic import CNN, BidirectionalRNN


class CRNNEncoder(nn.Module):

    def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
        super().__init__()
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type="default64", first_conv_opts=first_conv_opts)
        self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2,
                                    rnn_constructor=nn.LSTM)
        self.sigmoid = torch.nn.Sigmoid().to(device)

    def forward(self, input):
        cnn_output = self.cnn(input)

        rnn_output = self.rnn(cnn_output)  # width, batch, alphabet
        rnn_output[:, :, 2:] = self.sigmoid(rnn_output[:, :, 2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic

        return rnn_output