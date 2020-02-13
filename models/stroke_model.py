from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv

class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.3, num_layers=2, rnn_constructor=nn.LSTM)

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        rnn_output = self.rnn(cnn_output) # width, batch, alphabet
        # sigmoids are done in the loss
        return rnn_output
