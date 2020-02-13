from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv

MAX_LENGTH = 64

class StartPointModel(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=1024, bidirectional=True, dropout=.3, num_layers=1)
        # self.linear1 = nn.Linear
        self.decoder = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, dropout=.3)
        self.linear = nn.Linear(1024, vocab_size)
        self.device = device

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        _, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, 1024)).to(self.device)
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(output, hidden)
            #output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs

class StartPointModel2(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.decoder_size = 256
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=self.decoder_size, bidirectional=True, dropout=.3, num_layers=1)
        self.decoder = nn.LSTM(input_size=self.decoder_size, hidden_size=self.decoder_size, num_layers=2, dropout=.3)
        self.linear = nn.Linear(self.decoder_size, vocab_size)
        self.device = device

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        _, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, self.decoder_size)).to(self.device)
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(output, hidden)
            #output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs

class StartPointAttnModel(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=1)
        self.decoder = nn.LSTM(input_size=512, hidden_size=256, num_layers=1)
        self.linear = nn.Linear(256, vocab_size)
        self.device = device

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, 256)).to(self.device)
        for i in range(MAX_LENGTH):
            context, _ = self.attn(output, encoding, encoding)
            output, hidden = self.decoder(torch.cat([output, context], dim=-1), hidden)
            output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs
