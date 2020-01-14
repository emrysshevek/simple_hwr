import torch
from torch import nn
from models.basic import CNN, BidirectionalRNN
from models.CoordConv import CoordConv
from models.encoder import CRNNEncoder
from models.attention import MultiLayerSelfAttention
from models.decoder import RNNDecoder


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None):
        super().__init__()
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.linear_in = nn.Linear(1024, 128)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=1)
        # self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
        self.rnn = nn.LSTM(input_size=2*128, hidden_size=128, num_layers=1)
        self.linear_out = nn.Linear(128, vocab_size)
        self.sigmoid =torch.nn.Sigmoid().to(device)
        self.device = device

    def get_cnn(self):
        return self.cnn

    def forward(self, input):
        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input)
        cnn_output = self.linear_in(cnn_output)
        seq_len, batch_size, input_size = cnn_output.shape

        state = torch.zeros(1, batch_size, input_size).to(self.device)
        hidden = None
        output = []
        for i in range(seq_len):
            attn_output, attn_weights = self.attn(state, cnn_output, cnn_output)
            rnn_input = torch.cat([state, attn_output], dim=-1)
            state, hidden = self.rnn(rnn_input, hidden)  # width, batch, alphabet
            linear_output = self.linear_out(state)
            output.append(linear_output)

        output = torch.cat(output, dim=0)
        output[:,:,2:] = self.sigmoid(output[:,:,2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
        return output


# class StrokeRecoveryModel(nn.Module):
#     def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None):
#         super().__init__()
#         if first_conv_op:
#             first_conv_op = CoordConv
#         self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
#         self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
#         self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
#         self.sigmoid =torch.nn.Sigmoid().to(device)
#
#     def get_cnn(self):
#         return self.cnn
#
#     def forward(self, input):
#         if self.training:
#             return self._forward(input)
#         else:
#             with torch.no_grad():
#                 return self._forward(input)
#
#     def _forward(self, input):
#         cnn_output = self.cnn(input)
#         attn_output, attn_weights = self.attn(cnn_output, cnn_output, cnn_output)
#         rnn_output = self.rnn(attn_output)  # width, batch, alphabet
#         rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
#
#         return rnn_output


# class StrokeRecoveryModel(nn.Module):
#     """ORIGINAL MODEL: DO NOT TOUCH"""
#     def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None):
#         super().__init__()
#         if first_conv_op:
#             first_conv_op = CoordConv
#         self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
#         self.rnn = BidirectionalRNN(nIn=1024, nHidden=128, nOut=vocab_size, dropout=.5, num_layers=2, rnn_constructor=nn.LSTM)
#         self.sigmoid =torch.nn.Sigmoid().to(device)
#
#     def get_cnn(self):
#         return self.cnn
#
#     def forward(self, input):
#         if self.training:
#             return self._forward(input)
#         else:
#             with torch.no_grad():
#                 return self._forward(input)
#
#     def _forward(self, input):
#         cnn_output = self.cnn(input)
#         rnn_output = self.rnn(cnn_output)  # width, batch, alphabet
#         rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
#
#         return rnn_output
