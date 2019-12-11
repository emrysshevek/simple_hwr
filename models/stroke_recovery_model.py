import torch
from torch import nn
from models.basic import CNN, BidirectionalRNN
from models.CoordConv import CoordConv
from models.encoder import CRNNEncoder
from models.attention import MultiLayerSelfAttention
from models.decoder import RNNDecoder


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
        super().__init__()
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type="default64", first_conv_opts=first_conv_opts)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4)
        # self.rnn = nn.LSTM(2*1024, hidden_size=1024, num_layers=2)
        self.rnn = BidirectionalRNN(2*1024, 128, vocab_size)
        self.linear_out = nn.Linear(1024, vocab_size)
        self.sigmoid = torch.nn.Sigmoid().to(device)

    def get_cnn(self):
        return self.cnn

    def forward(self, input):
        cnn_output = self.cnn(input)
        attn_output, attn_weights = self.attn(cnn_output, cnn_output, cnn_output)
        rnn_output = self.rnn(torch.cat([cnn_output, attn_output], dim=-1))
        rnn_output[:,:,2:] = self.sigmoid(rnn_output[:,:,2:]) # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
        return rnn_output


# class StrokeRecoveryModel(nn.Module):
#     def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
#         super(StrokeRecoveryModel, self).__init__()
#         self.encoder = CRNNEncoder(vocab_size, device, first_conv_op, first_conv_opts)
#         self.attn = MultiLayerSelfAttention(vocab_size, num_layers=1)
#         self.decoder = RNNDecoder(vocab_size, vocab_size, 16, vocab_size, 2)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def get_cnn(self):
#         return self.encoder.cnn
#
#     def forward(self, input, labels=None):
#         rnn_output = self.encoder(input)
#         seq_len, batch_size, vocab_size = rnn_output.shape
#
#         decoder_state = torch.zeros(1, batch_size, vocab_size).to(rnn_output.device)
#         hidden = None
#         outputs = []
#
#         for i in range(seq_len):
#             torch.cuda.empty_cache()
#             context = self.attn(rnn_output, decoder_state)
#             decoder_state, hidden = self.decoder(decoder_state, context, hidden)
#             decoder_state[:, :, 2:] = self.sigmoid(decoder_state[:, :, 2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
#             outputs.append(decoder_state)
#         return torch.cat(outputs, dim=0)
#
