import torch
from torch import nn
from models.CoordConv import CoordConv
from models.encoder import CRNNEncoder
from models.attention import MultiLayerSelfAttention
from models.decoder import RNNDecoder


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", first_conv_op=CoordConv, first_conv_opts=None):
        super(StrokeRecoveryModel, self).__init__()
        self.encoder = CRNNEncoder(vocab_size, device, first_conv_op, first_conv_opts)
        self.attn = MultiLayerSelfAttention(vocab_size, num_layers=1)
        self.decoder = RNNDecoder(vocab_size, vocab_size, 16, vocab_size, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def get_cnn(self):
        return self.encoder.cnn

    def forward(self, input):
        # print(torch.cuda.memory_allocated()/1e9)
        rnn_output = self.encoder(input)
        seq_len, batch_size, vocab_size = rnn_output.shape

        decoder_state = torch.zeros(1, batch_size, vocab_size).to(rnn_output.device)
        hidden = None
        outputs = []

        for i in range(seq_len):
            torch.cuda.empty_cache()
            context = self.attn(rnn_output, decoder_state)
            decoder_state, hidden = self.decoder(decoder_state, context, hidden)
            decoder_state[:, :, 2:] = self.sigmoid(decoder_state[:, :, 2:])  # force SOS (start of stroke) and EOS (end of stroke) to be probabilistic
            outputs.append(decoder_state)
        return torch.cat(outputs, dim=0)

