from torch import nn
import torch
#import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_dim, n_layers, hidden_dim, dropout=0.01, char_freq=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.char_freq = char_freq
        self.rnn_input_dim = embed_dim + context_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(input_size=self.rnn_input_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout)
        self.linear_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, character, context, hidden=None):
        character = character.argmax(dim=1)
        character_embedding = self.embedding(character)

        rnn_input = torch.cat([character_embedding, context], dim=1)
        assert rnn_input.shape[1] == self.rnn_input_dim
        rnn_input = rnn_input.view(1, -1, self.rnn_input_dim)

        decoder_state, hidden = self.rnn(rnn_input, hidden)

        output = self.linear_proj(decoder_state).squeeze()

        # if self.char_freq is not None:
        #     output = output * self.char_freq

        return output, decoder_state, hidden
