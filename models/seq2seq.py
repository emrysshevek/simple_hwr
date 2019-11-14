import heapq

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder, output_max_len, vocab_size, sos_token):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size
        self.sos_token = sos_token

        self.linear_out = nn.Linear(vocab_size * 2, vocab_size)

    def forward(self, line_imgs, online, labels=None, teacher_force_rate=0.9, return_logits=True, train=True):
        if labels is None:
            train = False
            teacher_force_rate = 0

        encoder_output, _ = self.encoder(line_imgs, online)
        seq_len, batch_size, embed_dim = encoder_output.shape

        assert embed_dim == self.vocab_size

        device = encoder_output.device

        hidden = None
        context = torch.zeros(batch_size, embed_dim, device=device)
        decoder_state = torch.zeros(1, batch_size, embed_dim, device=device)
        tokenized_output = torch.tensor([self.sos_token for i in range(batch_size)], dtype=torch.long, device=device)
        output = self.one_hot(tokenized_output)

        if return_logits:
            sequence = [output]
        else:
            sequence = [tokenized_output]

        prev_attn = None
        for i in range(self.output_max_len-1):
            context, prev_attn = self.attention(encoder_output.permute(1, 0, 2), decoder_state.permute(1, 0, 2), prev_attn, mode='soft' if train else 'hard')
            decoder_state, hidden = self.decoder(output, context, hidden)
            # print(context)
            output = self.linear_out(torch.cat([decoder_state.squeeze(), context], dim=-1))
            tokenized_output = torch.argmax(output, dim=1)

            if return_logits:
                sequence.append(output)
            else:
                sequence.append(tokenized_output)

            if np.random.rand() < teacher_force_rate:
                teacher_input = labels[:, i+1].to(device)
                output = self.one_hot(teacher_input)

        sequence = torch.stack(sequence, dim=1)
        return encoder_output, sequence

    def one_hot(self, x):
        return F.one_hot(x.to(torch.long), num_classes=self.vocab_size).to(torch.float)
