import torch
from torch import nn
from torch.nn import functional as F


class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder, output_max_len, vocab_size, sos_token):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size
        self.sos_token = sos_token

    def forward(self, line_imgs, online, return_logits=True):
        encoder_output, _ = self.encoder(line_imgs, online)
        seq_len, batch_size, embed_dim = encoder_output.shape

        assert embed_dim == self.vocab_size
        device = encoder_output.device
        decoder_state = torch.zeros(1, batch_size, embed_dim, device=device)
        hidden = None
        tokenized_output = torch.tensor([self.sos_token for i in range(batch_size)], dtype=torch.long, device=device)
        output = F.one_hot(tokenized_output, num_classes=self.vocab_size).to(torch.float)
        if return_logits:
            sequence = [output]
        else:
            sequence = [tokenized_output]
        for i in range(self.output_max_len-1):
            attention_output = self.attention(encoder_output, decoder_state)
            output, decoder_state, hidden = self.decoder(output, attention_output, hidden)
            tokenized_output = torch.argmax(output, dim=1)

            if return_logits:
                sequence.append(output)
            else:
                sequence.append(tokenized_output)

        sequence = torch.stack(sequence, dim=1)
        return sequence
