import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, input_dim, embed_dim, attn='mono', device=torch.device('cpu')):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        if attn == 'mono':
            self.attn = MonotonicAttention(input_dim, embed_dim, device=device)
        elif attn == 'chunk':
            self.attn = MoChA(input_dim, embed_dim, device=device)
        elif attn == 'self':
            self.attn = MultiHeadAttention(input_dim, num_heads=1)
        self.linear_proj = nn.Linear(2*input_dim, input_dim)

    def forward(self, encoder_outputs, decoder_h, prev_alpha=None, mode='hard'):
        context, attn_weights = self.attn(encoder_outputs, decoder_h, prev_alpha, mode=mode)
        context = self.linear_proj(torch.cat([context, decoder_h.squeeze()], dim=-1))
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=5, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.n_layers = num_layers
        self.layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads, dropout) for i in range(num_layers)])

    def clone(self, tensor, n):
        return [tensor.clone() for i in range(n)]

    def forward(self, encoder_outputs, decoder_h, previous_alpha=None, mode=None):
        q, k, v = self.clone(encoder_outputs.transpose(0, 1), 3)
        for layer in self.layers[:-1]:
            result, weights = layer(q, k, v)
            q, k, v = self.clone(result, 3)
        q = decoder_h.transpose(0,1).clone()
        attn, weights = self.layers[-1](q, k, v)
        return attn.squeeze(), weights


class Energy(nn.Module):
    def __init__(self, input_dim, attn_dim=128, init_r=-4):
        """
        [Modified Bahdahnau attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        Used for Monotonic Attention and Chunk Attention
        """
        super().__init__()
        self.tanh = nn.Tanh()
        self.W = nn.Linear(input_dim, attn_dim, bias=False)
        self.V = nn.Linear(input_dim, attn_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(attn_dim).normal_())

        self.v = nn.utils.weight_norm(nn.Linear(attn_dim, 1))
        self.v.weight_g.data = torch.Tensor([1 / attn_dim]).sqrt()

        self.r = nn.Parameter(torch.Tensor([init_r]))

    def forward(self, encoder_outputs, decoder_h, prev_attn=None):
        """
        Args:
            encoder_outputs: [batch_size, sequence_length, input_dim]
            decoder_h: [batch_size, 1, input_dim]
            prev_attn: [batch_size, sequence_length]
        Return:
            Energy [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        # encoder_outputs = encoder_outputs.contiguous().view(-1, enc_dim)
        energy = self.tanh(self.W(encoder_outputs) +
                           self.V(decoder_h).repeat(1, sequence_length, 1) +
                           self.b)
        energy = self.v(energy).squeeze(-1) + self.r

        return energy.view(batch_size, sequence_length)


class MonotonicAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, device=torch.device('cpu')):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()

        self.monotonic_energy = Energy(input_dim=input_dim, attn_dim=embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def gaussian_noise(self, *size):
        """Additive gaussian nosie to encourage discreteness"""
        # if torch.cuda.is_available():
        #     return torch.cuda.FloatTensor(*size).normal_()
        # else:
        #     return torch.Tensor(*size).normal_()
        return torch.Tensor(*size).normal_().to(self.device)

    def safe_cumprod(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]
        * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
        * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
        """
        batch_size, sequence_length = x.size()
        # if torch.cuda.is_available():
        #     one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        # else:
        #     one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        one_x = torch.cat([torch.ones(batch_size, 1).to(self.device), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        p_select = self.sigmoid(monotonic_energy + self.gaussian_noise(monotonic_energy.size()))
        cumprod_1_minus_p = self.safe_cumprod(1 - p_select)

        if previous_alpha is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            alpha = torch.zeros(batch_size, sequence_length).to(self.device)
            alpha[:, 0] = torch.ones(batch_size)
            # if torch.cuda.is_available:
            #     alpha = alpha.cuda()

        else:
            alpha = p_select * cumprod_1_minus_p * \
                torch.cumsum(previous_alpha / cumprod_1_minus_p, dim=1)

        return alpha

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            attention = torch.zeros(batch_size, sequence_length).to(self.device)
            attention[:, 0] = torch.ones(batch_size)
            # if torch.cuda.is_available:
            #     attention = attention.cuda()
        else:
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
            monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
            # print(monotonic_energy.device)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (monotonic_energy > 0).float()
            p_select = above_threshold * torch.cumsum(previous_attention, dim=1)
            attention = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            attended = attention.sum(dim=1)
            for batch_i in range(batch_size):
                if not attended[batch_i]:
                    attention[batch_i, -1] = 1

            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        return attention

    def forward(self, encoder_outputs, decoder_h, prev_alpha=None, mode='hard'):
        """

        :param encoder_outputs:
        :param decoder_h:
        :param prev_alpha:
        :param mode:
        :return context:
        """
        if mode == 'hard':
            attn = self.hard(encoder_outputs, decoder_h, prev_alpha)
        elif mode == 'soft':
            attn = self.soft(encoder_outputs, decoder_h, prev_alpha)
        context = torch.matmul(encoder_outputs.transpose(-2, -1), attn.unsqueeze(-1)).squeeze()
        return context, attn


class MoChA(MonotonicAttention):
    def __init__(self, input_dim, embed_dim, chunk_size=3, device=torch.device('cpu')):
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__(input_dim, embed_dim)
        self.chunk_size = chunk_size
        self.chunk_energy = Energy(input_dim=input_dim, attn_dim=embed_dim)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def moving_sum(self, x, back, forward):
        """Parallel moving sum with 1D Convolution"""
        # Pad window before applying convolution
        # [batch_size,    back + sequence_length + forward]
        x_padded = F.pad(x, pad=[back, forward])

        # Fake channel dimension for conv1d
        # [batch_size, 1, back + sequence_length + forward]
        x_padded = x_padded.unsqueeze(1)

        # Apply conv1d with filter of all ones for moving sum
        filters = torch.ones(1, 1, back + forward + 1).to(self.device)
        # if torch.cuda.is_available():
        #     filters = filters.cuda()
        x_sum = F.conv1d(x_padded, filters)

        # Remove fake channel dimension
        # [batch_size, sequence_length]
        return x_sum.squeeze(1)

    def chunkwise_attention_soft(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # Numerical stability
        # Divide by same exponent => doesn't affect softmax
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # Limit range of logit
        exp_u = torch.clamp(exp_u, min=1e-5)

        # Moving sum:
        # Zero-pad (chunk size - 1) on the left + 1D conv with filters of 1s.
        # [batch_size, sequence_length]
        denominators = self.moving_sum(exp_u,
                                       back=self.chunk_size - 1, forward=0)

        # Compute beta (MoChA weights)
        beta = exp_u * self.moving_sum(alpha / denominators,
                                       back=0, forward=self.chunk_size - 1)
        return beta

    def chunkwise_attention_hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        # [batch_size]
        attended_indices = monotonic_attention.nonzero().cpu().data[:, 1].tolist()

        i = [[], []]
        total_i = 0
        for batch_i, attended_idx in enumerate(attended_indices):
            for window in range(self.chunk_size):
                if attended_idx - window >= 0:
                    i[0].append(batch_i)
                    i[1].append(attended_idx - window)
                    total_i += 1
        i = torch.LongTensor(i)
        v = torch.FloatTensor([1] * total_i)
        mask = torch.sparse.FloatTensor(i, v, monotonic_attention.size())
        mask = ~mask.to_dense().byte().to(self.device)
        # mask = ~mask.to_dense().cuda().byte()

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(mask, -float('inf'))
        return masked_energy

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """
        Soft monotonic chunkwise attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
            beta [batch_size, sequence_length]
        """
        alpha = super().soft(encoder_outputs, decoder_h, previous_alpha)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        beta = self.chunkwise_attention_soft(alpha, chunk_energy)
        return alpha, beta

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        """
        Hard monotonic chunkwise attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            monotonic_attention [batch_size, sequence_length]: hard alpha
            chunkwise_attention [batch_size, sequence_length]: hard beta
        """
        # hard attention (one-hot)
        # [batch_size, sequence_length]
        monotonic_attention = super().hard(encoder_outputs, decoder_h, previous_attention)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        masked_energy = self.chunkwise_attention_hard(monotonic_attention, chunk_energy)
        chunkwise_attention = self.softmax(masked_energy)
        return monotonic_attention, chunkwise_attention

    def forward(self, encoder_outputs, decoder_h, prev_alpha=None, mode='hard'):
        if mode == 'hard':
            mono_attn, chunk_attn = self.hard(encoder_outputs, decoder_h, prev_alpha)
        elif mode == 'soft':
            mono_attn, chunk_attn = self.soft(encoder_outputs, decoder_h, prev_alpha)
        attn = chunk_attn.unsqueeze(-1)
        encoder_outputs = encoder_outputs.transpose(-2, -1)
        return torch.matmul(encoder_outputs, attn).squeeze(), attn.squeeze()