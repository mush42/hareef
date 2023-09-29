# coding: utf-8

import math
from typing import Optional

import torch
from torch import nn
from torch import einsum
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import commons
from . import attentions


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5).float()

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq.float()
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos.float(), self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale


class TokenEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask



class HGRU(nn.Module):
    def __init__(self, *args, pad_idx, sum_bidi=False, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)
        self.batch_first = self.gru.batch_first
        self.hidden_size = self.gru.hidden_size
        self.bidirectional = self.gru.bidirectional
        self.pad_idx = pad_idx
        self.sum_bidi = sum_bidi

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        packed_input = pack_padded_sequence(
            input, lengths, batch_first=self.batch_first
        )
        output, hx = self.gru(packed_input, hx)
        output, _lengths = pad_packed_sequence(
            output, batch_first=self.batch_first, padding_value=self.pad_idx
        )
        if self.sum_bidi and self.bidirectional:
            output = output[:, :, : self.hidden_size] + output[:, :, self.hidden_size :]
        return output.tanh()


