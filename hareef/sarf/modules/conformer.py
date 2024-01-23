import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

# from .attention import MultiheadAttention
from .hgru import HGRU


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
        seq_len = x.size(1)
        device = x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos.float(), self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale


class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerGRUModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, dropout=0.0, pad_idx=0):
        super(ConformerGRUModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.layernorm = nn.LayerNorm(dim)
        self.gru1 = HGRU(
            dim, inner_dim,  batch_first=True, bidirectional=True, pad_idx=pad_idx, sum_bidi=True
        )
        self.act1 = nn.GLU(dim=1)
        self.gru2 = HGRU(
            dim, inner_dim,  batch_first=True, bidirectional=True, pad_idx=pad_idx, sum_bidi=True
        )
        self.batchnorm = nn.BatchNorm1d(inner_dim)
        self.act2 = nn.SiLU()
        self.gru3 = HGRU(
            inner_dim, dim,  batch_first=True, bidirectional=True, pad_idx=pad_idx, sum_bidi=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = self.layernorm(x)
        x = self.gru1(x, lengths)
        x = self.act1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.gru2(x, lengths)
        x = self.batchnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act2(x)
        x = self.gru3(x, lengths)
        return self.dropout(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout)

    def forward(self, x, mask):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, cgm_expansion_factor=2, 
                 ffm_dropout=0., attn_dropout=0., cgm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerGRUModule(dim, cgm_expansion_factor, dropout=cgm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, lengths, mask):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x, mask)
        x = x + self.ccm(x, lengths)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

