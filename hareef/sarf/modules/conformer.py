import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum

from .hgru import HGRU


def positional_encoding(d_model: int, length: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-4):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)


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


class ConformerMultiHeadedSelfAttention(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.attention = RelativeMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = key.size()  # pylint: disable=unused-variable
        encoding = encoding[:, : key.shape[1]]
        encoding = encoding.repeat(batch_size, 1, 1)
        outputs, attn = self.attention(query, key, value, pos_embedding=encoding, mask=mask)
        outputs = self.dropout(outputs)
        return outputs, attn


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        u_bias = self.u_bias.expand_as(query)
        v_bias = self.v_bias.expand_as(query)
        a = (query + u_bias).transpose(1, 2)
        content_score = a @ key.transpose(2, 3)
        b = (query + v_bias).transpose(1, 2)
        pos_score = b @ pos_embedding.permute(0, 2, 3, 1)
        pos_score = self._relative_shift(pos_score)

        score = content_score + pos_score
        score = score * (1.0 / self.sqrt_dim)

        score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)

        context = (attn @ value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:  # pylint: disable=no-self-use
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = torch.zeros((batch_size, num_heads, seq_length1, 1), device=pos_score.device)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score


class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            LayerNorm(dim),
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
        self.layernorm = LayerNorm(dim)
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
        self.emb_dim = dim
        self.attn = ConformerMultiHeadedSelfAttention(dim, n_head, dropout)
        self.layernorm = LayerNorm(dim)

    def forward(self, x, mask):
        pos_encoding = positional_encoding(self.emb_dim, x.size(1), device=x.device)
        x = self.layernorm(x)
        x, _ = self.attn(x,  x, x, mask, pos_encoding)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, cgm_expansion_factor=2, 
                 ffm_dropout=0., attn_dropout=0., cgm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.cgm = ConformerGRUModule(dim, cgm_expansion_factor, dropout=cgm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = LayerNorm(dim)

    def forward(self, x, lengths, mask):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x, mask)
        x = x + self.cgm(x, lengths)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

