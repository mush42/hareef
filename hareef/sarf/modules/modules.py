import math
import logging

import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvSeparable, EncSepConvLayer, ScaledSinusoidalEmbedding
from .utils import build_activation, pad_list



DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim=256,
        kernel_sizes=[3, 5, 17, 25, 31, 13, 9],
        activation='relu',
        dropout=0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            dim,
            dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=dropout
        )
        self.layers = nn.ModuleList([
            EncSepConvLayer(dim, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)

    def forward(self, x, padding_mask):
        """
        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: x: [B, T, H]
        """
        lx, hx = self.lstm(x)
        lx = lx.tanh()
        lx = self.dropout(lx)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask)

        x = self.layer_norm(x)
        x = x * (1 - padding_mask.float()).transpose(0, 1)[..., None]

        x = x.transpose(0, 1)
        x = lx + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim=256,
        kernel_sizes=[17, 21, 9, 7, 13, 17],
        activation='relu',
        dropout=0.2,
        max_source_positions = DEFAULT_MAX_TARGET_POSITIONS,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            dim,
            dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=dropout
        )
        self.pos_emb = ScaledSinusoidalEmbedding(dim, theta=max_source_positions)
        self.layers = nn.ModuleList([
            EncSepConvLayer(dim, kernel_size, dropout, activation)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask, *, require_w=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
        """
        lx, hx = self.lstm(x)
        lx = lx.tanh()
        lx = self.dropout(lx)

        positions = self.pos_emb(x[..., 0])
        x = x + positions
        x = x * (1 - padding_mask.float())[..., None]
        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn_w = []
        if require_w:
            for layer in self.layers:
                x, attn_w_i = layer(x, encoder_padding_mask=padding_mask, require_w=require_w)
                attn_w.append(attn_w_i)
        else:
            for layer in self.layers:
                # remember to assign back to x
                x = layer(x, encoder_padding_mask=padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        x = lx + x
        return (x, attn_w) if require_w else x


class POSPredictor(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        num_pos_tags,
        n_layers=5,
        intermediate_dim=384,
        kernel_size=5,
        dropout=0.1,
        activation='relu',
        padding='SAME',
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        """
        Args:
            dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            intermediate_dim (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(POSPredictor, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pos_emb = ScaledSinusoidalEmbedding(dim, theta=max_source_positions)
        self.conv = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvSeparable(dim if idx == 0 else intermediate_dim, intermediate_dim, kernel_size),
                build_activation(activation),
                LayerNorm(intermediate_dim, dim=1),
                torch.nn.Dropout(dropout)
            )
            for idx in range(n_layers - 1)
        ])
        self.conv.append(
            torch.nn.Sequential(
                ConvSeparable(intermediate_dim, dim, kernel_size),
                build_activation(activation),
                LayerNorm(dim, dim=1),
                torch.nn.Dropout(dropout)
            )
        )
        self.out_proj = ConvSeparable(dim, output_dim, 1)
        self.proj = nn.Linear(output_dim, num_pos_tags)

    def forward(self, xs, xs_mask):
        """
        :param xs: [B, T, H]
        :return: [B, T]
        """
        positions = self.pos_emb(xs)
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, dim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            xs = xs * (1 - xs_mask.float())[:, None, :]
        xs = self.out_proj(xs)
        xs = xs * (1 - xs_mask.float())[:, None, :]
        preds = self.proj(xs.transpose(1, 2)).squeeze(-1)
        preds = preds.masked_fill(xs_mask.unsqueeze(-1), 0.0)
        return xs.transpose(1, 2), preds


