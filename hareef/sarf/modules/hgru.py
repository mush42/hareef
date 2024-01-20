# coding: utf-8

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
            input, lengths, batch_first=self.batch_first, enforce_sorted=self.training
        )
        output, hx = self.gru(packed_input, hx)
        output, _lengths = pad_packed_sequence(
            output, batch_first=self.batch_first, padding_value=self.pad_idx
        )
        if self.sum_bidi and self.bidirectional:
            output = output[:, :, : self.hidden_size] + output[:, :, self.hidden_size :]
        return output.tanh()


