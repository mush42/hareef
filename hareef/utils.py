# coding: utf-8

import io
import os
import re
from itertools import repeat
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import torch
from diacritization_evaluation import der, wer
from diacritization_evaluation import util
from torch import nn

from hareef.constants import DIACRITIC_LABELS
from hareef.text_cleaners import valid_arabic_cleaner


CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>[0-9]+)-step=(?P<step>[0-9]+)")


def sequence_mask(length, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)



def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def plot_alignment(alignment: torch.Tensor, path: str, global_step: Any = 0):
    """
    Plot alignment and save it into a path
    Args:
    alignment (Tensor): the encoder-decoder alignment
    path (str): a path used to save the alignment plot
    global_step (int): used in the name of the output alignment plot
    """
    alignment = alignment.squeeze(1).transpose(0, 1).cpu().detach().numpy()
    fig, axs = plt.subplots()
    img = axs.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(img, ax=axs)
    xlabel = "Decoder timestep"
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    plot_name = f"{global_step}.png"
    plt.savefig(os.path.join(path, plot_name), dpi=300, format="png")
    plt.close()


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).bool().zero_()
    for idx, length in enumerate(memory_lengths):
        mask[idx][:length] = 1
    return ~mask


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def get_encoder_layers_attentions(model):
    attentions = []
    for layer in model.encoder.layers:
        attentions.append(layer.self_attention.attention)
    return attentions


def get_decoder_layers_attentions(model):
    self_attns, src_attens = [], []
    for layer in model.decoder.layers:
        self_attns.append(layer.self_attention.attention)
        src_attens.append(layer.encoder_attention.attention)
    return self_attns, src_attens


def display_attention(
    attention, path, global_step: int, name="att", n_heads=4, n_rows=2, n_cols=2
):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 15))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].transpose(0, 1).cpu().detach().numpy()
        cax = ax.imshow(_attention, aspect="auto", origin="lower", interpolation="none")

    plot_name = f"{global_step}-{name}.png"
    plt.savefig(os.path.join(path, plot_name), dpi=300, format="png")
    plt.close()


def plot_multi_head(model, path, global_step):
    encoder_attentions = get_encoder_layers_attentions(model)
    decoder_attentions, attentions = get_decoder_layers_attentions(model)
    for i in range(len(attentions)):
        display_attention(
            attentions[0][0], path, global_step, f"encoder-decoder-layer{i + 1}"
        )
    for i in range(len(decoder_attentions)):
        display_attention(
            decoder_attentions[0][0], path, global_step, f"decoder-layer{i + 1}"
        )
    for i in range(len(encoder_attentions)):
        display_attention(
            encoder_attentions[0][0], path, global_step, f"encoder-layer {i + 1}"
        )


def make_src_mask(src, pad_idx=0):
    # src = [batch size, src len]

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # src_mask = [batch size, 1, 1, src len]

    return src_mask


def get_angles(pos, i, model_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(model_dim)[np.newaxis, :],
        model_dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.from_numpy(pos_encoding)


def calculate_error_rates(
    original_file_path: str, target_file_path: str
) -> dict[str, float]:
    """
    Calculates der/wer error rates from paths
    """
    assert os.path.isfile(original_file_path)
    assert os.path.isfile(target_file_path)

    _wer = wer.calculate_wer_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=True
    )

    _wer_without_case_ending = wer.calculate_wer_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=False
    )

    _der = der.calculate_der_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=True
    )

    _der_without_case_ending = der.calculate_der_from_path(
        original_path=original_file_path, predicted_path=target_file_path, case_ending=False
    )

    return {
        "DER": _der,
        "WER": _wer,
        "DER*": _der_without_case_ending,
        "WER*": _wer_without_case_ending,
    }


def categorical_accuracy(preds, y, tag_pad_idx, device="cuda"):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    non_pad_elements = torch.nonzero((y != tag_pad_idx))
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


def make_src_mask(src: torch.Tensor, pad_idx=0):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_trg_mask(trg, trg_pad_idx=0):
    # trg = [batch size, trg len]

    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)

    # trg_pad_mask = [batch size, 1, 1, trg len]

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()

    # trg_sub_mask = [trg len, trg len]

    trg_mask = trg_pad_mask & trg_sub_mask

    # trg_mask = [batch size, 1, trg len, trg len]

    return trg_mask


def find_last_checkpoint(logs_root_directory):
    checkpoints_dir = Path(logs_root_directory)
    available_checkpoints = {
        file: CHECKPOINT_RE.match(file.stem)
        for file in (
            item for item in checkpoints_dir.rglob("*.ckpt")
            if item.is_file()
        )
    }
    available_checkpoints = {
        filename: (int(match.groupdict()["epoch"]), int(match.groupdict()["step"]))
        for (filename, match) in available_checkpoints.items()
        if match is not None
    }
    available_checkpoints = sorted(available_checkpoints.items(), key=lambda item: item[1])
    checkpoint = more_itertools.last(available_checkpoints, default=None)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoints were found")
    filename, (epoch, step) = checkpoint
    return os.fspath(filename.absolute()), epoch, step


def format_as_table(*cols: tuple[str, list[str]]) -> str:
    """Taken from lightening"""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width
    return summary


def format_error_rates_as_table(error_rates):
    metrics, values = [e[0] for e in error_rates], [e[1] for e in error_rates]
    cols = [
        ("".ljust(10), ["   DER", "   WER"]),
        ("With CE".ljust(10), [error_rates["DER"], error_rates["WER"]]),
        ("Without CE".ljust(10), [error_rates["DER*"], error_rates["WER*"]]),
    ]
    return format_as_table(*cols)


def generate_confusion_matrix(test_lines, pred_lines, plot=False, fig_save_path=None):
    confusion_dict = {}
    for test_line, pred_line in zip(test_lines, pred_lines):
        test_line, pred_line = valid_arabic_cleaner(test_line), valid_arabic_cleaner(pred_line)
        test_diacritics = util.extract_haraqat(test_line)[-1]
        pred_diacritics = util.extract_haraqat(pred_line)[-1]
        assert len(test_diacritics) == len(pred_diacritics), "Ground truth and predictions must be equal in length"
        for t_d, p_d in zip(test_diacritics, pred_diacritics):
            try:
                confusion_dict[t_d][p_d] += 1
            except KeyError:
                try:
                    confusion_dict[t_d][p_d] = 1
                except KeyError:
                    confusion_dict[t_d] = {p_d: 1}
    ys = set()
    for d in confusion_dict.values():
        ys = ys.union(d.keys())
    for info in confusion_dict.values():
        for p_d in ys:
            if p_d not in info:
                info[p_d] = 0

    confusion_matrix = np.array([[confusion_dict[x][y] for y in sorted(ys)] for x in sorted(confusion_dict.keys())])
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=-1, keepdims=True)
    plt.figure(dpi=150)
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.ylabel('Test data')
    plt.xlabel('Predicted data')
    ax = plt.gca()
    ax.set_yticks(np.arange(len(confusion_dict)))
    ax.set_xticks(np.arange(len(ys)))
    ax.set_yticklabels([DIACRITIC_LABELS.get(x, "None") for x in sorted(confusion_dict.keys())], fontname='Arial', fontsize=7)
    ax.set_xticklabels([DIACRITIC_LABELS.get(x, "None") for x in sorted(ys)], fontname='Arial', fontsize=7)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", va='bottom', rotation_mode="anchor")
    for i in range(len(confusion_dict)):
        for j in range(len(ys)):
            ax.text(j, i, '{:.1%}'.format(confusion_matrix[i, j]), ha="center", va="center", color="slategrey", fontsize=5)
    plt.tight_layout()

    if fig_save_path:
        plt.savefig(str(fig_save_path))

    if plot:
        plt.show()

    return confusion_matrix


def get_model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    num_bytes = len(buf.getbuffer())
    return num_bytes //1e6
