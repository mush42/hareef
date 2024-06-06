import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

from .autoregressive_wrapper import AutoregressiveWrapper
from .continuous_autoregressive_wrapper import ContinuousAutoregressiveWrapper
from .nonautoregressive_wrapper import NonAutoregressiveWrapper
from .x_transformers import (Attention, ContinuousTransformerWrapper,
                             CrossAttender, Decoder, Encoder,
                             TransformerWrapper, ViTransformerWrapper,
                             XTransformer)
from .xl_autoregressive_wrapper import XLAutoregressiveWrapper
