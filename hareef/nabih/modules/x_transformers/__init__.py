import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from .x_transformers import XTransformer, Encoder, Decoder, CrossAttender, Attention, TransformerWrapper, ViTransformerWrapper, ContinuousTransformerWrapper

from .autoregressive_wrapper import AutoregressiveWrapper
from .nonautoregressive_wrapper import NonAutoregressiveWrapper
from .continuous_autoregressive_wrapper import ContinuousAutoregressiveWrapper
from .xl_autoregressive_wrapper import XLAutoregressiveWrapper
