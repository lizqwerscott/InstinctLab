from dataclasses import MISSING
from typing import List, Literal, Sequence

from isaaclab.utils import configclass


@configclass
class InstinctRlParallelBlockCfg:
    """Configuration for the encoder network."""

    class_name: str = MISSING
    """The encoder class name. Default is Encoder."""

    component_names: List[str] = MISSING
    """The names of the observation components to be embedded."""

    output_size: int = MISSING
    """The output size of the encoder network."""

    takeout_input_components: bool = True
    """Whether to take out the input components from the embedded obs to the rest of the network."""


@configclass
class InstinctRlMlpCfg(InstinctRlParallelBlockCfg):
    """Configuration for the MLP encoder network."""

    class_name: str = "MlpModel"
    """The encoder class name. Default is MlpModel."""

    hidden_sizes: List[int] = MISSING
    """The hidden dimensions of the encoder network."""

    nonlinearity: str = MISSING
    """The activation function for the encoder network."""


@configclass
class InstinctRlConv2dHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for the Conv2d encoder network."""

    class_name: str = "Conv2dHeadModel"
    """The encoder class name. Default is Conv2dHeadModel."""

    channels: List[int] = MISSING
    """The number of channels."""

    kernel_sizes: List[int] = MISSING
    """The size of the kernel."""

    strides: List[int] = MISSING
    """The stride of the kernel."""

    hidden_sizes: List[int] = MISSING
    """The hidden dimensions of the output mlp head."""

    paddings: List[int] = MISSING
    """The padding of the kernel."""

    nonlinearity: str = MISSING
    """The activation function for the encoder network."""

    use_maxpool: bool = False
    """Whether to use max pooling in the convolutional layers."""


@configclass
class InstinctRlCrossAttnHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for the proprioception-queried cross-attention depth encoder.

    The block tokenizes ``component_names`` (a single image component) with a conv
    stack, runs self-attention over the tokens, then cross-attends with a query
    built from ``info_component_names`` (the proprioceptive components). Only the
    image component is taken out of the obs; the info components still flow to the
    downstream network.
    """

    class_name: str = "CrossAttnFuseHeadModel"
    """The class name. Default is CrossAttnFuseHeadModel."""

    info_component_names: List[str] = MISSING
    """The proprioceptive components used to build the cross-attention query."""

    channels: List[int] = MISSING
    """The number of channels per conv layer. channels[-1] is the token dim (d_model)."""

    kernel_sizes: List[int] = MISSING
    """The size of the kernel per conv layer."""

    strides: List[int] = MISSING
    """The stride per conv layer."""

    paddings: List[int] = MISSING
    """The padding per conv layer."""

    num_heads: int = 4
    """The number of attention heads. channels[-1] must be divisible by this."""

    num_self_attn_layers: int = 1
    """The number of stacked self-attention layers over the image tokens.
    0 skips self-attention (tokenizer -> LN -> K/V, the exact CReF form)."""

    ffn_expansion: int = 2
    """Hidden-size multiplier for the self-attention FFN."""

    info_hidden_sizes: list[int] | None = None
    """Hidden layer widths of the proprioceptive query MLP (info -> query token).
    The output is always d_model. None defaults to [d_model * ffn_expansion]."""

    use_grf: bool = False
    """Whether to apply CReF's Gated Residual Fusion (Eq. 12-14) on the
    concatenated [proprio token ; attended depth token] (dim 2 * d_model)
    before the output projection. With output_size == 2 * d_model the fused
    token passes through unprojected (the exact CReF form)."""

    nonlinearity: str = "ELU"
    """The activation function for the conv/FFN/info-projection."""

    use_maxpool: bool = False
    """Whether the conv tokenizer uses max pooling for downsampling."""


@configclass
class InstinctRlTransformerHeadCfg(InstinctRlParallelBlockCfg):
    """Configuration for the Transformer encoder network."""

    class_name: str = "TransformerHeadModel"
    """The class name. Default is TransformerHeadModel."""

    num_heads: int = 4
    """The number of attention heads."""

    num_layers: int = 1
    """The number of transformer encoder layers."""

    d_model: int = 256
    """The latent size of the transformer encoder."""

    dim_feedforward: int = 512
    """The feedforward dimension of the transformer encoder. Default in Transformer is 2048, we use 512."""

    dropout: float = 0.1
    """The dropout rate."""

    activation: str = "relu"
    """The activation function for the transformer encoder."""

    nonlinearity: str = "ReLU"
    """The nonlinearity layer for the mlp network."""

    layer_norm_eps: float = 1e-5
    """The epsilon value for layer normalization."""

    batch_first: bool = True
    """Whether the input is batch first."""

    norm_first: bool = False
    """Whether to apply normalization first."""

    mask_from_input_dim: int = -1
    """The dimension to get the self-attention mask from the input tensor. If -1, no mask is used."""

    output_selection: Literal["maxpool", "smallest_positive", "smallest_nonnegative"] = "maxpool"
    """The output selection method."""

    input_hidden_sizes: List[int] = []
    """The hidden dimensions of the input mlp head. If None, only a linear layer is used."""

    output_hidden_sizes: List[int] = []
    """The hidden dimensions of the output mlp head. If None, only a linear layer is used."""
