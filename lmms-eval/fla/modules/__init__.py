# -*- coding: utf-8 -*-

from .convolution import ImplicitLongConvolution, LongConvolution, ShortConvolution
from .fused_bitlinear import BitLinear, FusedBitLinear
from .fused_cross_entropy import FusedCrossEntropyLoss
from .fused_kl_div import FusedKLDivLoss
from .fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from .fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear
)
from fla.modules.l2norm import L2Norm
from fla.modules.layernorm import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from fla.modules.mlp import GatedMLP
from fla.modules.rotary import RotaryEmbedding
from fla.modules.token_shift import TokenShift

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution',
    'BitLinear', 'FusedBitLinear',
    'FusedCrossEntropyLoss', 'FusedLinearCrossEntropyLoss', 'FusedKLDivLoss',
    'L2Norm',
    'GroupNorm', 'GroupNormLinear', 'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedLayerNormGated', 'FusedLayerNormSwishGate', 'FusedLayerNormSwishGateLinear',
    'FusedRMSNormGated', 'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'RotaryEmbedding',
    'TokenShift'
]
