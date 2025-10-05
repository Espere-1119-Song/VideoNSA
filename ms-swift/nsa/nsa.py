import torch
from typing import Optional, Tuple, Union
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn import flash_attn_func
import os
import time
from pathlib import Path
from loguru import logger as eval_logger

from fla.ops.utils.pooling import mean_pooling
from fla.ops.nsa.parallel import parallel_nsa_topk

from .compression import compression_attention
from .selection import selection_attention


def nsa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: Optional[torch.Tensor] = None,
    g_slc: Optional[torch.Tensor] = None,
    g_swa: Optional[torch.Tensor] = None,
    block_count: int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    return_attn_weights: bool = False,
    layer_idx: Optional[int] = None,  
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    # Declare global variables at the beginning of function
    global _layer_timings, _timing_stats

    B, M, H, D = q.shape
    _, N, G, _ = k.shape

    assert g_cmp is not None and g_slc is not None and g_swa is not None, "g_cmp, g_slc, and g_swa are required"
    assert k.shape == (B, N, G, D), f"k shape: {k.shape} must be ({B}, {N}, {G}, {D})"
    assert v.shape == (B, N, G, D), f"v shape: {v.shape} must be ({B}, {N}, {G}, {D})"
    assert g_cmp.shape == (B, M, H), f"g_cmp shape: {g_cmp.shape} must be ({B}, {M}, {H})"
    assert g_slc.shape == (B, M, H), f"g_slc shape: {g_slc.shape} must be ({B}, {M}, {H})"
    assert g_swa.shape == (B, M, H), f"g_swa shape: {g_swa.shape} must be ({B}, {M}, {H})"

    if scale is None:
        scale = D ** -0.5

    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)

    # Ensure compressed tensors are on the same device as q
    k_cmp = k_cmp.to(q.device)
    v_cmp = v_cmp.to(q.device)

    N_block = k_cmp.shape[1] 

    def cmp_mask(b, h, q_idx, kv_idx):
        q_blk = q_idx // block_size      
        return kv_idx <= q_blk           

    block_mask = create_block_mask(cmp_mask, B, H, M, N_block)
    # Ensure block_mask is on the same device as the input tensors
    if hasattr(block_mask, 'to'):
        block_mask = block_mask.to(q.device)

    o_cmp, lse_cmp = compression_attention(q, k_cmp, v_cmp, block_mask)

    # Selection branch timing
    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        lse=lse_cmp,
        block_counts=block_count,
        block_size=block_size,
        scale=scale,
        # cu_seqlens=None
    )


    if return_attn_weights:
        o_slc, lse_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale,
            return_attn_probs=True
        )
    else:
        o_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale
        )

    # Sliding window branch timing
    o_swd = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(window_size-1, 0)
    )
    o = o_cmp * g_cmp.unsqueeze(-1) + o_slc * g_slc.unsqueeze(-1) + o_swd * g_swa.unsqueeze(-1)


    if return_attn_weights:
        attn_weights = {
            'compression': {
                'lse': lse_cmp,  # [B, H, M] - log-sum-exp values
                'block_indices': block_indices,  # [B, M, G, T] - selected block indices
            },
            'selection': {
                'lse': lse_slc,  # [B, H, M] - log-sum-exp values
                'block_indices': block_indices,  # [B, M, G, T] - selected block indices
            },
            'sliding_window': {
                'note': 'Flash attention does not return attention weights directly'
            },
            'gating_weights': {
                'g_cmp': g_cmp,  # [B, M, H] - compression gating weights
                'g_slc': g_slc,  # [B, M, H] - selection gating weights
                'g_swa': g_swa,  # [B, M, H] - sliding window gating weights
            },
            'timing_info': timing_info  # Add timing information to weights dict
        }
        return o, attn_weights
    else:
        return o

