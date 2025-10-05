import torch
from typing import Optional, Tuple, Union
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn import flash_attn_func
import os
import time
from pathlib import Path
from loguru import logger as eval_logger

import sys
sys.path.append("/home/ubuntu/jianwen-us-midwest-1/tulab/enxin/projects/ms-swift")
from fla.ops.utils.pooling import mean_pooling
from fla.ops.nsa.parallel import parallel_nsa_topk

from .compression import compression_attention
from .selection import selection_attention

# Global timing collection for statistics
_layer_timings = {}  # {layer_idx: {'compression': [], 'selection': [], 'sliding_window': [], 'total': []}}
_timing_stats = {'compression': [], 'selection': [], 'sliding_window': [], 'total': []}


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

    # Only log tensor shapes for first few calls for debugging
    call_count = len(_timing_stats.get('total', [])) + 1
    if call_count <= 2:
        eval_logger.info(f"NSA Input Shapes (call {call_count}) - q: {q.shape}, k: {k.shape}, v: {v.shape}")

    # Initialize timing measurements
    start_total = time.perf_counter()

    # Compression branch timing
    start_compression = time.perf_counter()
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
    end_compression = time.perf_counter()
    compression_time = end_compression - start_compression

    # Selection branch timing
    start_selection = time.perf_counter()
    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        lse=lse_cmp,
        block_counts=block_count,
        block_size=block_size,
        scale=scale,
        # cu_seqlens=None
    )

    # 根据return_attn_weights参数决定是否返回attention weights
    if return_attn_weights:
        o_slc, lse_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale,
            return_attn_probs=True
        )
    else:
        o_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale
        )
    end_selection = time.perf_counter()
    selection_time = end_selection - start_selection

    # Sliding window branch timing
    start_sliding = time.perf_counter()
    o_swd = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(window_size-1, 0)
    )
    end_sliding = time.perf_counter()
    sliding_time = end_sliding - start_sliding

    o = o_cmp * g_cmp.unsqueeze(-1) + o_slc * g_slc.unsqueeze(-1) + o_swd * g_swa.unsqueeze(-1)

    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Collect timing data for statistics
    # Add to global statistics
    _timing_stats['compression'].append(compression_time)
    _timing_stats['selection'].append(selection_time)
    _timing_stats['sliding_window'].append(sliding_time)
    _timing_stats['total'].append(total_time)


    # If layer_idx is provided, also collect per-layer statistics
    if layer_idx is not None:
        if layer_idx not in _layer_timings:
            _layer_timings[layer_idx] = {'compression': [], 'selection': [], 'sliding_window': [], 'total': []}

        _layer_timings[layer_idx]['compression'].append(compression_time)
        _layer_timings[layer_idx]['selection'].append(selection_time)
        _layer_timings[layer_idx]['sliding_window'].append(sliding_time)
        _layer_timings[layer_idx]['total'].append(total_time)

    # Calculate and log statistics every 10 calls or for the first few layers
    call_count = len(_timing_stats['total'])
    if call_count <= 5 or call_count % 10 == 0:
        # Calculate total times
        total_compression = sum(_timing_stats['compression'])
        total_selection = sum(_timing_stats['selection'])
        total_sliding = sum(_timing_stats['sliding_window'])
        total_all = sum(_timing_stats['total'])

        # Calculate percentages
        comp_pct = (total_compression / total_all * 100) if total_all > 0 else 0
        sel_pct = (total_selection / total_all * 100) if total_all > 0 else 0
        slide_pct = (total_sliding / total_all * 100) if total_all > 0 else 0

        # Calculate average per call for easier comparison
        avg_per_call = total_all / call_count if call_count > 0 else 0
        avg_comp_per_call = total_compression / call_count if call_count > 0 else 0
        avg_sel_per_call = total_selection / call_count if call_count > 0 else 0
        avg_slide_per_call = total_sliding / call_count if call_count > 0 else 0

        eval_logger.info(f"NSA_TOTAL_STATS (after {call_count} calls): Compression={total_compression:.3f}s ({comp_pct:.1f}%), Selection={total_selection:.3f}s ({sel_pct:.1f}%), SlidingWindow={total_sliding:.3f}s ({slide_pct:.1f}%), Total={total_all:.3f}s")
        print(f"NSA_TOTAL_STATS (after {call_count} calls): Compression={total_compression:.3f}s ({comp_pct:.1f}%), Selection={total_selection:.3f}s ({sel_pct:.1f}%), SlidingWindow={total_sliding:.3f}s ({slide_pct:.1f}%), Total={total_all:.3f}s")

        # Also show average per call for easy comparison across different max_frames
        eval_logger.info(f"NSA_AVG_PER_CALL: Compression={avg_comp_per_call:.6f}s, Selection={avg_sel_per_call:.6f}s, SlidingWindow={avg_slide_per_call:.6f}s, Total={avg_per_call:.6f}s")
        print(f"NSA_AVG_PER_CALL: Compression={avg_comp_per_call:.6f}s, Selection={avg_sel_per_call:.6f}s, SlidingWindow={avg_slide_per_call:.6f}s, Total={avg_per_call:.6f}s")

        # Also show per-layer totals if available
        if _layer_timings:
            layer_stats = []
            for layer_id in sorted(_layer_timings.keys()):
                layer_data = _layer_timings[layer_id]
                if layer_data['total']:  # Check if has data
                    layer_total_comp = sum(layer_data['compression'])
                    layer_total_sel = sum(layer_data['selection'])
                    layer_total_slide = sum(layer_data['sliding_window'])
                    layer_total_all = sum(layer_data['total'])
                    layer_stats.append(f"L{layer_id}: C={layer_total_comp:.3f}s S={layer_total_sel:.3f}s SW={layer_total_slide:.3f}s (Total={layer_total_all:.3f}s)")

            if layer_stats:
                eval_logger.info(f"NSA_LAYER_TOTALS: {' | '.join(layer_stats)}")
                print(f"NSA_LAYER_TOTALS: {' | '.join(layer_stats)}")

        # Force output to ensure visibility
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

    # Create timing info dictionary for return value
    timing_info = {
        'compression_time': compression_time,
        'selection_time': selection_time,
        'sliding_window_time': sliding_time,
        'total_time': total_time
    }

    if return_attn_weights:
        # 构建attention weights字典
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

