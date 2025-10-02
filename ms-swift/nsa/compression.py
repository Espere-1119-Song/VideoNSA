import torch
from torch.nn.attention.flex_attention import flex_attention


@torch.compile
def compression_attention(q, k_cmp, v_cmp, block_mask):
    o_cmp, lse_cmp = flex_attention(
        q.transpose(1, 2),
        k_cmp.transpose(1, 2),
        v_cmp.transpose(1, 2),
        block_mask=block_mask,
        enable_gqa=True,
        return_lse=True
    )
    return o_cmp.transpose(1, 2), lse_cmp