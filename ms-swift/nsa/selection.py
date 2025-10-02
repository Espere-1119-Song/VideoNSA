import torch
import triton
import triton.language as tl
import triton.testing

from fla.ops.nsa.parallel import ParallelNSAFunction


# Autotune configurations for the forward kernel
_sel_attn_fwd_configs = [
    triton.Config({}, num_warps=num_warps)
    for num_warps in [1, 2, 4, 8]
]

# Autotune configurations for the backward preprocess kernel
_sel_attn_bwd_preprocess_configs = [
    triton.Config({'BLOCK_M': 16, 'num_stages': 1, 'num_warps': 4}, num_ctas=1),
    triton.Config({'BLOCK_M': 32, 'num_stages': 1, 'num_warps': 4}, num_ctas=1),
    triton.Config({'BLOCK_M': 16, 'num_stages': 2, 'num_warps': 4}, num_ctas=1),
    triton.Config({'BLOCK_M': 32, 'num_stages': 2, 'num_warps': 4}, num_ctas=1),
    triton.Config({'BLOCK_M': 16, 'num_stages': 1, 'num_warps': 8}, num_ctas=1),
    triton.Config({'BLOCK_M': 32, 'num_stages': 1, 'num_warps': 8}, num_ctas=1),
]

# Autotune configurations for the main backward kernel
_sel_attn_bwd_configs = [
    triton.Config({}, num_warps=num_warps)
    for num_warps in [1, 2, 4, 8]
]

@triton.autotune( # Decorate the kernel
    configs=_sel_attn_fwd_configs,
    key=['M', 'N', 'D', 'SELECTION_BLOCK_SIZE', 'T', 'HEADS_PER_GROUP', 'causal'],
)
@triton.jit
def _sel_attn_fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Top_idx: tl.tensor,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    Out: tl.tensor,
    Lse: tl.tensor,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kg, stride_kn, stride_kd,
    stride_vb, stride_vg, stride_vn, stride_vd,
    stride_tb, stride_tg, stride_tm, stride_tt, 
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    DP: tl.constexpr,
    SELECTION_BLOCK_SIZE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    OFFSET_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    stride_hg = stride_qh * HEADS_PER_GROUP

    b = tl.program_id(0)
    m = tl.program_id(1) + OFFSET_M
    g = tl.program_id(2)

    # Base pointers
    q_base = Q + b * stride_qb + m * stride_qm + g * stride_hg
    k_base = K + b * stride_kb + g * stride_kg
    v_base = V + b * stride_vb + g * stride_vg
    t_base = Top_idx + b * stride_tb + m * stride_tm + g * stride_tg
    o_base = Out + b * stride_ob + m * stride_om + g * stride_hg
    l_base = Lse + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP

    # Offsets
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_n = tl.arange(0, SELECTION_BLOCK_SIZE)

    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)  # Keep as float16
 
    max_log   = tl.full([BLOCK_H], float('-inf'), dtype=tl.float32)
    sum_exp   = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    accum     = tl.zeros([BLOCK_H, DP], dtype=tl.float32)
    # 1/ln(2) = 1.44269504
    # log_scale = softmax_scale * 1.44269504

    max_col = max(0, N - M + m) if causal else N

    for idx in range(T):
        # NOTE: Ideally we load top_idx outside the loop, this can be done with a gather which will
        # supported in future versions of Triton
        top = tl.load(t_base + idx * stride_tt)

        col = top * SELECTION_BLOCK_SIZE 
        col = tl.multiple_of(col, SELECTION_BLOCK_SIZE)

        if not causal or (col <= max_col and col >= 0):
            cols = col + offs_n
            mask_n = cols < N

            k_ptrs = k_base + offs_d[:, None] * stride_kd + cols[None, :] * stride_kn
            k_blck = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

            v_ptrs = v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_blck = tl.load(v_ptrs, mask=mask_d[None, :] & mask_n[:, None], other=0.0).to(tl.float32)
            
            # qk = tl.dot(q_blck, k_blck) * log_scale # [BH, BN]
            qk = tl.dot(q_blck, k_blck) * softmax_scale # [BH, BN]
            # NOTE: We can move the multiplication by softmax_scale outside the loop

            causal_mask = cols <= max_col
            qk = tl.where(causal_mask[None, :], qk, float('-inf'))
            
            # stable mx-log-sum-exp
            new_max = tl.maximum(max_log, tl.max(qk, axis=1)) # [BH]
            # exp_qk  = tl.math.exp2(qk - new_max[:, None]) # [BH, BN]
            exp_qk  = tl.math.exp(qk - new_max[:, None]) # [BH, BN]
            sum_qk  = tl.sum(exp_qk, axis=1) # [BH]

            # alpha   = tl.math.exp2(max_log - new_max) # [BH]
            alpha   = tl.math.exp(max_log - new_max) # [BH]
            sum_exp = sum_exp * alpha + sum_qk # [BH]
            accum   = accum * alpha[:, None] # [BH, DP]

            accum = tl.dot(exp_qk, v_blck, accum)  # [BH, DP]
            max_log = new_max

    # epilog
    # fin_log = max_log + tl.math.log2(sum_exp) # [BH]
    # fin_log *= 0.69314718
    fin_log = max_log + tl.math.log(sum_exp) # [BH]
    out_vals = accum / sum_exp[:, None] # [BH, DP]

    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_vals, mask=mask_h[:, None] & mask_d[None, :])

    l_ptrs = l_base + offs_h * stride_lh
    tl.store(l_ptrs, fin_log, mask=mask_h)

@triton.autotune(
    configs=_sel_attn_bwd_preprocess_configs,
    key=['M', 'D', 'H'],
)
@triton.jit
def _sel_attn_bwd_preprocess_kernel(
    Out,
    DOut,
    Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # program indices
    m  = tl.program_id(0)
    bh = tl.program_id(1)
    b  = bh // H
    h  = bh % H

    # Base pointers
    o_base  = Out  + b * stride_ob  + h * stride_oh
    do_base = DOut + b * stride_dob + h * stride_doh

    # Offsets
    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DP)

    o_ptrs  = o_base  + offs_m[:, None] * stride_om  + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    
    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    
    o  = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    delta = tl.sum(o * do, axis=1)
    
    delta_ptr = Delta + b * stride_db + h * stride_dh + offs_m * stride_dm
    tl.store(delta_ptr, delta, mask=offs_m < M)

@triton.autotune(
    configs=_sel_attn_bwd_configs,
    key=['M', 'N', 'D', 'SELECTION_BLOCK_SIZE', 'T', 'HEADS_PER_GROUP', 'causal'],
    reset_to_zero=['DK', 'DV']
)
@triton.jit
def _sel_attn_bwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Top_idx: tl.tensor,
    Lse: tl.tensor,
    DOut: tl.tensor,
    Delta: tl.tensor,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    DQ: tl.tensor,
    DK: tl.tensor,
    DV: tl.tensor,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kg, stride_kn, stride_kd,
    stride_vb, stride_vg, stride_vn, stride_vd,
    stride_tb, stride_tg, stride_tm, stride_tt, 
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    DP: tl.constexpr,
    SELECTION_BLOCK_SIZE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    OFFSET_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # NOTE: Should we move this outside?
    stride_hg = stride_qh * HEADS_PER_GROUP
    b = tl.program_id(0)
    m = tl.program_id(1) + OFFSET_M
    g = tl.program_id(2)

    # Base pointers
    q_base = Q + b * stride_qb + m * stride_qm + g * stride_hg
    k_base = K + b * stride_kb + g * stride_kg
    v_base = V + b * stride_vb + g * stride_vg
    t_base = Top_idx + b * stride_tb + m * stride_tm + g * stride_tg
    l_base = Lse + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP
    do_base = DOut + b * stride_ob + m * stride_om + g * stride_hg
    d_base = Delta + b * stride_lb + m * stride_lm + g * stride_lh * HEADS_PER_GROUP
    dq_base = DQ + b * stride_qb + m * stride_qm + g * stride_hg
    dk_base = DK + b * stride_kb + g * stride_kg
    dv_base = DV + b * stride_vb + g * stride_vg

    # Offsets
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D
    offs_n = tl.arange(0, SELECTION_BLOCK_SIZE)

    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blck = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32) # [BH, DP]
    
    do_ptrs = do_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    do_blck = tl.load(do_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32) # [BH, DP]
    
    l_ptrs = l_base + offs_h * stride_lh
    l_blck = tl.load(l_ptrs, mask=mask_h, other=0.0) # [BH]
    
    d_ptrs = d_base + offs_h * stride_lh
    d_blck = tl.load(d_ptrs, mask=mask_h, other=0.0) # [BH]

    accum     = tl.zeros([BLOCK_H, DP], dtype=tl.float32)
    # 1/ln(2) = 1.44269504
    log_scale = softmax_scale * 1.44269504

    max_col = max(0, N - M + m) if causal else N

    for idx in range(T):
        # NOTE: Ideally we load top_idx outside the loop, this can be done with a gather which will
        # supported in future versions of Triton
        top = tl.load(t_base + idx * stride_tt)

        col = top * SELECTION_BLOCK_SIZE 
        col = tl.multiple_of(col, SELECTION_BLOCK_SIZE)

        if not causal or col <= max_col:
            cols = col + offs_n
            mask_n = cols < N

            k_ptrs = k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd 
            k_blck = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)  # [DP, BN]

            qk = tl.dot(q_blck, k_blck) * log_scale

            causal_mask = cols <= max_col
            qk = tl.where(causal_mask[None, :], qk, -1e6)

            l2 = l_blck * 1.44269504
            exp_qk = tl.math.exp2(qk - l2[:, None]) # [BH, BN]
            
            dv_inc = tl.dot(tl.trans(exp_qk), do_blck) # [BN, DP]

            dv_ptrs = dv_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
            # [BN, DP]
            tl.atomic_add(dv_ptrs, dv_inc.to(tl.float32), mask=mask_d[None, :] & mask_n[:, None], sem="release", scope="gpu")
            
            v_ptrs = v_base + cols[None, :] * stride_vn + offs_d[:, None] * stride_vd
            v_blck = tl.load(v_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32) # [D, BN]
            dp = tl.dot(do_blck, v_blck)  # [BH, BN]
            ds2 = exp_qk * (dp - d_blck[:, None])     # [BH, BN]
            ds  = ds2 * softmax_scale

            accum   = tl.dot(ds, tl.trans(k_blck), acc=accum)  # [BH, DP]

            dk_inc = tl.dot(tl.trans(ds), q_blck) # [BN, DP]

            dk_ptrs = dk_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
            tl.atomic_add(dk_ptrs, dk_inc.to(tl.float32), mask=mask_d[None, :] & mask_n[:, None], sem="release", scope="gpu")

    dq_ptrs = dq_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, accum, mask=mask_h[:, None] & mask_d[None, :])
    
class SelectionAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v, top_idx,
        selection_block_size, 
        softmax_scale=None, 
        causal=False, 
        return_attn_probs=False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        B, M, H, D = q.shape
        _, N, G, _ = k.shape
        _, _, _, T = top_idx.shape

        assert q.shape == (B, M, H, D)
        assert k.shape == (B, N, G, D)
        assert v.shape == (B, N, G, D)
        assert top_idx.shape == (B, M, G, T)


        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)
        
        # NOTE: Is it faster to only set the untouched elements?
        out = torch.zeros_like(q)
        lse = torch.full((B, H, M), float('-inf'), device=q.device, dtype=torch.float32)

        DP = triton.next_power_of_2(D)
        HEADS_PER_GROUP = H // G
        OFFSET_M = max(0, M - N) if causal else 0
        BLOCK_H = max(16, HEADS_PER_GROUP)

        grid = (B, M - OFFSET_M, G)

        _sel_attn_fwd_kernel[grid](
            q, k, v, top_idx,
            softmax_scale, causal,
            out, lse,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            top_idx.stride(0), top_idx.stride(2), top_idx.stride(1), top_idx.stride(3),
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, M, N, D, T, DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            OFFSET_M=OFFSET_M,
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(q, k, v, top_idx, out, lse)
        ctx.selection_block_size = selection_block_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal

        if return_attn_probs:
            return out, lse
        else:
            return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_out = grad_outputs[0]
        
        q, k, v, top_idx, out, lse = ctx.saved_tensors
        B, M, H, D = q.shape
        _, N, G, _ = k.shape
        _, _, _, T = top_idx.shape

        assert d_out.shape == (B, M, H, D)

        selection_block_size = ctx.selection_block_size
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal

        delta = torch.empty_like(lse)

        DP = triton.next_power_of_2(D)
        HEADS_PER_GROUP = H // G
        OFFSET_M = max(0, M - N) if causal else 0
        BLOCK_H = max(16, HEADS_PER_GROUP)

        def grid_preprocess(META):
            return (triton.cdiv(M, META['BLOCK_M']), B * H)

        _sel_attn_bwd_preprocess_kernel[grid_preprocess](
            out, d_out, delta,
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            d_out.stride(0), d_out.stride(2), d_out.stride(1), d_out.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, H, M, D, DP,
        )

        dq = torch.empty_like(q, dtype=q.dtype)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        grid_bwd = (B, M - OFFSET_M, G)

        _sel_attn_bwd_kernel[grid_bwd](
            q, k, v, top_idx, lse,
            d_out, delta,
            softmax_scale, causal,
            dq, dk, dv,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            top_idx.stride(0), top_idx.stride(2), top_idx.stride(1), top_idx.stride(3),
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, M, N, D, T, DP,
            SELECTION_BLOCK_SIZE=selection_block_size,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            OFFSET_M=OFFSET_M,
            BLOCK_H=BLOCK_H,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None

def selection_attention(
    q, k, v, 
    block_indices, block_count, block_size, scale,
    variant='two-pass', # 'one-pass' or 'two-pass'
    causal=True,
    return_attn_probs=False
):
    if variant == 'one-pass':
        return SelectionAttention.apply(
            q, k, v, block_indices, block_size, scale, causal, return_attn_probs
        )
    elif variant == 'two-pass':
        # FLA Backend for two-pass selection attention
        return ParallelNSAFunction.apply(
            q, k, v, block_indices, block_count, block_size, scale, None
        )
    else:
        raise ValueError(f"Invalid variant: {variant}")

