import torch
from nsa import nsa_func, compute_attention_weights_from_lse, get_attention_weights_summary

def example_nsa_with_attention_weights():
    """
    示例：使用NSA函数并获取attention weights
    """
    # 设置参数
    B, M, H, D = 2, 128, 8, 64  # batch_size, seq_len, num_heads, head_dim
    N, G = 128, 8  # key_seq_len, num_groups
    
    # 创建输入tensors
    q = torch.randn(B, M, H, D, device='cuda')
    k = torch.randn(B, N, G, D, device='cuda')
    v = torch.randn(B, N, G, D, device='cuda')
    
    # 创建gating weights
    g_cmp = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_slc = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_swa = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    
    # 调用NSA函数，获取attention weights
    output, attn_weights = nsa_func(
        q=q, k=k, v=v,
        g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
        block_count=16, block_size=64, window_size=128,
        return_attn_weights=True
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights keys: {attn_weights.keys()}")
    
    # 获取attention weights摘要
    summary = get_attention_weights_summary(attn_weights)
    print("\nAttention Weights Summary:")
    for component, stats in summary.items():
        print(f"\n{component.upper()}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # 计算compression attention的完整attention weights
    lse_cmp = attn_weights['compression']['lse']
    attn_weights_cmp = compute_attention_weights_from_lse(
        lse=lse_cmp, q=q, k=k, block_size=64
    )
    print(f"\nCompression attention weights shape: {attn_weights_cmp.shape}")
    print(f"Compression attention weights sum per head: {attn_weights_cmp.sum(dim=-1).mean():.4f}")
    
    # 计算selection attention的完整attention weights
    lse_slc = attn_weights['selection']['lse']
    block_indices = attn_weights['selection']['block_indices']
    attn_weights_slc = compute_attention_weights_from_lse(
        lse=lse_slc, q=q, k=k, 
        block_indices=block_indices, block_size=64
    )
    print(f"Selection attention weights shape: {attn_weights_slc.shape}")
    print(f"Selection attention weights sum per head: {attn_weights_slc.sum(dim=-1).mean():.4f}")
    
    # 可视化某个head的attention weights
    import matplotlib.pyplot as plt
    
    # 选择第一个batch，第一个head的attention weights
    head_idx = 0
    batch_idx = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compression attention weights
    cmp_weights = attn_weights_cmp[batch_idx, head_idx].detach().cpu()
    im1 = axes[0].imshow(cmp_weights, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Compression Attention (Head {head_idx})')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Selection attention weights
    slc_weights = attn_weights_slc[batch_idx, head_idx].detach().cpu()
    im2 = axes[1].imshow(slc_weights, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Selection Attention (Head {head_idx})')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('nsa_attention_weights.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return output, attn_weights

def example_compare_with_and_without_weights():
    """
    示例：比较有和没有attention weights的性能差异
    """
    import time
    
    # 设置参数
    B, M, H, D = 1, 256, 16, 64
    N, G = 256, 16
    
    # 创建输入tensors
    q = torch.randn(B, M, H, D, device='cuda')
    k = torch.randn(B, N, G, D, device='cuda')
    v = torch.randn(B, N, G, D, device='cuda')
    
    g_cmp = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_slc = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_swa = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    
    # 预热
    for _ in range(10):
        _ = nsa_func(q, k, v, g_cmp, g_slc, g_swa)
        _ = nsa_func(q, k, v, g_cmp, g_slc, g_swa, return_attn_weights=True)
    
    torch.cuda.synchronize()
    
    # 测试不返回attention weights的性能
    start_time = time.time()
    for _ in range(100):
        output = nsa_func(q, k, v, g_cmp, g_slc, g_swa)
    torch.cuda.synchronize()
    time_without_weights = time.time() - start_time
    
    # 测试返回attention weights的性能
    start_time = time.time()
    for _ in range(100):
        output, attn_weights = nsa_func(q, k, v, g_cmp, g_slc, g_swa, return_attn_weights=True)
    torch.cuda.synchronize()
    time_with_weights = time.time() - start_time
    
    print(f"Time without attention weights: {time_without_weights:.4f}s")
    print(f"Time with attention weights: {time_with_weights:.4f}s")
    print(f"Overhead: {(time_with_weights - time_without_weights) / time_without_weights * 100:.2f}%")

if __name__ == "__main__":
    print("Running NSA with attention weights example...")
    example_nsa_with_attention_weights()
    
    print("\n" + "="*50 + "\n")
    
    print("Running performance comparison...")
    example_compare_with_and_without_weights() 