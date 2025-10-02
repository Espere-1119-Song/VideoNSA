# NSA Attention Weights 功能说明

## 概述

NSA (Neural Sparse Attention) 现在支持返回attention weights，这对于模型分析、可视化和调试非常有用。本文档详细说明了如何使用这个功能。

## 主要功能

### 1. 基本用法

```python
import torch
from nsa import nsa_func

# 创建输入tensors
q = torch.randn(2, 128, 8, 64, device='cuda')  # [B, M, H, D]
k = torch.randn(2, 128, 8, 64, device='cuda')  # [B, N, G, D]
v = torch.randn(2, 128, 8, 64, device='cuda')  # [B, N, G, D]

# 创建gating weights
g_cmp = torch.softmax(torch.randn(2, 128, 8, device='cuda'), dim=-1)
g_slc = torch.softmax(torch.randn(2, 128, 8, device='cuda'), dim=-1)
g_swa = torch.softmax(torch.randn(2, 128, 8, device='cuda'), dim=-1)

# 不返回attention weights (默认行为)
output = nsa_func(q, k, v, g_cmp, g_slc, g_swa)

# 返回attention weights
output, attn_weights = nsa_func(
    q, k, v, g_cmp, g_slc, g_swa,
    return_attn_weights=True
)
```

### 2. Attention Weights 结构

当 `return_attn_weights=True` 时，函数返回一个包含以下结构的字典：

```python
attn_weights = {
    'compression': {
        'lse': torch.Tensor,           # [B, H, M] - log-sum-exp values
        'block_indices': torch.Tensor, # [B, M, G, T] - selected block indices
    },
    'selection': {
        'lse': torch.Tensor,           # [B, H, M] - log-sum-exp values  
        'block_indices': torch.Tensor, # [B, M, G, T] - selected block indices
    },
    'sliding_window': {
        'note': str,                   # 说明信息
    },
    'gating_weights': {
        'g_cmp': torch.Tensor,         # [B, M, H] - compression gating weights
        'g_slc': torch.Tensor,         # [B, M, H] - selection gating weights
        'g_swa': torch.Tensor,         # [B, M, H] - sliding window gating weights
    }
}
```

### 3. 从LSE计算Attention Weights

由于NSA使用LSE (log-sum-exp) 来优化内存使用，我们提供了辅助函数来从LSE计算完整的attention weights：

```python
from nsa import compute_attention_weights_from_lse

# 计算compression attention的完整attention weights
lse_cmp = attn_weights['compression']['lse']
attn_weights_cmp = compute_attention_weights_from_lse(
    lse=lse_cmp, q=q, k=k, block_size=64
)
# 返回: [B, H, M, N] - 完整的attention weights矩阵

# 计算selection attention的完整attention weights
lse_slc = attn_weights['selection']['lse']
block_indices = attn_weights['selection']['block_indices']
attn_weights_slc = compute_attention_weights_from_lse(
    lse=lse_slc, q=q, k=k, 
    block_indices=block_indices, block_size=64
)
# 返回: [B, H, M, N] - 稀疏的attention weights矩阵
```

### 4. 获取统计摘要

```python
from nsa import get_attention_weights_summary

summary = get_attention_weights_summary(attn_weights)
print(summary)
# 输出包含各种统计信息的字典，如均值、标准差、最大值、最小值等
```

## 使用场景

### 1. 模型分析

```python
# 分析不同组件的attention patterns
def analyze_attention_patterns(attn_weights):
    # 分析compression attention
    lse_cmp = attn_weights['compression']['lse']
    print(f"Compression LSE range: {lse_cmp.min():.3f} to {lse_cmp.max():.3f}")
    
    # 分析gating weights分布
    g_cmp = attn_weights['gating_weights']['g_cmp']
    g_slc = attn_weights['gating_weights']['g_slc']
    g_swa = attn_weights['gating_weights']['g_swa']
    
    print(f"Compression gating weight: {g_cmp.mean():.3f}")
    print(f"Selection gating weight: {g_slc.mean():.3f}")
    print(f"Sliding window gating weight: {g_swa.mean():.3f}")
```

### 2. 可视化

```python
import matplotlib.pyplot as plt

def visualize_attention_weights(attn_weights, q, k, head_idx=0, batch_idx=0):
    # 计算完整的attention weights
    lse_cmp = attn_weights['compression']['lse']
    attn_weights_cmp = compute_attention_weights_from_lse(
        lse=lse_cmp, q=q, k=k, block_size=64
    )
    
    # 可视化
    weights = attn_weights_cmp[batch_idx, head_idx].detach().cpu()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.title(f'Attention Weights (Head {head_idx})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    plt.show()
```

### 3. 调试和验证

```python
def validate_attention_weights(attn_weights, q, k, v):
    """验证attention weights的正确性"""
    
    # 计算compression attention weights
    lse_cmp = attn_weights['compression']['lse']
    attn_weights_cmp = compute_attention_weights_from_lse(
        lse=lse_cmp, q=q, k=k, block_size=64
    )
    
    # 验证概率和为1
    prob_sums = attn_weights_cmp.sum(dim=-1)  # [B, H, M]
    print(f"Probability sums (should be close to 1): {prob_sums.mean():.4f}")
    
    # 验证非负性
    is_non_negative = (attn_weights_cmp >= 0).all()
    print(f"All weights non-negative: {is_non_negative}")
```

## 性能考虑

### 1. 内存使用

- **不返回attention weights**: 内存使用最小，适合生产环境
- **返回attention weights**: 会增加内存使用，主要用于分析和调试

### 2. 计算开销

```python
import time

# 性能测试
def benchmark_performance():
    # 设置参数
    B, M, H, D = 1, 256, 16, 64
    N, G = 256, 16
    
    q = torch.randn(B, M, H, D, device='cuda')
    k = torch.randn(B, N, G, D, device='cuda')
    v = torch.randn(B, N, G, D, device='cuda')
    
    g_cmp = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_slc = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    g_swa = torch.softmax(torch.randn(B, M, H, device='cuda'), dim=-1)
    
    # 测试不返回attention weights
    start = time.time()
    for _ in range(100):
        output = nsa_func(q, k, v, g_cmp, g_slc, g_swa)
    torch.cuda.synchronize()
    time_without = time.time() - start
    
    # 测试返回attention weights
    start = time.time()
    for _ in range(100):
        output, attn_weights = nsa_func(
            q, k, v, g_cmp, g_slc, g_swa, return_attn_weights=True
        )
    torch.cuda.synchronize()
    time_with = time.time() - start
    
    print(f"Without attention weights: {time_without:.4f}s")
    print(f"With attention weights: {time_with:.4f}s")
    print(f"Overhead: {(time_with - time_without) / time_without * 100:.2f}%")
```

## 注意事项

1. **内存使用**: 返回attention weights会增加内存使用，特别是在长序列上
2. **计算开销**: 计算完整的attention weights矩阵需要额外的计算
3. **精度**: LSE到attention weights的转换可能引入数值精度问题
4. **稀疏性**: Selection attention的attention weights是稀疏的，大部分位置为0

## 示例代码

完整的示例代码请参考 `example_usage.py` 文件，其中包含了：
- 基本使用示例
- 可视化示例
- 性能对比示例
- 统计分析示例 