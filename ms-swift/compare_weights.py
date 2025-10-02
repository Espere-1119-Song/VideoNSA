#!/usr/bin/env python3
"""
Script to compare weights across different checkpoints to see if they're identical
"""

import safetensors
from safetensors.torch import load_file
import torch
import os

# Base path and checkpoints to compare
base_path = "output/final_gate/v168-20250827-085616"
checkpoints = ["checkpoint-200", "checkpoint-400", "checkpoint-600", "checkpoint-800"]

# Weight key to compare
weight_key = "model.layers.0.self_attn.g_proj_2.weight"

# Store weights from each checkpoint
weights = {}

print(f"Comparing weight '{weight_key}' across checkpoints...")
print("=" * 60)

for checkpoint in checkpoints:
    safetensors_path = os.path.join(base_path, checkpoint, "model-00001-of-00004.safetensors")
    
    if os.path.exists(safetensors_path):
        print(f"\nLoading {checkpoint}...")
        tensors = load_file(safetensors_path)
        
        if weight_key in tensors:
            weight_tensor = tensors[weight_key]
            weights[checkpoint] = weight_tensor
            
            # Print basic statistics
            print(f"  Shape: {weight_tensor.shape}")
            print(f"  Min: {weight_tensor.min().item():.6f}")
            print(f"  Max: {weight_tensor.max().item():.6f}")
            print(f"  Mean: {weight_tensor.mean().item():.6f}")
            print(f"  Std: {weight_tensor.std().item():.6f}")
        else:
            print(f"  Weight key '{weight_key}' not found!")
    else:
        print(f"  File not found: {safetensors_path}")

# Compare weights between checkpoints
print("\n" + "=" * 60)
print("COMPARISON RESULTS:")
print("=" * 60)

checkpoint_names = list(weights.keys())
for i in range(len(checkpoint_names)):
    for j in range(i+1, len(checkpoint_names)):
        cp1 = checkpoint_names[i]
        cp2 = checkpoint_names[j]
        
        if cp1 in weights and cp2 in weights:
            w1 = weights[cp1]
            w2 = weights[cp2]
            
            # Check if tensors are identical
            are_identical = torch.equal(w1, w2)
            
            # Calculate differences
            diff = torch.abs(w1 - w2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n{cp1} vs {cp2}:")
            print(f"  Identical: {are_identical}")
            print(f"  Max absolute difference: {max_diff:.10f}")
            print(f"  Mean absolute difference: {mean_diff:.10f}")
            
            if not are_identical:
                # Show some statistics about differences
                nonzero_diff = diff[diff > 0]
                if len(nonzero_diff) > 0:
                    print(f"  Number of different elements: {len(nonzero_diff)}")
                    print(f"  Percentage of different elements: {len(nonzero_diff)/w1.numel()*100:.2f}%")

# If all weights are identical, this indicates a potential problem
all_identical = True
if len(checkpoint_names) > 1:
    base_weight = weights[checkpoint_names[0]]
    for cp in checkpoint_names[1:]:
        if not torch.equal(base_weight, weights[cp]):
            all_identical = False
            break

print("\n" + "=" * 60)
if all_identical and len(checkpoint_names) > 1:
    print("⚠️  WARNING: All weights are IDENTICAL across different checkpoints!")
    print("   This suggests the model is not learning or there's a training issue.")
else:
    print("✅ Weights are different across checkpoints (this is expected).")
print("=" * 60)