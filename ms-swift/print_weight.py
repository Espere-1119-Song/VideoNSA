#!/usr/bin/env python3
"""
Script to print the values of model.layers.0.self_attn.g_proj_2.weight from safetensors file
"""

import safetensors
from safetensors.torch import load_file
import torch

# Path to the safetensors file
safetensors_path = "output/final_gate/v168-20250827-085616/checkpoint-800/model-00001-of-00004.safetensors"

# Weight key to extract
weight_key = "model.layers.0.self_attn.g_proj_2.weight"


# Load the safetensors file
print(f"Loading safetensors file: {safetensors_path}")
tensors = load_file(safetensors_path)

# Check if the weight exists
if weight_key in tensors:
    weight_tensor = tensors[weight_key]
    import pdb; pdb.set_trace()

    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Min: {weight_tensor.min().item()}")
    print(f"Max: {weight_tensor.max().item()}")
    print(f"Mean: {weight_tensor.mean().item()}")
    print(f"Std: {weight_tensor.std().item()}")
else:
    print(f"Weight key '{weight_key}' not found in the safetensors file.")
    print("Available keys:")
    for key in tensors.keys():
        print(f"  - {key}")

