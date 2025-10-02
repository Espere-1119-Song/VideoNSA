#!/bin/bash
mkdir -p clock_results
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=128 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results/32_256_flash \
    2>&1 | tee clock_results/32_256_flash_runtime.log