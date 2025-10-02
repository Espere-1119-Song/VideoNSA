#!/bin/bash
mkdir -p clock_results

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen25_vl_xattention \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=4 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results/xatt_1 \
    2>&1 | tee clock_results/xatt_1.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=8 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_2 \
#     2>&1 | tee clock_results/flash_2.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=16 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_4 \
#     2>&1 | tee clock_results/flash_4.log


# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=32 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_8 \
#     2>&1 | tee clock_results/flash_8.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=64 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_16 \
#     2>&1 | tee clock_results/flash_16.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=128 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_32 \
#     2>&1 | tee clock_results/flash_32.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=256 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_64 \
#     2>&1 | tee clock_results/flash_64.log

# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,attn_implementation=flash_attention_2,max_num_frames=512 \
#     --tasks mlvu_test \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --limit 8 \
#     --output_path clock_results/flash_128 \
#     2>&1 | tee clock_results/flash_128.log