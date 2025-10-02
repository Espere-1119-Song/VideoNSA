export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Attention longtimescope

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=4,window_size=1900,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/4_1900

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=34,window_size=128,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/34_128

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=35,window_size=64,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/35_64

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=36,window_size=32,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/36_32

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=20,window_size=1024,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/20_1024

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=100352,min_pixels=100352,interleave_visuals=False,fps=4,block_counts=28,window_size=512,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks longtimescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_longtimescope/28_512