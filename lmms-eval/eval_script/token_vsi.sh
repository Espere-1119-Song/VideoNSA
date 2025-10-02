export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=16,window_size=128,attn_implementation=flash_attention_2,max_num_frames=128 \
    --tasks tomato \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/attention_vsibench/16_128
