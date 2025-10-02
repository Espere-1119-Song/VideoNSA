export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=401408,min_pixels=401408,fps=4,interleave_visuals=False,max_num_frames=128,attn_implementation=flash_attention_2 \
    --tasks longvideobench_val_v \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/token_longvideobench_val_v/512_128

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=401408,min_pixels=401408,fps=4,interleave_visuals=False,max_num_frames=128,attn_implementation=flash_attention_2 \
    --tasks longvideobench_val_v \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/token_longvideobench_val_v/512_256
