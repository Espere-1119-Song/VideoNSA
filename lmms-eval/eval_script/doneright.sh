export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-8600,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500,attn_implementation=flash_attention_2 \
    --tasks vsibench \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path nsa_results/vsibench/1fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=4,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks longvideobench_val_v \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/longvideobench/4fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=8,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks longvideobench_val_v \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/longvideobench/8fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks vsibench \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/vsibench/1fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=2,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks vsibench \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/vsibench/2fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=4,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks vsibench \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/vsibench/4fps

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model doneright \
#     --model_args=pretrained=ckpt/nsa_64_32_256/v11-20250808-062621/checkpoint-2500,max_pixels=50176,interleave_visuals=False,fps=8,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks vsibench \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path nsa_results/vsibench/8fps