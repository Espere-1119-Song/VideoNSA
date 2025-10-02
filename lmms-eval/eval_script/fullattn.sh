export HF_HOME="/home/ubuntu/.cache/huggingface"

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500,attn_implementation=flash_attention_2 \
#     --tasks timescope \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --log_samples \
#     --output_path results/full_flash

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500 \
    --tasks timescope \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path results/full

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=768 \
#     --tasks mlvu_dev \
#     --batch_size 1