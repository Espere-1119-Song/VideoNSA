export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen25_vl_minference \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=$MAX_PIXELS,interleave_visuals=False,fps=$FPS,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=flash_attention_2 \
    --tasks $TASK_NAME \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path results/

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen25_vl_xattention \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=$MAX_PIXELS,interleave_visuals=False,fps=$FPS,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=flash_attention_2 \
    --tasks $TASK_NAME \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path results/

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen25_vl_flexprefill \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=$MAX_PIXELS,interleave_visuals=False,fps=$FPS,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=flash_attention_2 \
    --tasks $TASK_NAME \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path results/

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen25_vl_trishape \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=$MAX_PIXELS,interleave_visuals=False,fps=$FPS,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=flash_attention_2 \
    --tasks $TASK_NAME \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --output_path results/