accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=checkpoint-3438,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500,attn_implementation=flash_attention_2 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --output_path sft_results

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=50176,interleave_visuals=False,fps=1,max_num_frames=500 \
#     --tasks mlvu_test,longvideobench_val_v \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path full_results