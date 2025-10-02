mkdir -p clock_results_win_512
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=4 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_1 \
    2>&1 | tee clock_results_win_512/nsa_1.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=8 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_2 \
    2>&1 | tee clock_results_win_512/nsa_2.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=16 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_4 \
    2>&1 | tee clock_results_win_512/nsa_4.log


accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=32 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_8 \
    2>&1 | tee clock_results_win_512/nsa_8.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=64 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_16 \
    2>&1 | tee clock_results_win_512/nsa_16.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=128 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_32 \
    2>&1 | tee clock_results_win_512/nsa_32.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=256 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_64 \
    2>&1 | tee clock_results_win_512/nsa_64.log

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model doneright \
    --model_args=pretrained=ckpt/checkpoint-42600,max_pixels=200704,min_pixels=200704,interleave_visuals=False,fps=4,block_counts=32,window_size=256,attn_implementation=flash_attention_2,max_num_frames=512 \
    --tasks mlvu_test \
    --verbosity=DEBUG \
    --batch_size 1 \
    --log_samples \
    --limit 8 \
    --output_path clock_results_win_512/nsa_128 \
    2>&1 | tee clock_results_win_512/nsa_128.log