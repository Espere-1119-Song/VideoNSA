# 修复版本：解决DeepSpeed和device_map冲突
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 
export VIDEO_MAX_PIXELS=50176
export IMAGE_MAX_PIXELS=50176
export MAX_PIXELS=50176

# 明确设置分布式训练的设备
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0

export SEQUENCE_PARALLEL_IMPL=ulysses
# 根据序列长度调整CELOSS_PARALLEL_SIZE，65536/4=16384
export CELOSS_PARALLEL_SIZE=16384 

swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type doneright \
    --train_type full \
    --deepspeed zero2 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --attn_impl flash_attn \
    --sequence_parallel_size 8 \
    --use_hf true \
    --dataset 'datasets/jsonl/filter_llava_350_550' \
    --train_dataloader_shuffle false \
    --dataset_shuffle false \
    --data_seed 42 \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0.001 \
    --save_strategy steps \
    --save_steps 200 \
    --torch_dtype bfloat16 \
    --max_steps 52500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-06 \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 65536 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 1 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --loss_scale default \
    --load_from_cache_file true \
    --save_safetensors true \
    --report_to tensorboard \
    --logging_dir output/final_gate \
    --output_dir output/final_gate_log \
    --strict true \
    --resume_only_model true \
    --resume_from_checkpoint output/final_gate_log/v3-20250830-060640/checkpoint-9800