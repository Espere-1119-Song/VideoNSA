# 修复版本：解决DeepSpeed和device_map冲突
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 
export VIDEO_MAX_PIXELS=50176
export IMAGE_MAX_PIXELS=50176
export MAX_PIXELS=50176


swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type doneright \
    --train_type full \
    --deepspeed zero2 \
    --freeze_llm false \
    --freeze_aligner false \
    --freeze_vit false \
    --attn_impl flash_attn \
    --use_hf true \
    --dataset 'datasets/jsonl/test.jsonl' \
    --train_dataloader_shuffle false \
    --dataset_shuffle false \
    --data_seed 42 \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0.001 \
    --save_strategy steps \
    --save_steps 10 \
    --torch_dtype bfloat16 \
    --max_steps 52500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 6e-04 \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 65536 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --loss_scale default \
    --load_from_cache_file true \
    --save_safetensors true \
    --report_to tensorboard \
    --logging_dir output/h100_sparse \
    --output_dir output/h100 \
    --strict true