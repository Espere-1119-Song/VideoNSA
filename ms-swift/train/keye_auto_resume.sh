#!/bin/bash

# 修复版本：解决DeepSpeed和device_map冲突 + 自动检测最新checkpoint
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

# 输出目录配置
OUTPUT_DIR="output/nsa_64_32_256"

# 函数：自动检测最新的checkpoint
find_latest_checkpoint() {
    local output_dir=$1
    
    # 检查输出目录是否存在
    if [[ ! -d "$output_dir" ]]; then
        echo "输出目录 $output_dir 不存在，将从头开始训练" >&2
        return 1
    fi
    
    # 查找最新的版本目录 (按时间戳排序)
    local latest_version_dir=$(find "$output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -n 1)
    
    if [[ -z "$latest_version_dir" ]]; then
        echo "未找到任何版本目录，将从头开始训练" >&2
        return 1
    fi
    
    echo "找到最新版本目录: $latest_version_dir" >&2
    
    # 在版本目录中查找最新的checkpoint
    local latest_checkpoint=$(find "$latest_version_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [[ -z "$latest_checkpoint" ]]; then
        echo "在 $latest_version_dir 中未找到任何checkpoint，将从头开始训练" >&2
        return 1
    fi
    
    echo "找到最新checkpoint: $latest_checkpoint" >&2
    
    # 验证checkpoint目录的有效性
    if [[ -f "$latest_checkpoint/trainer_state.json" ]] && [[ -f "$latest_checkpoint/training_args.bin" ]]; then
        echo "✓ Checkpoint验证通过: $latest_checkpoint" >&2
        echo "$latest_checkpoint"
        return 0
    else
        echo "✗ Checkpoint验证失败，缺少必要文件，将从头开始训练" >&2
        return 1
    fi
}

# 检测最新checkpoint
echo "=== 开始检测最新checkpoint ==="
LATEST_CHECKPOINT=$(find_latest_checkpoint "$OUTPUT_DIR")
CHECKPOINT_STATUS=$?

# 构建训练命令
SWIFT_CMD="swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type full \
    --deepspeed zero1 \
    --attn_impl flash_attn \
    --sequence_parallel_size 8 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --use_hf true \
    --dataset 'datasets/jsonl/filter_llava_350_500#10000' \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0.001 \
    --save_strategy steps \
    --save_steps 100 \
    --torch_dtype bfloat16 \
    --max_steps 2500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 65536 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing true \
    --max_grad_norm 0.5 \
    --loss_scale default \
    --load_from_cache_file true \
    --save_safetensors true \
    --report_to tensorboard \
    --logging_dir output/sparse \
    --output_dir $OUTPUT_DIR"

# 如果找到了有效的checkpoint，添加resume参数
if [[ $CHECKPOINT_STATUS -eq 0 ]]; then
    echo "=== 将从checkpoint恢复训练: $LATEST_CHECKPOINT ==="
    SWIFT_CMD="$SWIFT_CMD --resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "=== 将从头开始训练 ==="
fi

echo ""
echo "=== 执行训练命令 ==="
echo "$SWIFT_CMD"
echo ""

# 执行训练
eval $SWIFT_CMD