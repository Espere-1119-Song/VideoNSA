#!/usr/bin/env python3
"""
脚本功能：对比原始LLaVA数据集和过滤后的数据集，提取缺失的数据
用法：python extract_missing_data.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        # 先获取文件行数用于进度条
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc=f"加载 {os.path.basename(file_path)}", unit="行"):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"警告: 文件未找到 {file_path}")
        return []
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        return []
    return data

def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"保存 {os.path.basename(file_path)}", unit="条"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条记录到 {file_path}")

def create_data_id(item):
    """创建数据项的唯一标识ID"""
    # 使用video和conversations的组合创建唯一标识
    if 'video' in item and 'conversations' in item:
        video_info = item['video'] if isinstance(item['video'], str) else str(item['video'])
        # 取conversations的前100字符作为标识的一部分
        conv_text = str(item['conversations'])[:100] if 'conversations' in item else ""
        return f"{video_info}_{hash(conv_text)}"
    return str(hash(str(item)))

def analyze_missing_data():
    """分析并提取缺失的数据"""
    # 定义路径
    original_dir = "/lambda/nfs/jianwen-us-south-2/tulab/enxin/projects/ms-swift/datasets/jsonl/LLaVA-Video-117K"
    filtered_dir = "/lambda/nfs/jianwen-us-south-2/tulab/enxin/projects/ms-swift/datasets/jsonl/filter_llava_350_550"
    missing_dir = "/lambda/nfs/jianwen-us-south-2/tulab/enxin/projects/ms-swift/datasets/jsonl/missing_data_350_550"
    
    # 需要处理的文件列表（只处理过滤目录中存在的文件）
    filtered_files = [
        "1_2_m_academic_v0_1.jsonl",
        "1_2_m_activitynetqa.jsonl", 
        "1_2_m_nextqa.jsonl",
        "1_2_m_youtube_v0_1.jsonl",
        "2_3_m_academic_v0_1.jsonl",
        "2_3_m_activitynetqa.jsonl",
        "2_3_m_nextqa.jsonl",
        "2_3_m_youtube_v0_1.jsonl"
    ]
    
    # 统计信息
    total_original = 0
    total_filtered = 0 
    total_missing = 0
    
    print("开始分析数据缺失情况...")
    print("=" * 80)
    
    for filename in tqdm(filtered_files, desc="处理文件", unit="文件"):
        print(f"\n处理文件: {filename}")
        print("-" * 50)
        
        # 加载原始数据和过滤数据
        original_path = os.path.join(original_dir, filename)
        filtered_path = os.path.join(filtered_dir, filename)
        
        original_data = load_jsonl(original_path)
        filtered_data = load_jsonl(filtered_path)
        
        if not original_data:
            print(f"跳过 {filename}: 原始文件为空或不存在")
            continue
            
        print(f"原始数据: {len(original_data)} 条")
        print(f"过滤数据: {len(filtered_data)} 条")
        
        # 创建过滤数据的ID集合用于快速查找
        print("创建过滤数据ID索引...")
        filtered_ids = set()
        for item in tqdm(filtered_data, desc="索引过滤数据", unit="条"):
            filtered_ids.add(create_data_id(item))
        
        # 找出缺失的数据
        print("查找缺失数据...")
        missing_data = []
        for item in tqdm(original_data, desc="对比数据", unit="条"):
            item_id = create_data_id(item)
            if item_id not in filtered_ids:
                missing_data.append(item)
        
        print(f"缺失数据: {len(missing_data)} 条")
        
        # 保存缺失的数据
        if missing_data:
            missing_path = os.path.join(missing_dir, filename)
            save_jsonl(missing_data, missing_path)
        
        # 更新统计
        total_original += len(original_data)
        total_filtered += len(filtered_data)
        total_missing += len(missing_data)
    
    # 打印总体统计
    print("\n" + "=" * 80)
    print("总体统计:")
    print(f"原始总数据量: {total_original:,} 条")
    print(f"过滤后数据量: {total_filtered:,} 条") 
    print(f"缺失数据量: {total_missing:,} 条")
    print(f"数据保留率: {total_filtered/total_original*100:.2f}%")
    print(f"数据缺失率: {total_missing/total_original*100:.2f}%")
    
    # 创建汇总的缺失数据文件
    print(f"\n合并所有缺失数据到: {missing_dir}/all_missing_data.jsonl")
    all_missing = []
    for filename in filtered_files:
        missing_path = os.path.join(missing_dir, filename)
        if os.path.exists(missing_path):
            missing_data = load_jsonl(missing_path)
            all_missing.extend(missing_data)
    
    if all_missing:
        all_missing_path = os.path.join(missing_dir, "all_missing_data.jsonl")
        save_jsonl(all_missing, all_missing_path)
    
    print(f"\n✅ 分析完成! 缺失数据已保存到: {missing_dir}/")

if __name__ == "__main__":
    analyze_missing_data()