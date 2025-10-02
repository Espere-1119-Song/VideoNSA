#!/usr/bin/env python3
"""
脚本功能：检查JSONL文件中引用的图片文件是否存在，输出缺失的图片列表
用法：python check_missing_images.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"加载 {os.path.basename(file_path)}", unit="行"):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        return []
    return data

def check_missing_images():
    """检查缺失的图片文件"""
    # 定义路径
    base_dir = "/home/ubuntu/jianwen-us-south-2/tulab/enxin/projects/ms-swift"
    jsonl_dir = os.path.join(base_dir, "datasets/jsonl/missing_data_350_550")
    
    # 需要检查的文件列表
    jsonl_files = [
        "1_2_m_academic_v0_1.jsonl",
        "1_2_m_activitynetqa.jsonl", 
        "1_2_m_nextqa.jsonl",
        "1_2_m_youtube_v0_1.jsonl",
        "2_3_m_academic_v0_1.jsonl",
        "2_3_m_activitynetqa.jsonl",
        "2_3_m_nextqa.jsonl",
        "2_3_m_youtube_v0_1.jsonl"
    ]
    
    all_missing_images = set()
    total_images = 0
    
    print("开始检查图片文件...")
    print("=" * 80)
    
    for filename in jsonl_files:
        file_path = os.path.join(jsonl_dir, filename)
        if not os.path.exists(file_path):
            print(f"跳过 {filename}: 文件不存在")
            continue
            
        print(f"\n检查文件: {filename}")
        print("-" * 50)
        
        data = load_jsonl(file_path)
        if not data:
            print(f"跳过 {filename}: 文件为空")
            continue
        
        missing_in_file = set()
        total_in_file = 0
        
        for item in tqdm(data, desc="检查图片", unit="条"):
            if 'images' in item and item['images']:
                for image_path in item['images']:
                    total_images += 1
                    total_in_file += 1
                    
                    # 转换为绝对路径
                    if image_path.startswith('datasets/'):
                        full_path = os.path.join(base_dir, image_path)
                    else:
                        full_path = image_path
                    
                    # 检查文件是否存在
                    if not os.path.exists(full_path):
                        missing_in_file.add(image_path)
                        all_missing_images.add(image_path)
        
        print(f"文件中图片总数: {total_in_file}")
        print(f"缺失图片数量: {len(missing_in_file)}")
        if missing_in_file:
            print(f"缺失率: {len(missing_in_file)/total_in_file*100:.2f}%")
    
    # 输出总体统计
    print("\n" + "=" * 80)
    print("总体统计:")
    print(f"总图片数量: {total_images:,}")
    print(f"缺失图片数量: {len(all_missing_images):,}")
    if total_images > 0:
        print(f"缺失率: {len(all_missing_images)/total_images*100:.2f}%")
    
    # 保存缺失图片列表
    if all_missing_images:
        missing_list_file = os.path.join(jsonl_dir, "missing_images_list.txt")
        with open(missing_list_file, 'w', encoding='utf-8') as f:
            for img_path in sorted(all_missing_images):
                f.write(img_path + '\n')
        print(f"\n缺失图片列表已保存到: {missing_list_file}")
    
    print(f"\n✅ 检查完成!")

if __name__ == "__main__":
    check_missing_images()