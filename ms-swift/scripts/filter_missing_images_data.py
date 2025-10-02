#!/usr/bin/env python3
"""
脚本功能：逐个处理JSONL文件，筛选出包含缺失图片的数据条目并保存
用法：python filter_missing_images_data.py
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

def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"保存 {os.path.basename(file_path)}", unit="条"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 已保存 {len(data)} 条记录到 {file_path}")

def process_single_file(input_file, output_file, base_dir):
    """处理单个JSONL文件"""
    print(f"\n处理文件: {os.path.basename(input_file)}")
    print("-" * 60)
    
    # 加载数据
    data = load_jsonl(input_file)
    if not data:
        print(f"❌ 文件为空或无法读取")
        return
    
    missing_data = []
    missing_images = set()
    total_images = 0
    total_items = len(data)
    
    # 检查每个数据条目
    for item in tqdm(data, desc="检查图片", unit="条"):
        if 'images' in item and item['images']:
            has_missing = False
            item_missing_images = []
            
            for image_path in item['images']:
                total_images += 1
                
                # 转换为绝对路径
                if image_path.startswith('datasets/'):
                    full_path = os.path.join(base_dir, image_path)
                else:
                    full_path = image_path
                
                # 检查文件是否存在
                if not os.path.exists(full_path):
                    missing_images.add(image_path)
                    item_missing_images.append(image_path)
                    has_missing = True
            
            # 如果有缺失图片，保存这个数据条目
            if has_missing:
                # 复制原数据并添加缺失信息
                item_copy = item.copy()
                item_copy['missing_images'] = item_missing_images
                missing_data.append(item_copy)
    
    # 打印统计信息
    print(f"📊 统计信息:")
    print(f"   总数据条目: {total_items}")
    print(f"   总图片数量: {total_images}")
    print(f"   缺失图片数量: {len(missing_images)}")
    print(f"   包含缺失图片的条目数: {len(missing_data)}")
    
    if total_images > 0:
        print(f"   图片缺失率: {len(missing_images)/total_images*100:.2f}%")
    if total_items > 0:
        print(f"   条目缺失率: {len(missing_data)/total_items*100:.2f}%")
    
    # 保存结果
    if missing_data:
        save_jsonl(missing_data, output_file)
        
        # 保存缺失图片列表
        missing_list_file = output_file.replace('.jsonl', '_missing_list.txt')
        with open(missing_list_file, 'w', encoding='utf-8') as f:
            for img_path in sorted(missing_images):
                f.write(img_path + '\n')
        print(f"📝 缺失图片列表已保存到: {missing_list_file}")
    else:
        print(f"✅ 该文件没有缺失图片")
    
    return len(missing_data), len(missing_images)

def main():
    """主函数"""
    # 定义路径
    base_dir = "/home/ubuntu/jianwen-us-south-2/tulab/enxin/projects/ms-swift"
    input_dir = os.path.join(base_dir, "datasets/jsonl/filter_llava_350_550")
    output_dir = os.path.join(base_dir, "datasets/jsonl/filtered_missing_images")
    
    # 需要处理的文件列表
    files_to_process = [
        "2_3_m_nextqa.jsonl"
    ]
    
    print("🚀 开始处理JSONL文件，筛选缺失图片数据...")
    print("=" * 80)
    
    # 统计变量
    total_missing_items = 0
    total_missing_images = 0
    processed_files = 0
    
    # 逐个处理文件
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"filtered_{filename}")
        
        if not os.path.exists(input_file):
            print(f"⚠️  跳过 {filename}: 文件不存在")
            continue
        
        try:
            missing_items, missing_imgs = process_single_file(input_file, output_file, base_dir)
            total_missing_items += missing_items
            total_missing_images += missing_imgs
            processed_files += 1
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")
    
    # 输出总体统计
    print("\n" + "=" * 80)
    print("🎯 总体统计:")
    print(f"   处理文件数: {processed_files}")
    print(f"   总缺失数据条目: {total_missing_items:,}")
    print(f"   总缺失图片数量: {total_missing_images:,}")
    print(f"   输出目录: {output_dir}")
    
    print(f"\n✅ 处理完成!")

if __name__ == "__main__":
    main()