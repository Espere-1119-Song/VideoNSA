#!/usr/bin/env python3
"""
脚本功能：删除所有images字段为空的数据
用法：python remove_empty_images.py --input input.jsonl --output output.jsonl
"""

import json
import argparse
import os
from tqdm import tqdm

def is_empty_images(item):
    """检查images字段是否为空"""
    if 'images' not in item:
        return True
    
    images = item['images']
    
    # 如果是None或者空字符串
    if images is None or images == "":
        return True
    
    # 如果是空列表
    if isinstance(images, list) and len(images) == 0:
        return True
    
    # 如果是包含空字符串的列表
    if isinstance(images, list):
        # 过滤掉空字符串和None
        non_empty = [img for img in images if img is not None and img.strip() != ""]
        if len(non_empty) == 0:
            return True
    
    return False

def clean_jsonl_file(input_path, output_path):
    """清理单个JSONL文件，删除images为空的数据"""
    print(f"处理文件: {input_path}")
    
    # 先统计总行数
    print("统计文件行数...")
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    valid_data = []
    removed_count = 0
    
    # 读取并过滤数据
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="过滤数据", unit="行"):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                if is_empty_images(item):
                    removed_count += 1
                else:
                    valid_data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}, 跳过这行")
                continue
    
    # 保存过滤后的数据
    print(f"保存清理后的数据到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(valid_data, desc="保存数据", unit="条"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成!")
    print(f"原始数据: {total_lines} 条")
    print(f"删除数据: {removed_count} 条 ({removed_count/total_lines*100:.2f}%)")
    print(f"保留数据: {len(valid_data)} 条 ({len(valid_data)/total_lines*100:.2f}%)")
    
    return len(valid_data), removed_count

def clean_directory(input_dir, output_dir):
    """清理整个目录中的所有JSONL文件"""
    print(f"批量处理目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    total_original = 0
    total_kept = 0 
    total_removed = 0
    
    # 获取所有jsonl文件
    jsonl_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            jsonl_files.append(filename)
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    for filename in tqdm(jsonl_files, desc="处理文件", unit="文件"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n{'='*60}")
        kept_count, removed_count = clean_jsonl_file(input_path, output_path)
        
        total_kept += kept_count
        total_removed += removed_count
        total_original += (kept_count + removed_count)
    
    print(f"\n{'='*80}")
    print("批量处理完成!")
    print(f"总原始数据: {total_original:,} 条")
    print(f"总删除数据: {total_removed:,} 条 ({total_removed/total_original*100:.2f}%)")
    print(f"总保留数据: {total_kept:,} 条 ({total_kept/total_original*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='删除images字段为空的数据')
    parser.add_argument('--input', required=True, help='输入文件或目录路径')
    parser.add_argument('--output', required=True, help='输出文件或目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理目录中的所有jsonl文件')
    
    args = parser.parse_args()
    
    if args.batch:
        clean_directory(args.input, args.output)
    else:
        clean_jsonl_file(args.input, args.output)

if __name__ == "__main__":
    main()