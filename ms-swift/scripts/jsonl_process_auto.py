import json
import re
import argparse
import os
from glob import glob

def auto_calculate_step(total_frames, target_frames):
    """
    根据原始帧数和目标帧数自动计算采样间隔
    返回采样间隔，使得采样后的帧数接近但不超过目标帧数
    """
    if total_frames <= target_frames:
        return 0  # 无需采样
    
    # 计算需要的采样间隔
    # 例如：36帧 -> 27帧，需要保留27帧，丢弃9帧
    # 采样间隔 = (36 - 27) / 27 ≈ 0.33，向上取整为1
    # 即隔1个采样1个：取第0,2,4,6,8...帧
    
    # 方法1：基于丢弃比例计算
    keep_ratio = target_frames / total_frames
    step = int((1 - keep_ratio) / keep_ratio)
    
    # 方法2：直接计算间隔
    # 如果总共有N帧，要保留M帧，则间隔为 (N-M)/M
    step_alt = int((total_frames - target_frames) / target_frames)
    
    # 取两种方法的较小值，确保不超过目标帧数
    step = min(step, step_alt)
    
    # 验证采样结果
    sampled_indices = list(range(0, total_frames, step + 1))
    if len(sampled_indices) > target_frames:
        # 如果仍然超过，进一步调整
        step = int((total_frames - target_frames) / target_frames)
        sampled_indices = list(range(0, total_frames, step + 1))
        sampled_indices = sampled_indices[:target_frames]
    
    return step

def custom_sample(lst, step, max_frames):
    """
    根据采样间隔进行采样
    """
    if len(lst) <= max_frames:
        return lst
    
    indices = list(range(0, len(lst), step + 1))
    
    # 确保不超过最大帧数
    if len(indices) > max_frames:
        indices = indices[:max_frames]
    
    return [lst[i] for i in indices]

def process_line(data, target_frames=27):
    """处理单行数据"""
    # 找到 user content 里的 <image> token
    user_msg = next(m for m in data['messages'] if m['role'] == 'user')
    content = user_msg['content']
    image_tokens = re.findall(r'<image>', content)
    total = len(image_tokens)
    
    if total <= target_frames:
        # 如果原始帧数不超过限制，直接返回
        return data
    
    # 自动计算采样间隔
    step = auto_calculate_step(total, target_frames)
    
    # 自定义采样
    sampled_indices = list(range(0, total, step + 1))
    if len(sampled_indices) > target_frames:
        sampled_indices = sampled_indices[:target_frames]
    
    # 处理 content
    new_content = ''
    count = 0
    for match in re.finditer(r'<image>', content):
        if count in sampled_indices:
            new_content += '<image>'
        count += 1
    
    # 保留 <image> 后面的文本
    rest = re.split(r'(<image>)+', content, maxsplit=1)[-1]
    if rest.strip() and not rest.startswith('<image>'):
        new_content += rest
    
    user_msg['content'] = new_content
    
    # 处理 images
    images = data.get('images', [])
    data['images'] = [images[i] for i in sampled_indices]
    
    return data, step, len(sampled_indices)

def main():
    parser = argparse.ArgumentParser(description='自动计算采样间隔处理视频帧')
    parser.add_argument('--input', type=str, default=None, help='输入文件路径')
    parser.add_argument('--input_dir', type=str, default='datasets/jsonl/LLaVA-Video-178K', help='输入文件夹路径（批量处理）')
    parser.add_argument('--output_dir', type=str, default='datasets/jsonl/keye/full', help='输出文件夹路径（批量处理）')
    parser.add_argument('--target_frames', type=int, default=6, help='目标帧数')
    args = parser.parse_args()

    if args.input_dir:
        input_files = glob(os.path.join(args.input_dir, "*.jsonl"))
        output_dir = args.output_dir or args.input_dir
        os.makedirs(output_dir, exist_ok=True)
        for input_file in input_files:
            base = os.path.basename(input_file)
            output_file = os.path.join(output_dir, base)
            print(f"处理: {input_file} -> {output_file}")
            process_file(input_file, output_file, args.target_frames)
    elif args.input:
        output_file = args.output or (args.input + ".processed.jsonl")
        process_file(args.input, output_file, args.target_frames)
    else:
        print("请指定 --input 或 --input_dir")
        return

def process_file(input_path, output_path, target_frames):
    processed_count = 0
    total_count = 0
    step_stats = {}
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            data = json.loads(line)
            user_msg = next(m for m in data['messages'] if m['role'] == 'user')
            content = user_msg['content']
            image_tokens = re.findall(r'<image>', content)
            original_frames = len(image_tokens)
            if original_frames > target_frames:
                processed_count += 1
                new_data, step, final_frames = process_line(data, target_frames)
                step_stats[step] = step_stats.get(step, 0) + 1
                fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            else:
                fout.write(line)
    print(f'文件 {input_path} 处理完成！总样本数: {total_count}，处理样本数: {processed_count}')
    print(f'采样间隔统计: {step_stats}')

if __name__ == '__main__':
    main() 