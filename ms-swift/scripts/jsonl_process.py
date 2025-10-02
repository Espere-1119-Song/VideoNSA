import json
import re

def even_sample(lst, n):
    """均匀采样n个元素"""
    if n >= len(lst):
        return lst
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

def process_line(data):
    # 找到 user content 里的 <image> token
    user_msg = next(m for m in data['messages'] if m['role'] == 'user')
    content = user_msg['content']
    image_tokens = re.findall(r'<image>', content)
    total = len(image_tokens)
    half_n = max(1, total // 4)
    # 均匀采样保留的索引
    keep_indices = [int(i * total / half_n) for i in range(half_n)]
    # 处理 content
    new_content = ''
    count = 0
    for match in re.finditer(r'<image>', content):
        if count in keep_indices:
            new_content += '<image>'
        count += 1
    # 保留 <image> 后面的文本
    rest = re.split(r'(<image>)+', content, maxsplit=1)[-1]
    if rest.strip() and not rest.startswith('<image>'):
        new_content += rest
    user_msg['content'] = new_content
    # 处理 images
    images = data.get('images', [])
    data['images'] = even_sample(images, half_n)
    return data

input_path = 'datasets/jsonl/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1.jsonl'
output_path = 'datasets/jsonl/LLaVA-Video-178K/0_30_s_academic_v0_1.jsonl'

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        new_data = process_line(data)
        fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print('处理完成，输出文件：', output_path)