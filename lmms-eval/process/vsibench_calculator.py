#!/usr/bin/env python3
"""
VSIBench结果计算器 - 支持文本数字转换的增强版评分计算
"""

import json
import numpy as np
import pandas as pd
import re
from pathlib import Path
import argparse


def text_to_number(text):
    """将文本数字转换为数值，支持多种格式"""
    if text is None:
        return None

    text = str(text).lower().strip()

    # 首先尝试直接转换为数字
    try:
        return float(text)
    except ValueError:
        pass

    # 文本数字映射
    text_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }

    # 直接查找简单情况
    if text in text_numbers:
        return float(text_numbers[text])

    # 处理分数表达式
    if '/' in text:
        try:
            parts = text.split('/')
            if len(parts) == 2:
                numerator = text_to_number(parts[0].strip())
                denominator = text_to_number(parts[1].strip())
                if numerator is not None and denominator is not None and denominator != 0:
                    return numerator / denominator
        except:
            pass

    # 处理小数表达式如 "one point five"
    decimal_words = ['point', 'dot']
    for word in decimal_words:
        if word in text:
            try:
                parts = text.split(word)
                if len(parts) == 2:
                    whole_part = text_to_number(parts[0].strip())
                    decimal_part = parts[1].strip()
                    if whole_part is not None:
                        # 逐位转换小数部分
                        decimal_value = 0
                        for i, char in enumerate(decimal_part.replace(' ', '')):
                            digit_val = text_to_number(char)
                            if digit_val is not None:
                                decimal_value += digit_val * (10 ** -(i+1))
                            else:
                                break
                        return whole_part + decimal_value
            except:
                pass

    # 处理复合数字如 "twenty-one", "twenty one"
    for separator in ['-', ' ']:
        if separator in text:
            try:
                parts = text.split(separator)
                total = 0
                for part in parts:
                    part = part.strip()
                    if part in text_numbers:
                        total += text_numbers[part]
                    else:
                        # 尝试递归转换
                        val = text_to_number(part)
                        if val is not None:
                            total += val
                        else:
                            total = None
                            break
                if total is not None:
                    return float(total)
            except:
                pass

    # 从包含数字的文本中提取第一个数字
    number_match = re.search(r'\d+\.?\d*', text)
    if number_match:
        try:
            return float(number_match.group())
        except:
            pass

    return None


def fuzzy_matching(pred):
    """模糊匹配处理，去掉标点符号"""
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    """精确匹配（大小写不敏感）"""
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    """计算相对误差"""
    if target == 0:
        return abs(pred - target)
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """计算平均相对准确度（MRA）"""
    if pred is None or target is None:
        return 0.0

    num_pts = int((end - start) / interval) + 1
    conf_intervs = np.linspace(start, end, num_pts)
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


# VSIBench任务类型定义
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


def calculate_scores(data):
    """计算每个样本的得分"""
    for item in data:
        vsibench_score = item.get('vsibench_score', {})
        question_type = vsibench_score.get('question_type')
        prediction = vsibench_score.get('prediction', '')
        ground_truth = vsibench_score.get('ground_truth', '')

        # 处理预测结果
        processed_pred = fuzzy_matching(prediction)

        if question_type in MCA_QUESTION_TYPES:
            # 多选题：精确匹配
            score = exact_match(processed_pred, ground_truth)
            vsibench_score['accuracy'] = score

        elif question_type in NA_QUESTION_TYPES:
            # 数值答案：使用增强的数字转换
            pred_num = text_to_number(processed_pred)
            gt_num = text_to_number(ground_truth)

            if pred_num is not None and gt_num is not None:
                mra_score = mean_relative_accuracy(pred_num, gt_num)
                vsibench_score['MRA:.5:.95:.05'] = mra_score
            else:
                vsibench_score['MRA:.5:.95:.05'] = 0.0

    return data


def aggregate_results(data):
    """聚合结果，计算各子集和总分"""
    df = pd.DataFrame([item['vsibench_score'] for item in data])

    results = {}

    # 按问题类型分组计算
    for question_type in df['question_type'].unique():
        subset_data = df[df['question_type'] == question_type]

        if question_type in MCA_QUESTION_TYPES:
            if 'accuracy' in subset_data.columns:
                results[f"{question_type}_accuracy"] = subset_data['accuracy'].mean()
        elif question_type in NA_QUESTION_TYPES:
            if 'MRA:.5:.95:.05' in subset_data.columns:
                results[f"{question_type}_MRA:.5:.95:.05"] = subset_data['MRA:.5:.95:.05'].mean()

    # 合并方向准确度（三个难度级别的平均值）
    direction_accuracies = []
    for level in ['easy', 'medium', 'hard']:
        key = f"object_rel_direction_{level}_accuracy"
        if key in results:
            direction_accuracies.append(results.pop(key))

    if direction_accuracies:
        results["object_rel_direction_accuracy"] = sum(direction_accuracies) / len(direction_accuracies)

    # 计算总分（所有指标的平均值）
    if results:
        results["overall"] = sum(results.values()) / len(results)

    return results


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_results(results, output_path):
    """保存结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='VSIBench结果计算器')
    parser.add_argument('input_file', help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', help='输出JSON文件路径（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 加载数据
    print(f"正在加载文件: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"加载了 {len(data)} 个样本")

    # 重新计算得分
    print("正在重新计算得分...")
    data = calculate_scores(data)

    # 聚合结果
    print("正在聚合结果...")
    results = aggregate_results(data)

    # 输出结果
    print("\n=== VSIBench 评估结果 ===")
    # 指定输出顺序
    output_order = [
        "overall",
        "obj_appearance_order_accuracy",
        "object_abs_distance_MRA:.5:.95:.05",
        "object_counting_MRA:.5:.95:.05",
        "object_rel_distance_accuracy",
        "object_size_estimation_MRA:.5:.95:.05",
        "room_size_estimation_MRA:.5:.95:.05",
        "route_planning_accuracy",
        "object_rel_direction_accuracy"
    ]

    for key in output_order:
        if key in results:
            print(f"{key}: {results[key]:.4f}")

    # 输出任何不在指定顺序中的其他结果
    for key, value in results.items():
        if key not in output_order:
            print(f"{key}: {value:.4f}")

    # 保存结果
    if args.output:
        save_results(results, args.output)
        print(f"\n结果已保存到: {args.output}")

    if args.verbose:
        print("\n=== 详细统计 ===")
        df = pd.DataFrame([item['vsibench_score'] for item in data])
        print(f"总样本数: {len(df)}")
        print("按问题类型分布:")
        print(df['question_type'].value_counts())


if __name__ == "__main__":
    main()