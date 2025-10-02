#!/usr/bin/env python3
"""
Script to analyze TOMATO subset scores from samples file
"""
import json
from collections import defaultdict

def analyze_tomato_subsets(samples_file):
    """Analyze TOMATO results by subset"""
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0})
    
    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["tomato_score"]
            
            reason_type = result["reason_type"]
            demonstration_type = result["demonstration_type"]
            score = result["score"]  # 1.0 = correct, 0.0 = incorrect
            
            # Create subset keys
            reason_key = f"reason_{reason_type}"
            demo_key = f"demo_{demonstration_type}"
            combined_key = f"{reason_type}_{demonstration_type}"
            
            # Update statistics
            for key in [reason_key, demo_key, combined_key]:
                category2score[key]["answered"] += 1
                category2score[key]["correct"] += int(score == 1.0)
    
    # Overall accuracy first
    # Count each sample only once for overall accuracy
    total_samples = defaultdict(int)
    correct_samples = defaultdict(int)

    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["tomato_score"]
            question_id = result["question_id"]
            score = result["score"]

            total_samples[question_id] = 1
            correct_samples[question_id] = int(score == 1.0)

    total_correct = sum(correct_samples.values())
    total_answered = len(total_samples)
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0

    print("=== TOMATO Subset Results ===\n")
    print(f"Overall: {overall_accuracy:.1f}% ({total_correct}/{total_answered})")

    # Define the specific order for reason types
    reason_order = ["direction", "count", "rotation", "shape&trend", "velocity&frequency", "visual cues"]

    # Print by reason type in specified order
    for reason_type in reason_order:
        key = f"reason_{reason_type}"
        if key in category2score:
            correct = category2score[key]["correct"]
            answered = category2score[key]["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{reason_type}: {accuracy:.1f}% ({correct}/{answered})")

    # Define the specific order for demonstration types
    demo_order = ["human", "simulated", "object"]

    # Print by demonstration type in specified order
    for demo_type in demo_order:
        key = f"demo_{demo_type}"
        if key in category2score:
            correct = category2score[key]["correct"]
            answered = category2score[key]["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{demo_type}: {accuracy:.1f}% ({correct}/{answered})")
    
    return category2score

if __name__ == "__main__":
    samples_file = "nsa_results/attention_vsibench/16_128/ckpt__checkpoint-42600/20250925_041131_samples_tomato.jsonl"
    analyze_tomato_subsets(samples_file)