#!/usr/bin/env python3
"""
Script to analyze LongTimeScope subset scores from samples file
"""
import json
from collections import defaultdict

def analyze_longtimescope_subsets(samples_file):
    """Analyze LongTimeScope results by subset"""
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0})
    
    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["timescope_perception_score"]
            
            task_type = result["task_type"]
            length = result["length"]
            pred_answer = result["pred_answer"]
            answer = result["answer"]

            
            # Create subset key
            key = f"{length}_{task_type}"
            category2score[key]["answered"] += 1
            category2score[key]["correct"] += pred_answer.lower() == answer.lower()
    
    # Print detailed results
    print("=== LongTimeScope Subset Results ===\n")
    
    # Get unique lengths for overall length statistics
    lengths = set()
    for key in category2score:
        length = key.split("_")[0]
        lengths.add(int(length))
    
    # Print by length and task type
    for key in sorted(category2score.keys()):
        length, task_type = key.split("_")
        correct = category2score[key]["correct"]
        answered = category2score[key]["answered"]
        accuracy = 100 * correct / answered if answered > 0 else 0
        print(f"Video Length: {length}, Task: {task_type}: {accuracy:.1f}% ({correct}/{answered})")
    
    print()
    
    # Print by video length overall
    for length in sorted(lengths):
        total_correct = 0
        total_answered = 0
        for key, stats in category2score.items():
            if str(length) == key.split("_")[0]:
                total_correct += stats["correct"]
                total_answered += stats["answered"]
        
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        print(f"Video Length: {length}: {accuracy:.1f}% ({total_correct}/{total_answered})")
    
    print()
    
    # Overall accuracy
    total_correct = sum(stats["correct"] for stats in category2score.values())
    total_answered = sum(stats["answered"] for stats in category2score.values())
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_answered})")
    
    return category2score

if __name__ == "__main__":
    samples_file = "nsa_results/attention_longtimescope/64_512/ckpt__checkpoint-42600/20250924_070503_samples_longtimescope.jsonl"
    analyze_longtimescope_subsets(samples_file)