#!/usr/bin/env python3
"""
Script to analyze MLVU subset scores from samples file
"""
import json
from collections import defaultdict

def analyze_mlvu_subsets(samples_file):
    """Analyze MLVU results by subset"""
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0})
    
    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["mlvu_percetion_score"]
            
            task_type = result["task_type"]
            pred_answer = result["pred_answer"].strip()
            answer = result["answer"].strip()
            
            # Create subset key
            key = task_type
            category2score[key]["answered"] += 1
            
            # Compare first letter of pred_answer and answer
            pred_first_letter = pred_answer[0].upper() if pred_answer else ""
            answer_first_letter = answer[0].upper() if answer else ""
            category2score[key]["correct"] += pred_first_letter == answer_first_letter
    
    # Calculate overall accuracy first
    total_correct = sum(stats["correct"] for stats in category2score.values())
    total_answered = sum(stats["answered"] for stats in category2score.values())
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0

    # Define output order
    output_order = [
        "Overall", "plotQA", "needleQA", "ego", "count", "order", "anomaly_reco", "topic_reasoning"
    ]

    # Print detailed results
    print("=== MLVU Subset Results ===\n")

    # Add overall to category2score for consistent handling
    category2score["Overall"] = {"correct": total_correct, "answered": total_answered}

    # Print in specified order
    for key in output_order:
        if key in category2score:
            correct = category2score[key]["correct"]
            answered = category2score[key]["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            if key == "Overall":
                print(f"{key}: {accuracy:.1f}% ({correct}/{answered})")
            else:
                print(f"{key}: {accuracy:.1f}% ({correct}/{answered})")

    # Print any remaining categories not in the specified order
    remaining_keys = set(category2score.keys()) - set(output_order)
    if remaining_keys:
        print("\n=== Additional Categories ===")
        for key in sorted(remaining_keys):
            correct = category2score[key]["correct"]
            answered = category2score[key]["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{key}: {accuracy:.1f}% ({correct}/{answered})")
    
    return category2score

if __name__ == "__main__":
    samples_file = "nsa_results/attention_mlvu/112_2048/ckpt__checkpoint-42600/20250924_013710_samples_mlvu_test.jsonl"
    analyze_mlvu_subsets(samples_file)