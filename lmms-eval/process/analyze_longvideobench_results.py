#!/usr/bin/env python3
"""
Script to analyze LongVideoBench subset results from samples file
"""
import json
from collections import defaultdict, Counter

def analyze_longvideobench_results(samples_file):
    """Analyze LongVideoBench results by subset"""
    subset_scores = defaultdict(lambda: {"correct": 0, "answered": 0})
    duration_groups = set()
    question_categories = set()
    
    print("=== LongVideoBench Result Analysis ===\n")
    
    # First pass to collect all data
    with open(samples_file, 'r') as f:
        for line_num, line in enumerate(f):
            sample = json.loads(line)
            result = sample["lvb_acc"]
            
            duration_group = result["duration_group"]
            question_category = result["question_category"]
            answer = result["answer"]
            parsed_pred = result["parsed_pred"]
            
            # Track unique values
            duration_groups.add(duration_group)
            question_categories.add(question_category)
            
            # Calculate accuracy for each subset
            subset_scores[duration_group]["answered"] += 1
            subset_scores[duration_group]["correct"] += (answer == parsed_pred)
            
            subset_scores[question_category]["answered"] += 1
            subset_scores[question_category]["correct"] += (answer == parsed_pred)
            
            # Print first few samples for debugging
            if line_num < 3:
                print(f"Sample {line_num}:")
                print(f"  ID: {result['id']}")
                print(f"  Duration Group: {duration_group}")
                print(f"  Question Category: {question_category}")
                print(f"  Answer: {answer}, Predicted: {parsed_pred}")
                print(f"  Correct: {answer == parsed_pred}")
                print()
    
    # Define output order
    output_order = [
        "overall", "600", "TOS", "S2E", "E3E", "S2A", "SAA", "O3O", "T3O", "T3E",
        "O2E", "T2O", "S2O", "TAA", "T2E", "E2O", "SSS", "T2A", "60", "SOS", "15", "3600"
    ]

    # Calculate overall accuracy first for display
    total_correct = 0
    total_answered = 0

    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["lvb_acc"]
            answer = result["answer"]
            parsed_pred = result["parsed_pred"]

            total_answered += 1
            total_correct += (answer == parsed_pred)

    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0

    # Print results in specified order
    print("=== Results ===")

    # Add overall to subset_scores for consistent handling
    subset_scores["overall"] = {"correct": total_correct, "answered": total_answered}

    for key in output_order:
        # Try both string and integer versions of the key
        actual_key = key
        if key not in subset_scores and key.isdigit():
            actual_key = int(key)

        if actual_key in subset_scores:
            stats = subset_scores[actual_key]
            correct = stats["correct"]
            answered = stats["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{key}: {accuracy:.2f}% ({correct}/{answered})")

    # Print any remaining categories not in the specified order
    all_keys = set(subset_scores.keys())
    # Convert output_order to include both string and int versions
    output_order_expanded = set(output_order)
    for key in output_order:
        if key.isdigit():
            output_order_expanded.add(int(key))

    remaining_keys = all_keys - output_order_expanded
    if remaining_keys:
        print("\n=== Additional Categories ===")
        for key in sorted(remaining_keys, key=str):
            stats = subset_scores[key]
            correct = stats["correct"]
            answered = stats["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{key}: {accuracy:.2f}% ({correct}/{answered})")

    print()

    # Summary stats
    duration_groups = [key for key in subset_scores.keys() if isinstance(key, (int, str)) and str(key).isdigit()]
    question_categories = [key for key in subset_scores.keys() if not (isinstance(key, (int, str)) and str(key).isdigit()) and key != "overall"]

    print(f"=== Summary ===")
    print(f"Duration Groups: {len(duration_groups)} groups")
    print(f"Question Categories: {len(question_categories)} categories")
    print(f"Total Samples: {total_answered}")

    return subset_scores, overall_accuracy

if __name__ == "__main__":
    samples_file = "nsa_results/attention_longvid/17_64/ckpt__checkpoint-42600/20250924_075251_samples_longvideobench_val_v.jsonl"
    analyze_longvideobench_results(samples_file)