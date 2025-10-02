#!/usr/bin/env python3
"""
Script to analyze VideoMME subset results from samples file
"""
import json
from collections import defaultdict, Counter

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]
TASK_CATEGORIES = [
    "Temporal Perception", "Spatial Perception", "Attribute Perception", "Action Recognition",
    "Object Recognition", "OCR Problems", "Counting Problem", "Temporal Reasoning",
    "Spatial Reasoning", "Action Reasoning", "Object Reasoning", "Information Synopsis",
]

def analyze_videomme_results(samples_file):
    """Analyze VideoMME results by subset"""
    video_type_scores = defaultdict(lambda: {"correct": 0, "answered": 0})
    category_scores = defaultdict(lambda: {"correct": 0, "answered": 0})
    sub_category_scores = defaultdict(lambda: {"correct": 0, "answered": 0})
    task_category_scores = defaultdict(lambda: {"correct": 0, "answered": 0})
    
    print("=== VideoMME Result Analysis ===\n")
    
    # Collect all unique sub_categories
    all_sub_categories = set()
    
    with open(samples_file, 'r') as f:
        for line_num, line in enumerate(f):
            sample = json.loads(line)
            result = sample["videomme_perception_score"]
            
            duration = result["duration"]
            category = result["category"]
            sub_category = result["sub_category"]
            task_category = result["task_category"]
            pred_answer = result["pred_answer"]
            answer = result["answer"]
            
            all_sub_categories.add(sub_category)
            is_correct = (pred_answer == answer)
            
            # Update scores for each dimension
            video_type_scores[duration]["answered"] += 1
            video_type_scores[duration]["correct"] += is_correct
            
            category_scores[category]["answered"] += 1
            category_scores[category]["correct"] += is_correct
            
            sub_category_scores[sub_category]["answered"] += 1
            sub_category_scores[sub_category]["correct"] += is_correct
            
            task_category_scores[task_category]["answered"] += 1
            task_category_scores[task_category]["correct"] += is_correct
            
            # Print first few samples for debugging
            if line_num < 3:
                print(f"Sample {line_num}:")
                print(f"  Question ID: {result['question_id']}")
                print(f"  Duration: {duration}")
                print(f"  Category: {category}")
                print(f"  Sub Category: {sub_category}")
                print(f"  Task Category: {task_category}")
                print(f"  Answer: {answer}, Predicted: {pred_answer}")
                print(f"  Correct: {is_correct}")
                print()
    
    # Print results by video type (duration)
    print("=== Results by Video Type (Duration) ===")
    for video_type in VIDEO_TYPE:
        if video_type in video_type_scores:
            stats = video_type_scores[video_type]
            correct = stats["correct"]
            answered = stats["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{video_type}: {accuracy:.2f}% ({correct}/{answered})")
    
    print()
    
    # Print results by main category
    print("=== Results by Main Category ===")
    for category in CATEGORIES:
        if category in category_scores:
            stats = category_scores[category]
            correct = stats["correct"]
            answered = stats["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{category}: {accuracy:.2f}% ({correct}/{answered})")
    
    print()
    
    # Print results by task category
    print("=== Results by Task Category ===")
    for task_category in TASK_CATEGORIES:
        if task_category in task_category_scores:
            stats = task_category_scores[task_category]
            correct = stats["correct"]
            answered = stats["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            print(f"{task_category}: {accuracy:.2f}% ({correct}/{answered})")
    
    print()
    
    # Print results by sub category (top and bottom performing)
    print("=== Results by Sub Category (sorted by accuracy) ===")
    sub_cat_results = []
    for sub_category, stats in sub_category_scores.items():
        correct = stats["correct"]
        answered = stats["answered"]
        accuracy = 100 * correct / answered if answered > 0 else 0
        sub_cat_results.append((sub_category, accuracy, correct, answered))
    
    sub_cat_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 Sub Categories:")
    for sub_category, accuracy, correct, answered in sub_cat_results[:10]:
        print(f"  {sub_category}: {accuracy:.2f}% ({correct}/{answered})")
    
    print("\nBottom 10 Sub Categories:")
    for sub_category, accuracy, correct, answered in sub_cat_results[-10:]:
        print(f"  {sub_category}: {accuracy:.2f}% ({correct}/{answered})")
    
    print()
    
    # Calculate overall accuracy
    total_correct = sum(stats["correct"] for stats in video_type_scores.values())
    total_answered = sum(stats["answered"] for stats in video_type_scores.values())
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    
    print(f"=== Overall Performance ===")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_answered})")
    
    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Video Types: {len([v for v in VIDEO_TYPE if v in video_type_scores])}")
    print(f"Main Categories: {len([c for c in CATEGORIES if c in category_scores])}")
    print(f"Sub Categories: {len(all_sub_categories)}")
    print(f"Task Categories: {len([t for t in TASK_CATEGORIES if t in task_category_scores])}")
    print(f"Total Samples: {total_answered}")
    
    return overall_accuracy

if __name__ == "__main__":
    samples_file = "results/sft_flash/v2-20250818-063019__checkpoint-14600/20250829_150102_samples_videomme.jsonl"
    analyze_videomme_results(samples_file)