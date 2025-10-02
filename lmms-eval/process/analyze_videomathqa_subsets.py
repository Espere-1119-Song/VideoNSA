#!/usr/bin/env python3
"""
Script to analyze VideoMathQA subset scores from samples file
"""
import json
from collections import defaultdict

VIDEO_LENGTH = ["short", "medium", "long"]
CATEGORIES = ["Geometry Angle", "Geometry Area", "Geometry Length", "Chart", "Statistics", 
             "Arithmetic", "Topology", "Graph Theory", "Counting", "Puzzle"]

def analyze_videomathqa_subsets(samples_file):
    """Analyze VideoMathQA results by subset"""
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0})
    
    with open(samples_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            result = sample["videomathqa_perception_score"]
            
            duration = result["duration"]
            category = result["category"]
            pred_answer = result["pred_answer"]
            answer = result["answer"]
            
            # Create subset key
            key = f"{duration}_{category}"
            category2score[key]["answered"] += 1
            category2score[key]["correct"] += pred_answer == answer
    
    # Print detailed results
    print("=== VideoMathQA Subset Results ===\n")
    
    # Print by video length
    print("Results by Video Length:")
    for video_length in VIDEO_LENGTH:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_length in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        print(f"  {video_length}: {score:.1f}% ({total_correct}/{total_answered})")
    
    print()
    
    # Print by category
    print("Results by Category:")
    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        score = 100 * total_correct / total_answered if total_answered > 0 else 0
        print(f"  {category}: {score:.1f}% ({total_correct}/{total_answered})")
    
    print()
    
    # Print detailed breakdown (length x category)
    print("Detailed Breakdown (Length × Category):")
    for key in sorted(category2score.keys()):
        if category2score[key]["answered"] > 0:  # Only show categories with samples
            correct = category2score[key]["correct"]
            answered = category2score[key]["answered"]
            accuracy = 100 * correct / answered
            length, category = key.split("_", 1)
            print(f"  {length} - {category}: {accuracy:.1f}% ({correct}/{answered})")
    
    print()
    
    # Overall accuracy
    total_correct = sum(v["correct"] for v in category2score.values())
    total_answered = sum(v["answered"] for v in category2score.values())
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_answered})")
    
    return category2score

if __name__ == "__main__":
    samples_file = "results/full/Qwen__Qwen2.5-VL-7B-Instruct/20250828_050859_samples_videomathqa_mcq.jsonl"
    analyze_videomathqa_subsets(samples_file)