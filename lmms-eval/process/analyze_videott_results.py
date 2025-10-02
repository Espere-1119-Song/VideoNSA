#!/usr/bin/env python3
"""
Script to analyze VideoOTT results and diagnose why scores are 0
"""
import json
from collections import defaultdict, Counter

CATEGORIES = [
    "Objective Causality",
    "Objective Causality (Videography Phenomenon & Illusion)",
    "Element Attributes (Optical Illusion)",
    "Displacement Attribute",
    "Plot Attribute (Montage)",
    "Plot Attribute",
    "Element Attributes",
    "Element Counting",
    "Professional Knowledge",
    "Character Motivation Causality",
    "Element Localization",
    "Character Reaction Causality",
    "Event Counting",
    "Local Event Attribute",
    "Event Localization",
    "Positional Relationship",
    "Event Duration & Speed Attribute",
    "Character Emotion Attribute",
]

def analyze_videott_results(samples_file):
    """Analyze VideoOTT results and diagnose scoring issues"""
    category2score = defaultdict(lambda: {"correct": 0, "answered": 0, "scores": [], "correctness": []})
    all_scores = []
    all_correctness = []
    
    print("=== VideoOTT Result Analysis ===\n")
    
    with open(samples_file, 'r') as f:
        for line_num, line in enumerate(f):
            sample = json.loads(line)
            result = sample["videott_open_ended_score"]
            
            capability = result["capability"]
            scores = result["scores"]  # [pred, score]
            correctness = result["correctness"]
            
            category2score[capability]["answered"] += 1
            # The threshold is correctness >= 3 (from line 244 in utils.py)
            category2score[capability]["correct"] += int(correctness >= 3)
            category2score[capability]["scores"].append(scores[1])
            category2score[capability]["correctness"].append(correctness)
            
            all_scores.append(scores[1])
            all_correctness.append(correctness)
            
            # Print first few samples for debugging
            if line_num < 5:
                print(f"Sample {line_num}:")
                print(f"  Capability: {capability}")
                print(f"  GPT Scores: {scores}")
                print(f"  Correctness: {correctness}")
                print(f"  Passes threshold (>=3): {correctness >= 3}")
                print()
    
    # Overall statistics
    total_samples = len(all_scores)
    passing_samples = sum(1 for c in all_correctness if c >= 3)
    
    print(f"=== Overall Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"Samples passing threshold (score >= 3): {passing_samples} ({100*passing_samples/total_samples:.1f}%)")
    print(f"Average correctness score: {sum(all_correctness)/len(all_correctness):.2f}")
    print(f"Score distribution: {Counter([int(c) for c in all_correctness])}")
    print()
    
    # By category analysis
    print(f"=== Results by Category ===")
    for capability in sorted(category2score.keys()):
        stats = category2score[capability]
        correct = stats["correct"]
        answered = stats["answered"]
        avg_score = sum(stats["correctness"]) / len(stats["correctness"])
        passing_rate = 100 * correct / answered if answered > 0 else 0
        
        print(f"{capability}:")
        print(f"  Passing rate: {passing_rate:.1f}% ({correct}/{answered})")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Score distribution: {Counter([int(c) for c in stats['correctness']])}")
        print()
    
    # Overall performance
    total_correct = sum(stats["correct"] for stats in category2score.values())
    total_answered = sum(stats["answered"] for stats in category2score.values())
    overall_performance = 100 * total_correct / total_answered if total_answered > 0 else 0
    
    print(f"=== Final Performance ===")
    print(f"Overall Performance: {overall_performance:.1f}% ({total_correct}/{total_answered})")
    
    # Diagnose why performance is 0
    if overall_performance == 0:
        print(f"\n=== Why Performance is 0? ===")
        print(f"VideoOTT uses LLM-as-a-judge scoring with threshold >= 3.0")
        print(f"All {total_answered} samples scored < 3.0, hence 0% performance")
        print(f"This suggests the model's answers don't match the expected answers well")
        print(f"Most scores are {max(Counter([int(c) for c in all_correctness]).items(), key=lambda x: x[1])[0]}")
    
    return category2score

if __name__ == "__main__":
    samples_file = "results/sft_flash/v2-20250818-063019__checkpoint-14600/20250829_003134_samples_videott_all.jsonl"
    analyze_videott_results(samples_file)