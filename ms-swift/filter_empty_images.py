#!/usr/bin/env python3
import json
import os
from pathlib import Path
from tqdm import tqdm

def count_lines(file_path):
    """Count total lines in a file for progress bar"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def filter_empty_images(input_dir, output_dir):
    """
    Filter out entries with empty images array from JSONL files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_entries = 0
    filtered_entries = 0
    
    # Process all JSONL files in the input directory
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file.name}...")
        
        # Count total lines for progress bar
        total_lines = count_lines(jsonl_file)
        print(f"Total lines: {total_lines}")
        
        output_file = output_path / jsonl_file.name
        
        with open(jsonl_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Use tqdm for progress bar
            for line in tqdm(infile, total=total_lines, desc="Processing entries", unit="entries"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_entries += 1
                    
                    # Check if images field exists and is not empty
                    if 'images' in data and data['images'] and len(data['images']) > 0:
                        outfile.write(line + '\n')
                        filtered_entries += 1
                    # Optionally print filtered entries (comment out if too verbose)
                    # else:
                    #     print(f"  Filtered out entry with empty images: {data.get('video_id', 'unknown')}")
                        
                except json.JSONDecodeError as e:
                    print(f"  Error parsing JSON: {e}")
                    continue
        
        print(f"Saved filtered data to {output_file}")
    
    print(f"\n=== Summary ===")
    print(f"Total entries processed: {total_entries}")
    print(f"Entries with valid images: {filtered_entries}")
    print(f"Entries filtered out: {total_entries - filtered_entries}")
    print(f"Retention rate: {filtered_entries/total_entries*100:.2f}%" if total_entries > 0 else "No entries processed")

if __name__ == "__main__":
    input_directory = "datasets/jsonl/filter_llava_350_550"
    output_directory = "datasets/jsonl/filter_llava_350_550_no_empty"
    
    filter_empty_images(input_directory, output_directory)