#!/usr/bin/env python3
"""
Script to upload tar files to HuggingFace Hub
Usage: python upload_tars_to_hf.py --repo_id your-username/VideoNSA --token your-hf-token
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, HfFolder

def upload_tar_files(repo_id: str, token: str = None):
    """Upload tar files to HuggingFace Hub"""

    # Initialize HF API
    api = HfApi(token=token)

    # Define tar files to upload
    tar_files = [
        "checkpoint-42600-inference.tar",
        "checkpoint-42800-inference.tar",
        "datasets-jsonl.tar.gz"
    ]

    # Check if tar files exist
    for tar_file in tar_files:
        if not os.path.exists(tar_file):
            print(f"Error: Tar file '{tar_file}' not found")
            return False

    print(f"Uploading tar files to '{repo_id}'...")

    try:
        # Create repository if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        print(f"Repository '{repo_id}' created/verified")

        # Upload each tar file
        for tar_file in tar_files:
            print(f"Uploading {tar_file}...")
            file_size_gb = os.path.getsize(tar_file) / (1024**3)
            print(f"File size: {file_size_gb:.1f} GB")

            upload_file(
                path_or_fileobj=tar_file,
                path_in_repo=tar_file,
                repo_id=repo_id,
                token=token,
                repo_type="model",
                commit_message=f"Upload {tar_file} - inference-only checkpoint"
            )
            print(f"✅ Successfully uploaded {tar_file}")

        # Create a README file
        readme_content = """# VideoNSA Model Checkpoints and Datasets

This repository contains inference-ready model checkpoints and datasets from VideoNSA training.

## Files

### Model Checkpoints
- `checkpoint-42600-inference.tar` (17 GB) - Model checkpoint at step 42600 (inference-only files)
- `checkpoint-42800-inference.tar` (17 GB) - Model checkpoint at step 42800 (inference-only files)

### Datasets
- `datasets-jsonl.tar.gz` (5.1 GB) - Complete training datasets in JSONL format (compressed from 147 GB)

## Usage

### Using Model Checkpoints
1. Download the desired checkpoint tar file
2. Extract the tar file: `tar -xf checkpoint-XXXXX-inference.tar`
3. Use the extracted files for inference with ms-swift or transformers

### Using Datasets
1. Download the datasets file: `datasets-jsonl.tar.gz`
2. Extract: `tar -xzf datasets-jsonl.tar.gz`
3. Access various dataset configurations in the extracted `datasets/jsonl/` directory

## Model Checkpoint Contents

Each model checkpoint tar file contains:
- Model weights (safetensors format)
- Model configuration files
- Tokenizer files
- Template and preprocessing configurations

## Dataset Contents

The datasets package includes:
- **LLaVA-Video-117K/**: Original video datasets
- **filter_llava/**: Filtered datasets
- **filter_llava_350_550/**: Datasets filtered by 350-550 pixel resolution
- **filter_llava_400_600/**: Datasets filtered by 400-600 pixel resolution
- **keye/**: Key training datasets
- **missing_data_350_550/**: Missing data collections
- Various other filtered and processed datasets

## Model Details

These checkpoints are from VideoNSA training using the ms-swift framework.
The checkpoint files have been filtered to include only inference-necessary components,
excluding training-specific files like optimizer states and random number generator states.

The datasets represent the complete training data pipeline used for VideoNSA model development.
"""

        upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="model",
            commit_message="Add README with usage instructions"
        )
        print("✅ Successfully uploaded README.md")

        print(f"🎉 All files uploaded successfully to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload tar files to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True,
                       help="HuggingFace repository ID (username/VideoNSA)")
    parser.add_argument("--token", required=True,
                       help="HuggingFace access token")

    args = parser.parse_args()

    success = upload_tar_files(args.repo_id, args.token)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()