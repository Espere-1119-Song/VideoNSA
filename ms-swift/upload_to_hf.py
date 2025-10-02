#!/usr/bin/env python3
"""
Script to upload model to HuggingFace Hub
Usage: python upload_to_hf.py --repo_id your-username/model-name --token your-hf-token
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, Repository, upload_folder

def upload_model(repo_id: str, token: str, model_path: str = "huggingface_model"):
    """Upload model to HuggingFace Hub"""
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' not found")
        return False
    
    print(f"Uploading model from '{model_path}' to '{repo_id}'...")
    
    try:
        # Create repository if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        print(f"Repository '{repo_id}' created/verified")
        
        # Upload the entire folder
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            repo_type="model",
            commit_message="Upload fine-tuned model from ms-swift checkpoint",
            create_pr=False
        )
        
        print(f"✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True, 
                       help="HuggingFace repository ID (username/model-name)")
    parser.add_argument("--token", required=True,
                       help="HuggingFace access token")
    parser.add_argument("--model_path", default="huggingface_model",
                       help="Path to model directory (default: huggingface_model)")
    
    args = parser.parse_args()
    
    success = upload_model(args.repo_id, args.token, args.model_path)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()