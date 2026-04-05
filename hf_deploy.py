#!/usr/bin/env python3
"""
Hugging Face Deployment Script
Usage: python hf_deploy.py
"""

import os
import sys
import subprocess
from pathlib import Path

def get_token():
    """Get Hugging Face token"""
    # Check environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # Check saved token file
    env_file = os.path.expanduser("~/.huggingface_token")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            return f.read().strip()
    
    return None


def check_backend_files(backend_dir):
    """Check if required files exist"""
    required = [
        "agent/rag_agent.py",
        "api/chat.py",
        "retrieval/retriever.py",
        "vector_store/qdrant_store.py",
        "simple_server.py",
        "requirements-hf.txt",
        "Dockerfile-hf"
    ]
    
    print("📦 Checking required files...")
    print()
    
    all_exist = True
    for file_path in required:
        full_path = backend_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NOT FOUND!")
            all_exist = False
    
    print()
    
    return all_exist


def create_repository(token, repo_name):
    """Create Hugging Face repository"""
    print(f"📁 Creating repository: {repo_name}")
    print()
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Create repository
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True
        )
        
        print(f"✅ Repository created: {repo_name}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False


def upload_files(token, backend_dir, repo_name):
    """Upload files to Hugging Face"""
    print(f"🚀 Uploading files to {repo_name}...")
    print()
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Files to upload
        include_patterns = [
            "agent/**/*",
            "api/**/*",
            "retrieval/**/*",
            "vector_store/**/*",
            "simple_server.py",
            "requirements-hf.txt",
            "Dockerfile-hf"
        ]
        
        # Upload folder
        api.upload_folder(
            folder_path=str(backend_dir),
            repo_id=repo_name,
            repo_type="model",
            ignore_patterns=["__pycache__/", "*.pyc", "venv/", ".venv/", "data/", "tests/"]
        )
        
        print(f"✅ Files uploaded successfully!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False


def main():
    """Main deployment function"""
    print("🚀 Hugging Face Backend Deployment")
    print("=" * 50)
    print()
    
    # Get token
    token = get_token()
    if not token:
        print("❌ Not logged in!")
        print()
        print("Run: python hf_login.py")
        sys.exit(1)
    
    print(f"✅ Logged in with token: {token[:10]}...")
    print()
    
    # Get backend directory
    script_dir = Path(__file__).parent
    backend_dir = script_dir / "backend"
    
    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        sys.exit(1)
    
    # Check files
    if not check_backend_files(backend_dir):
        print("❌ Missing required files!")
        print()
        print("Make sure you have these files in backend/:")
        print("  - agent/rag_agent.py")
        print("  - api/chat.py")
        print("  - retrieval/retriever.py")
        print("  - vector_store/qdrant_store.py")
        print("  - simple_server.py")
        print("  - requirements-hf.txt")
        print("  - Dockerfile-hf")
        sys.exit(1)
    
    # Get username
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Logged in as: {username}")
        print()
    except Exception as e:
        print(f"❌ Error getting username: {e}")
        print("Using default username...")
        username = input("Enter your Hugging Face username: ").strip()
    
    # Repository name
    repo_name = f"{username}/humanoid-robotics-backend"
    
    # Create repository
    if not create_repository(token, repo_name):
        print("❌ Failed to create repository")
        sys.exit(1)
    
    # Upload files
    if not upload_files(token, backend_dir, repo_name):
        print("❌ Failed to upload files")
        sys.exit(1)
    
    # Success!
    print("=" * 50)
    print("🎉 Deployment Successful!")
    print("=" * 50)
    print()
    print(f"Repository: https://huggingface.co/{repo_name}")
    print()
    print("Next steps:")
    print("1. Go to: https://huggingface.co/inference-endpoints")
    print("2. Click 'Create Endpoint'")
    print(f"3. Select: {repo_name}")
    print("4. Choose: AWS, us-east-1, CPU x2")
    print("5. Set environment variable: HF_TOKEN")
    print()


if __name__ == "__main__":
    main()
