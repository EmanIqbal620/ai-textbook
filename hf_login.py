#!/usr/bin/env python3
"""
Hugging Face Login and Deployment Script for Windows
Usage: python hf_login.py
"""

import os
import sys

def login():
    """Login to Hugging Face"""
    print("🔐 Hugging Face Login")
    print("=" * 50)
    print()
    
    # Get token from user
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ Token is required!")
        return False
    
    if not token.startswith("hf_"):
        print("❌ Invalid token format! Tokens start with 'hf_'")
        return False
    
    # Save token to environment
    env_file = os.path.expanduser("~/.huggingface_token")
    
    try:
        with open(env_file, "w") as f:
            f.write(token)
        
        # Set environment variable
        os.environ["HF_TOKEN"] = token
        
        print()
        print("✅ Login successful!")
        print()
        print("Token saved to:", env_file)
        print()
        print("Next steps:")
        print("1. Run: python hf_deploy.py")
        print("2. Or set HF_TOKEN environment variable")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        return False


def check_login():
    """Check if already logged in"""
    env_file = os.path.expanduser("~/.huggingface_token")
    
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            token = f.read().strip()
        
        if token:
            os.environ["HF_TOKEN"] = token
            print("✅ Already logged in!")
            return True
    
    return False


if __name__ == "__main__":
    # Check if already logged in
    if check_login():
        print("Token found. Ready to deploy!")
        print()
        print("Run: python hf_deploy.py")
    else:
        # Login
        success = login()
        
        if not success:
            sys.exit(1)
