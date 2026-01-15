#!/usr/bin/env python3
"""
FaceForensics++ Dataset Downloader

This script helps you download the FaceForensics++ dataset.
Since FaceForensics++ requires registration and manual approval,
this script provides guidance and helper functions.

Dataset: FaceForensics++ (c23 compression)
Target: /Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++
"""

import os
import sys
import subprocess
import platform

# Configuration
DATASET_ROOT = "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++"
FF_REPO_URL = "https://github.com/ondyari/FaceForensics.git"
FF_REPO_DIR = os.path.join(DATASET_ROOT, "FaceForensics-repo")

def print_banner():
    print("=" * 80)
    print(" FaceForensics++ Dataset Download Helper")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    # Check git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ Git installed")
    except:
        print("❌ Git not found. Please install git first:")
        print("   brew install git")
        return False
    
    # Check Python
    print(f"✅ Python {sys.version.split()[0]} installed")
    
    return True

def clone_faceforensics_repo():
    """Clone the official FaceForensics repository"""
    print("\n📥 Cloning FaceForensics++ repository...")
    
    if os.path.exists(FF_REPO_DIR):
        print(f"⚠️  Repository already exists at: {FF_REPO_DIR}")
        response = input("Do you want to update it? (y/n): ").lower()
        if response == 'y':
            os.chdir(FF_REPO_DIR)
            subprocess.run(["git", "pull"], check=True)
            print("✅ Repository updated")
        return True
    
    try:
        subprocess.run(["git", "clone", FF_REPO_URL, FF_REPO_DIR], check=True)
        print(f"✅ Repository cloned to: {FF_REPO_DIR}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return False

def show_registration_instructions():
    """Display instructions for dataset access"""
    print("\n" + "=" * 80)
    print(" IMPORTANT: Dataset Registration Required")
    print("=" * 80)
    print()
    print("FaceForensics++ requires academic/research access approval.")
    print()
    print("📝 Steps to get access:")
    print()
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Scroll to 'Access' section")
    print("3. Fill out the registration form")
    print("4. Wait for approval email (usually 1-2 days)")
    print("5. You'll receive download credentials")
    print()
    print("=" * 80)
    print()

def show_download_instructions():
    """Show how to download after getting credentials"""
    print("\n" + "=" * 80)
    print(" Download Instructions (After Approval)")
    print("=" * 80)
    print()
    print("Once you receive your credentials, run:")
    print()
    print(f"cd {FF_REPO_DIR}")
    print()
    print("# Download c23 (compressed) version - Recommended (38GB)")
    print("python download-FaceForensics.py \\")
    print("  -d FaceForensics++ \\")
    print("  -c c23 \\")
    print("  -t videos \\")
    print(f"  -o '{DATASET_ROOT}'")
    print()
    print("# Or download raw version (500GB - requires lots of space)")
    print("# python download-FaceForensics.py -d FaceForensics++ -c raw -t videos -o '{DATASET_ROOT}'")
    print()
    print("=" * 80)
    print()

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    dirs = [
        DATASET_ROOT,
        os.path.join(DATASET_ROOT, "videos"),
        os.path.join(DATASET_ROOT, "frames"),
        os.path.join(DATASET_ROOT, "frames/real"),
        os.path.join(DATASET_ROOT, "frames/fake"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Created: {d}")
    
    return True

def check_existing_data():
    """Check if dataset already exists"""
    print("\n🔍 Checking for existing data...")
    
    video_dir = os.path.join(DATASET_ROOT, "videos")
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        if video_files:
            print(f"✅ Found {len(video_files)} videos in {video_dir}")
            return True
    
    print("❌ No videos found. You need to download the dataset first.")
    return False

def main():
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Create directories
    create_directory_structure()
    
    # Step 3: Clone repository
    if not clone_faceforensics_repo():
        return
    
    # Step 4: Show registration instructions
    show_registration_instructions()
    
    # Step 5: Check if user has credentials
    print("Do you already have download credentials? (y/n): ", end="")
    has_credentials = input().lower() == 'y'
    
    if has_credentials:
        show_download_instructions()
        
        print("\n📋 After downloading, you can:")
        print("1. Extract frames: python model/extract_ff_frames.py")
        print("2. Fine-tune model: python model/src/finetune_faceforensics.py")
    else:
        print("\n⏳ Please follow the registration steps above.")
        print("   Once approved, run this script again and select 'yes' when asked about credentials.")
    
    print("\n✅ Setup complete!")
    print(f"📁 Dataset location: {DATASET_ROOT}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
