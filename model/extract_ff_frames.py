#!/usr/bin/env python3
"""
Extract frames from FaceForensics++ videos

This script extracts frames from downloaded FaceForensics++ videos
and organizes them into train/val directories for fine-tuning.

Usage:
    python extract_ff_frames.py

Configuration:
    - Extracts 10 frames per video (evenly spaced)
    - Organizes into real/fake directories
    - 80/20 train/val split
"""

import os
import cv2
import glob
from tqdm import tqdm
import random
import shutil

# Configuration
FF_DATASET_ROOT = "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++"
VIDEOS_DIR = os.path.join(FF_DATASET_ROOT, "videos")
FRAMES_OUTPUT = os.path.join(FF_DATASET_ROOT, "frames")

# Extraction settings
FRAMES_PER_VIDEO = 10  # Extract 10 evenly-spaced frames per video
TRAIN_SPLIT = 0.8      # 80% train, 20% validation

def get_video_info(video_path):
    """Get video frame count and FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return total_frames, fps

def extract_frames(video_path, output_dir, num_frames=10):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted = 0
    
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            output_path = os.path.join(output_dir, f"{video_name}_frame{idx:03d}.jpg")
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
    
    cap.release()
    return extracted

def organize_dataset():
    """Organize extracted frames into train/val split"""
    print("\n📊 Organizing dataset into train/val splits...")
    
    # Create directories
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            dir_path = os.path.join(FRAMES_OUTPUT, split, label)
            os.makedirs(dir_path, exist_ok=True)
    
    # Get all frames
    all_real = glob.glob(os.path.join(FRAMES_OUTPUT, "real", "*.jpg"))
    all_fake = glob.glob(os.path.join(FRAMES_OUTPUT, "fake", "*.jpg"))
    
    print(f"   Real frames: {len(all_real)}")
    print(f"   Fake frames: {len(all_fake)}")
    
    # Shuffle and split
    random.shuffle(all_real)
    random.shuffle(all_fake)
    
    real_split = int(len(all_real) * TRAIN_SPLIT)
    fake_split = int(len(all_fake) * TRAIN_SPLIT)
    
    # Move files
    for img_path in tqdm(all_real[:real_split], desc="Moving real (train)"):
        shutil.move(img_path, os.path.join(FRAMES_OUTPUT, "train", "real", os.path.basename(img_path)))
    
    for img_path in tqdm(all_real[real_split:], desc="Moving real (val)"):
        shutil.move(img_path, os.path.join(FRAMES_OUTPUT, "val", "real", os.path.basename(img_path)))
    
    for img_path in tqdm(all_fake[:fake_split], desc="Moving fake (train)"):
        shutil.move(img_path, os.path.join(FRAMES_OUTPUT, "train", "fake", os.path.basename(img_path)))
    
    for img_path in tqdm(all_fake[fake_split:], desc="Moving fake (val)"):
        shutil.move(img_path, os.path.join(FRAMES_OUTPUT, "val", "fake", os.path.basename(img_path)))
    
    print("✅ Dataset organized!")

def main():
    print("=" * 80)
    print(" FaceForensics++ Frame Extraction")
    print("=" * 80)
    
    # Check if videos exist
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Videos directory not found: {VIDEOS_DIR}")
        print("   Please download FaceForensics++ videos first using download_faceforensics.py")
        return
    
    # Find video files
    print(f"\n🔍 Searching for videos in: {VIDEOS_DIR}")
    
    # FaceForensics++ structure: videos are in subdirectories
    # Looking for: original_sequences, manipulated_sequences
    video_patterns = [
        os.path.join(VIDEOS_DIR, "original_sequences", "youtube", "c23", "videos", "*.mp4"),
        os.path.join(VIDEOS_DIR, "manipulated_sequences", "Deepfakes", "c23", "videos", "*.mp4"),
        os.path.join(VIDEOS_DIR, "manipulated_sequences", "Face2Face", "c23", "videos", "*.mp4"),
        os.path.join(VIDEOS_DIR, "manipulated_sequences", "FaceSwap", "c23", "videos", "*.mp4"),
        os.path.join(VIDEOS_DIR, "manipulated_sequences", "NeuralTextures", "c23", "videos", "*.mp4"),
    ]
    
    real_videos = []
    fake_videos = []
    
    # Collect real videos
    pattern = os.path.join(VIDEOS_DIR, "original_sequences", "youtube", "c23", "videos", "*.mp4")
    real_videos = glob.glob(pattern)
    
    # Collect fake videos (from all manipulation methods)
    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        pattern = os.path.join(VIDEOS_DIR, "manipulated_sequences", method, "c23", "videos", "*.mp4")
        fake_videos.extend(glob.glob(pattern))
    
    # If standard structure not found, try searching recursively
    if not real_videos and not fake_videos:
        print("⚠️  Standard FaceForensics++ structure not found. Searching recursively...")
        all_videos = glob.glob(os.path.join(VIDEOS_DIR, "**", "*.mp4"), recursive=True)
        
        for v in all_videos:
            if "original" in v.lower() or "youtube" in v.lower():
                real_videos.append(v)
            else:
                fake_videos.append(v)
    
    print(f"\n📹 Found videos:")
    print(f"   Real: {len(real_videos)}")
    print(f"   Fake: {len(fake_videos)}")
    
    if not real_videos and not fake_videos:
        print("\n❌ No videos found!")
        print("   Please ensure FaceForensics++ videos are downloaded to:")
        print(f"   {VIDEOS_DIR}")
        return
    
    # Create output directories
    os.makedirs(os.path.join(FRAMES_OUTPUT, "real"), exist_ok=True)
    os.makedirs(os.path.join(FRAMES_OUTPUT, "fake"), exist_ok=True)
    
    # Extract frames from real videos
    print(f"\n⚙️  Extracting frames from real videos...")
    total_real_frames = 0
    for video_path in tqdm(real_videos, desc="Real videos"):
        extracted = extract_frames(
            video_path, 
            os.path.join(FRAMES_OUTPUT, "real"),
            num_frames=FRAMES_PER_VIDEO
        )
        total_real_frames += extracted
    
    # Extract frames from fake videos  
    print(f"\n⚙️  Extracting frames from fake videos...")
    total_fake_frames = 0
    for video_path in tqdm(fake_videos, desc="Fake videos"):
        extracted = extract_frames(
            video_path,
            os.path.join(FRAMES_OUTPUT, "fake"),
            num_frames=FRAMES_PER_VIDEO
        )
        total_fake_frames += extracted
    
    print(f"\n✅ Frame extraction complete!")
    print(f"   Real frames: {total_real_frames}")
    print(f"   Fake frames: {total_fake_frames}")
    print(f"   Total frames: {total_real_frames + total_fake_frames}")
    
    # Organize into train/val
    organize_dataset()
    
    print(f"\n📁 Dataset ready at: {FRAMES_OUTPUT}")
    print(f"   Train: {FRAMES_OUTPUT}/train/")
    print(f"   Val: {FRAMES_OUTPUT}/val/")
    print(f"\n🚀 Next step: python model/src/finetune_faceforensics.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
