import os
import sys

def inspect_dataset(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("❌ Path does not exist.")
        return

    video_exts = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
    image_exts = ('.jpg', '.jpeg', '.png', '.webp')
    
    v_count = 0
    i_count = 0
    
    file_list = []
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(video_exts):
                v_count += 1
                if len(file_list) < 5: file_list.append(os.path.join(root, f))
            elif f.lower().endswith(image_exts):
                i_count += 1
                if len(file_list) < 5: file_list.append(os.path.join(root, f))

    print(f"\n--- Recursive Summary for {os.path.basename(path)} ---")
    print(f"Total Videos: {v_count}")
    print(f"Total Images: {i_count}")
    print("Sample Files:")
    for f in file_list:
        print(f" - {os.path.basename(f)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_dataset.py <path>")
    else:
        inspect_dataset(sys.argv[1])
