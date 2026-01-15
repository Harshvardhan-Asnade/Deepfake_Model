# FaceForensics++ Integration Guide

## 🎯 Overview

This guide will help you download FaceForensics++ and fine-tune your existing DeepGuard model to improve detection accuracy on video-based deepfakes.

---

## 📋 Requirements

Before starting:
- ✅ **70-500GB free disk space** (depending on version)
  - c23 (compressed): ~40GB
  - Raw: ~500GB
- ✅ **Git installed**: `git --version`
- ✅ **Python 3.8+**
- ✅ **Existing trained model**: `best_model.safetensors`

---

## 🚀 Step-by-Step Instructions

### **Step 1: Request Dataset Access** (1-2 days wait)

FaceForensics++ requires academic/research approval.

1. Visit: https://github.com/ondyari/FaceForensics
2. Scroll to **"Access"** section
3. Fill out the registration form
4. Wait for approval email (usually 1-2 days)
5. You'll receive download credentials

---

### **Step 2: Run Download Helper**

```bash
cd "/Users/harshvardhan/Developer/Deepfake Project /Morden Detections system"
python model/download_faceforensics.py
```

This script will:
- ✅ Create directory structure
- ✅ Clone FaceForensics++ repository
- ✅ Show download instructions
- ✅ Guide you through the process

**Dataset will be downloaded to:**
```
/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++
```

---

### **Step 3: Download Videos** (After approval)

Once you receive credentials:

```bash
cd "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++/FaceForensics-repo"

# Download c23 version (RECOMMENDED - 38GB)
python download-FaceForensics.py \
  -d FaceForensics++ \
  -c c23 \
  -t videos \
  -o "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++"
```

**Download time**: 2-6 hours (depending on internet speed)

---

### **Step 4: Extract Frames**

```bash
cd "/Users/harshvardhan/Developer/Deepfake Project /Morden Detections system"
python model/extract_ff_frames.py
```

This will:
- ✅ Extract 10 frames per video (evenly spaced)
- ✅ Organize into `real/` and `fake/` folders
- ✅ Create 80/20 train/val split
- ✅ Save frames as high-quality JPG

**Expected output:**
- Real frames: ~10,000 images
- Fake frames: ~40,000 images (4 manipulation methods)
- Total: ~50,000 images

**Extraction time**: 1-3 hours

---

### **Step 5: Fine-Tune Your Model**

```bash
cd "/Users/harshvardhan/Developer/Deepfake Project /Morden Detections system"
python model/src/finetune_faceforensics.py
```

**Fine-tuning settings:**
- Learning Rate: 5e-6 (very low - preserves existing knowledge)
- Epochs: 3
- Batch Size: Your config setting (default 32)

**Training time**: 4-8 hours (depends on GPU/CPU)

---

### **Step 6: Evaluate Results**

```bash
# Test the fine-tuned model
python model/evaluate_custom.py

# Compare with old model
python model/compare_models.py
```

---

## 📊 Expected Results

### **Before Fine-tuning:**
- Your current model: 99.15% on existing datasets
- FaceForensics++ (untested): Unknown performance

### **After Fine-tuning:**
- Existing datasets: ~99% (should maintain)
- FaceForensics++: **85-95%** (new capability!)
- Video deepfakes: Significantly improved detection

---

## 🔄 Quick Reference Commands

```bash
# Full workflow
cd "/Users/harshvardhan/Developer/Deepfake Project /Morden Detections system"

# 1. Setup (one-time)
python model/download_faceforensics.py

# 2. Download videos (after approval - run in FF repo)
cd "/Users/harshvardhan/Developer/Deepfake Project /DataSet/FaceForensics++/FaceForensics-repo"
python download-FaceForensics.py -d FaceForensics++ -c c23 -t videos -o "../"

# 3. Extract frames
cd "/Users/harshvardhan/Developer/Deepfake Project /Morden Detections system"
python model/extract_ff_frames.py

# 4. Fine-tune
python model/src/finetune_faceforensics.py

# 5. Evaluate
python model/evaluate_custom.py
```

---

## 📁 Directory Structure

After completion:

```
/Users/harshvardhan/Developer/Deepfake Project /DataSet/
└── FaceForensics++/
    ├── FaceForensics-repo/          (Git repository)
    ├── videos/                       (Downloaded videos)
    │   ├── original_sequences/
    │   └── manipulated_sequences/
    └── frames/                       (Extracted frames)
        ├── train/
        │   ├── real/
        │   └── fake/
        └── val/
            ├── real/
            └── fake/
```

---

## ⚠️ Troubleshooting

### "Videos not found"
- Ensure videos are downloaded to correct location
- Check directory structure matches FaceForensics++ format

### "No frames extracted"
- Install opencv: `pip install opencv-python`
- Check video codecs: `ffmpeg -version`

### "Out of memory during training"
- Reduce batch size in `model/src/config.py`
- Use gradient checkpointing (advanced)

---

## 🎯 What's Next?

After fine-tuning on FaceForensics++, consider:

1. **Celeb-DF v2** - Even harder deepfakes
2. **DFDC** - Massive diversity
3. **Temporal model** - Analyze video sequences (not just frames)

---

## ✅ Checklist

- [ ] Requested FaceForensics++ access
- [ ] Received approval email
- [ ] Downloaded c23 videos (~38GB)
- [ ] Extracted frames (~50k images)
- [ ] Fine-tuned model (3 epochs)
- [ ] Evaluated results
- [ ] Updated best_model if improved

---

**Questions?** Check the [FAQ](documentation/FAQ.md) or review the scripts' output messages.
