---
title: Deepfake Detection Model
emoji: ��️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: backend/app.py
app_port: 7860
pinned: false
---

# DeepGuard: AI-Powered Deepfake Detection

![Accuracy](https://img.shields.io/badge/Accuracy-96.97%25-brightgreen)
![Model](https://img.shields.io/badge/Model-Mark--V-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**DeepGuard** is a state-of-the-art, privacy-focused tool designed to detect AI-generated images with **96.97% accuracy**. It runs entirely on your local machine using a Hybrid Multi-Branch Neural Network.

![Radar Chart](model/visualizations/6_model_radar_comparison.png)

## 🚀 Quick Links

*   **[📝 Overview & How it Works](Documentation/OVERVIEW.md)**
*   **[⚡ Getting Started Guide](Documentation/GETTING_STARTED.md)**
*   **[🏗️ System Architecture](Documentation/ARCHITECTURE.md)**
*   **[🔒 Security & Privacy](Documentation/SECURITY.md)**
*   **[🛠️ Backend API](Documentation/BACKEND.md)**
*   **[🎨 Frontend Guide](Documentation/FRONTEND.md)**

## 🏆 Current Performance (Mark-V)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **96.97%** | Tested on Universal Dataset |
| **Reliability** | **Generative** | Wide coverage of generation methods |
| **FPS** | **~25** | Real-time analysis on GPU |

## 📦 Features

*   **Multi-Branch Detection**: Combines RGB, Frequency (FFT), Patch analysis, and Vision Transformers.
*   **Defense-in-Depth**: Automatically detects C2PA credentials and invisible watermarks (Stable Diffusion).
*   **Local-First**: No data ever leaves your computer.
*   **History Tracking**: Keep a local log of your scans.

## 💻 Quick Install

```bash
git clone https://github.com/your-username/DeepGuard.git
cd DeepGuard/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements_web.txt
python app.py
```

Open `http://localhost:7860` in your browser.

---

For full documentation, please visit the **[Documentation Folder](Documentation/)**.

