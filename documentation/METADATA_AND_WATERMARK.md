# Metadata & Invisible Watermark Detection Strategy

## 1. Executive Summary

This document outlines the strategy for enhancing the Deepfake Detection System by adding "Defense-in-Depth" layers. We will check for **Metadata/Content Credentials (C2PA)** and **Invisible Watermarks** before passing media to the deep learning model.

**Recommendation:** Proceed with implementation.
**Reasoning:** These checks provide 100% certainty for compliant AI content (DALL-E, Firefly, Stable Diffusion), drastically reducing false negatives for high-quality generations and providing clear explanations to users.

---

## 2. Analysis: Pros & Cons

| Feature | Pros | Cons |
| :--- | :--- | :--- |
| **Metadata Check** | • **Zero False Positives:** If C2PA says "DALL-E", it IS DALL-E.<br>• **Speed:** Instant check (milliseconds).<br>• **Low Cost:** Saves GPU compute.<br>• **Explainability:** Specific tool identification. | • **Easily Stripped:** Screenshots/editing remove it.<br>• **Voluntary:** Not all generators use it.<br>• **Malicious Actors:** Will strip it intentionally. |
| **Watermark Check** | • **Robustness:** Survives standard edits better than metadata.<br>• **Specificity:** Can detect Stable Diffusion/Midjourney specifically. | • **Complexity:** Requires specialized decoders.<br>• **Reliability:** Not perfect; can be distorted.<br>• **Proprietary:** Some (like SynthID) are API-only. |

---

## 3. Technical Implementation Plan

The system will implement a pipeline approach in `backend/app.py`:

### Layer 1: Metadata & Credentials (The "Passport" Check)
**Goal:** Check file headers for C2PA manifests and Exif/XMP tags.
*   **Library:** `c2pa-python` (Official Content Authenticity Initiative).
*   **Library:** `ExifRead` (For standard legacy tags).
*   **Targets:** Adobe Firefly, DALL-E 3, Bing Image Creator.

### Layer 2: Invisible Watermarks (The "Fingerprint" Check)
**Goal:** Detect hidden signal patterns in pixels.
*   **Library:** `invisible-watermark` (Python).
*   **Targets:** Stable Diffusion (`sd_private` watermark), potentially Midjourney.

### Layer 3: Visual Deepfake Detector (Existing)
**Goal:** Deep Learning (CNN/ViT) analysis.
*   **Role:** Final arbiter for files that pass previous checks.

---

## 4. Architecture Changes

### Backend Components

1.  **`backend/requirements_web.txt`**
    *   Add `c2pa-python`, `invisible-watermark`, `ExifRead`.

2.  **`backend/src/metadata_checker.py`**
    *   `check_metadata(filepath) -> dict`: Returns `{detected: bool, source: str, method: 'C2PA'|'EXIF'}`.

3.  **`backend/src/watermark_checker.py`**
    *   `check_watermarks(filepath) -> dict`: Returns `{detected: bool, type: 'Stable Diffusion'}`.

4.  **`backend/app.py`**
    *   Modify `/predict` endpoint to run these checks sequentially.
    *   If detected, include specific flags in the JSON response.

### Frontend Integration

*   **UI Update:** Display distinct badges/alerts for "Digital Signature Detected" or "Watermark Detected" alongside the AI probability score.

---

## 5. Verification Strategy

1.  **Positive Control:** Upload raw images from DALL-E 3 and Stable Diffusion. Verify accurate flagging.
2.  **Negative Control:** Upload real camera photos. Verify no false flags.
3.  **Adversarial Control:** Upload a "stripped" AI image (screenshot). Verify it falls through to the Visual Model.
