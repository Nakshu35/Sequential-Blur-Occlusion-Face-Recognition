# Sequential Blur-Aware Face Recognition with Occlusion Handling

## Overview

This project presents a **sequential degradation-aware face recognition approach**, which can handle **motion blurs and occlusions** in face images.

Unlike the existing face recognition approaches, which handle degradations simultaneously, this work proposes a **structured face recognition pipeline**, which consists of:

1. Blur Removal (Blur Decoder)
2. Occlusion Handling (Mask / Restoration)
3. Face Recognition (ArcFace / CosFace)

The main motivation behind this work is:

> Restoring the quality of the images (deblurring) before handling occlusions improves the feature representation and recognition performance.

---

## Abstract

Face recognition systems may experience considerable performance degradation in the presence of real-world degradations such as motion blur and occlusion. Current methods tackle the above challenges separately or in an integrated manner. However, this may not be the best approach because of the inherent differences between the degradations.

We present a **sequential degradation-aware framework** in this paper. The proposed framework explicitly addresses the above degradations in a structured manner. The proposed framework consists of a **lightweight blur decoder**, which removes the effects of motion blur and occlusion from the face images. The processed images are then fed into the **occlusion handling module**, which removes the occlusion from the images. The processed images are then used for face recognition using margin-based loss functions such as ArcFace and CosFace.

Our proposed approach enables the above components to focus on specific degradations. The performance of the proposed approach is demonstrated in the experimental section.

---

## Methodology

### Pipeline

```
Input Image (Blur + Occlusion)
↓
Blur Decoder (Motion-aware Residual CNN)
↓
Refined Image
↓
Occlusion Module (Mask / Restoration)
↓
Backbone Network
↓
Embedding
↓
ArcFace / CosFace  Loss
↓
Final Prediction
```

---

## Key Components

### 1. Blur Decoder

A lightweight CNN designed to reduce motion blur.

**Core Idea:**

```
I_deblur = I_input + f(I_input)
```

**Features:**

- Residual learning (preserves identity)
- Direction-aware convolutions:
  - Horizontal (1×5)
  - Vertical (5×1)
  - Standard (3×3)
- Lightweight architecture (efficient for real-time use)

**Goal:**

- Recover structural details (edges, facial patterns)
- Improve downstream recognition performance

---

### 2. Occlusion Module

Two possible approaches:

#### (A) Mask-based (FROM-style)

- Learns a mask to suppress occluded regions
- Operates in feature space

#### (B) Restoration-based

- Reconstructs occluded regions
- Produces refined image before recognition

---

### 3. Recognition Head

Supports margin-based loss functions:

- ArcFace
- CosFace

These improve embedding separability and recognition accuracy.

---

## Loss Function

```
L_final = L_id + λ * L_blur + β * L_mask
```

Where:

- `L_id` → Identity loss (ArcFace / CosFace / SphereFace)
- `L_blur` → Deblurring loss (L1 + SSIM)
- `L_mask` → Occlusion loss (BCE / consistency)

---

## Training Strategy

1. Train Blur Decoder independently
2. Train Occlusion Module
3. Integrate full pipeline
4. Fine-tune end-to-end

---

## Experiments

### Evaluation Conditions

- Clean images
- Blur only
- Occlusion only
- Blur + Occlusion

---

### Baselines

- Recognition only (no degradation handling)
- Blur-only model
- Occlusion-only model
- Joint model (single network handling both)

---

### Metrics

- Accuracy
- TAR @ FAR
- ROC Curve
- PSNR / SSIM (for blur stage)
- Inference time

---

## Expected Results

- Sequential (Blur → Occlusion) outperforms:
  - individual modules
  - joint models

---

## Key Insight

Rather than treating degradations in the same manner, the research demonstrates that:

> "Handling blur before occlusion results in better feature quality and improved recognition performance."

---

## Author

- Nakshatra Gandhe  
- Priyansh Balwani  
- Harsh Shah

---

## License
MIT License

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Framework](https://img.shields.io/badge/framework-PyTorch-red)