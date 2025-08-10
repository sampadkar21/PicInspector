# PicInspector  
📸 AI-Powered Image Quality Analysis Tool  

A multi-modal system that automatically evaluates images through:  
- ✔ Quantitative scoring (Sharpness, Noise, Color, etc.)  
- ✔ Scene understanding (Landscape, Portrait, Night, etc.)  
- ✔ Natural language feedback ("This image has motion blur...")  

---

## Overview

PicInspector is a comprehensive tool for evaluating image quality through AI-driven caption generation and quantitative scoring. This repository contains Jupyter notebooks demonstrating training and inference pipelines for two key components:

- **Caption Generation:** Uses a fine-tuned vision-language model (based on SmolLM) to generate detailed textual descriptions of image quality aspects like sharpness, noise, and overall aesthetics.  
- **Score Generation:** Employs a multi-task model built on DINOv2 to predict numerical scores for quality attributes (e.g., MOS, brightness) and classify scene categories (e.g., landscape, indoor).  

The project leverages datasets like SPAQ-10K for training and evaluation, including MOS scores, attribute labels, and scene categories. Evaluation includes metrics like ROUGE for captions and loss functions for regression/classification tasks.

**Goals:** Provide an accessible way to assess image quality for applications in photography, content moderation, or AI-assisted editing.

---

## Model Architectures

### 1. Caption Generation Model (Based on Huggingface SmolVLM)

- Fine-tuned vision-language model generating quality descriptions.  
- Processes images with a vision encoder and fuses features into a language model decoder for text generation.  

**Architecture:**  
Image input → Vision Transformer Encoder → Text embeddings + image features → Language Model Decoder → Detailed captions describing image quality  

### 2. Score and Scene Prediction Model (Multi-Head DINOv2)

- Multi-task heads on top of DINOv2 backbone for:  
  - Regression of quality scores (6 numerical outputs)  
  - Classification of scene categories (9 classes)  

**Architecture:**  
Image input → DINOv2 Encoder (feature pooling) →  
&nbsp;&nbsp;&nbsp;&nbsp;→ Regression Head (linear layers) for scores  
&nbsp;&nbsp;&nbsp;&nbsp;→ Classification Head (linear layers + softmax) for scenes  

---

## Installation

To run the notebooks, install the required dependencies:

```bash
pip install torch transformers accelerate bitsandbytes evaluate rouge_score pandas matplotlib seaborn scikit-learn
```

### Example Inference Workflow
- Load an image (from URL or file).
- Run the caption generation model to get a quality description.
- Run the score prediction model to get numerical quality attributes and scene classification.

Example output:

```python
Description: "The image has moderate sharpness with noticeable noise in low-light areas..."  
Scores: MOS: 65/100, Brightness: 70/100, etc.  
Scene: Landscape (probability 0.85)
```
