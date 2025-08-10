# PicInspector
📸 AI-Powered Image Quality Analysis Tool  
A multi-modal system that automatically evaluates images through: 
- ✔ Quantitative scoring (Sharpness, Noise, Color, etc.)
- ✔ Scene understanding (Landscape, Portrait, Night, etc.) 
- ✔ Natural language feedback ("This image has motion blur...")

Overview:\n
PicInspector is a comprehensive tool for evaluating image quality through AI-driven caption generation and quantitative scoring. This repository contains Jupyter notebooks that demonstrate the training and inference pipelines for two key components: generating detailed quality descriptions using a fine-tuned language-vision model and predicting quality scores (e.g., brightness, sharpness) along with scene classification using a multi-task DINOv2-based model. Visualizations of model architectures are included to illustrate the underlying structures.
Table of Contents

Project Description
Model Architectures
Installation
Usage
Notebook Descriptions
Contributing
License

Project Description
This project focuses on image quality inspection ("PicInspector") by combining multimodal AI techniques:

Caption Generation: Uses a fine-tuned vision-language model (based on SmolLM) to generate detailed textual descriptions of image quality aspects like sharpness, noise, and overall aesthetics.
Score Generation: Employs a multi-task model built on DINOv2 to predict numerical scores for quality attributes (e.g., MOS, brightness) and classify scene categories (e.g., landscape, indoor).
Datasets: Leverages datasets like SPAQ-10K for training and evaluation, including MOS scores, attribute labels, and scene categories.
Evaluation: Includes metrics like ROUGE for captions and loss functions for regression/classification tasks.
Goals: To provide an accessible way to assess image quality for applications in photography, content moderation, or AI-assisted editing.

The notebooks cover end-to-end workflows from data preparation and model training to inference, with saved models for reuse.
Model Architectures
The project utilizes two primary models, with visualizations below for clarity:
1. Caption Generation Model (Based on SmolLM)
This is a fine-tuned vision-language model for generating quality descriptions. It processes images and prompts to output detailed captions.

Key Components: Vision encoder for image features, fused with a language model for text generation.
Architecture Diagram:
Failed to load imageView link

(Description: The diagram shows the image input feeding into a vision transformer, concatenated with text embeddings in the LLM decoder, leading to generated captions.)

2. Score and Scene Prediction Model (Multi-Task DINOv2)
A multi-task head on top of DINOv2 for regression (quality scores) and classification (scene categories).

Key Components: DINOv2 as the backbone encoder, followed by separate heads for regression (6 outputs) and classification (9 classes).
Architecture Diagram:
Failed to load imageView link

(Description: The diagram illustrates the DINOv2 encoder pooling features, branching into a regression head (linear layers for scores) and a classification head (linear layers with softmax for scenes).)

These diagrams were created using tools like Draw.io or Lucidchart and saved as PNGs in the repository for easy reference.
Installation
To run the notebooks, install the required dependencies in a Python environment (e.g., via pip or conda). Key packages include:
text!pip install torch transformers accelerate bitsandbytes evaluate rouge_score pandas matplotlib seaborn sklearn

Ensure GPU access for faster training (e.g., via Kaggle or Colab).
Download models from Hugging Face (e.g., "facebook/dinov2-base") or use the provided saved models.

Usage

Clone the repository:
textgit clone https://github.com/yourusername/picinspector.git
cd picinspector

Open notebooks in Jupyter:
textjupyter notebook

For inference:

Load an image URL or file.
Run the caption generation to get a quality description.
Use the score model to get numerical attributes and scene predictions.
Example output:

Description: "The image has moderate sharpness with noticeable noise in low-light areas..."
Scores: MOS: 65/100, Brightness: 70/100, etc.
Scene: Landscape (probability 0.85)


Notebook Descriptions

PicInspector-part1-caption_generation.ipynb: Focuses on training and evaluating the caption generation model. Includes data loading, fine-tuning with LoRA, ROUGE evaluation, and model saving.
PicInspector-part2-scores_generation.ipynb: Handles score prediction and scene classification. Covers data preprocessing, multi-task training on DINOv2, and validation.
inferencing.ipynb: Demonstrates end-to-end inference using the trained models. Includes functions for downloading images, generating descriptions, assessing scores, and visualizing results (e.g., radar charts for attributes).
