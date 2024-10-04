# Histopathological Cancer Detection

Deep learning pipeline for detecting and grading cancer from histopathological whole-slide images, served via TensorFlow Serving with GPU acceleration. Built as a research project at VIT Chennai.

## The Problem

Histopathology is the gold standard for cancer diagnosis — a pathologist examines stained tissue slides under a microscope to identify malignant cells. The challenge is scale: each whole-slide image can run into gigapixels, and manual examination is slow, expensive, and subject to inter-observer variability. An automated system that flags suspicious regions can speed up diagnosis and act as a second opinion.

## Approach

**Patch-based classification** — the standard approach for whole-slide images:

1. Tile the slide into smaller patches (e.g., 256×256 or 512×512 pixels)
2. Run each patch through a CNN to get a local prediction (cancerous / non-cancerous)
3. Aggregate patch-level predictions across the slide to produce a slide-level diagnosis

This makes the problem tractable — instead of processing a 100,000×100,000 image at once, you process thousands of smaller tiles that fit in GPU memory.

**Tasks covered:**
- Binary classification: cancerous vs. non-cancerous tissue
- Multi-class grading: staging cancer severity from the slide

**Deployment:** TensorFlow Serving for model hosting and REST API inference. GPU acceleration (CUDA) required for training and efficient batch inference on patch sets.

## Repository Contents

| File | Description |
|---|---|
| `cancer research paper.pdf` | Research paper on histopathological cancer detection |
| `7_TensorFlow.pdf` | TensorFlow implementation documentation |
| `ppt.pdf` | Project presentation slides |

## Stack

Python · TensorFlow · Keras · TensorFlow Serving · CUDA / GPU