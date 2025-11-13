#!/usr/bin/env python3
"""
Debug detection - show all raw detections before NMS
"""

import cv2
import torch
import yaml
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

print("Config loaded:")
print(f"  BOX_THRESHOLD: {config['config']['BOX_THRESHOLD']}")
print(f"  TEXT_THRESHOLD: {config['config']['TEXT_THRESHOLD']}")

# Load sample image
sample_image = "/home/serverai/ltdoanh/LayoutGeneration/data/samples/keyframe4check/39778_keyframes/keyframes_preview.png"
image = Image.open(sample_image).convert("RGB")
print(f"\nImage: {image.size}")

# Load Grounding DINO
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"

print(f"\nLoading Grounding DINO from {model_id}...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Run detection
text_prompt = config['config']['TEXT_PROMPT']
print(f"Text prompt: '{text_prompt}'")

inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

print(f"\nRaw model outputs:")
print(f"  logits shape: {outputs.logits.shape}")
print(f"  boxes shape: {outputs.pred_boxes.shape}")

# Post-process with threshold
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=config['config']['BOX_THRESHOLD'],
    target_sizes=[image.size[::-1]]
)

print(f"\nPost-processed results (threshold={config['config']['BOX_THRESHOLD']}):")
print(f"  Boxes: {len(results[0]['boxes'])}")
print(f"  Scores: {results[0]['scores']}")
print(f"  Labels: {results[0]['labels']}")

if len(results[0]['boxes']) > 0:
    print(f"\nDetailed detections:")
    for idx, (box, score, label) in enumerate(zip(results[0]['boxes'], results[0]['scores'], results[0]['labels']), 1):
        print(f"  [{idx}] score={score:.4f}, box={box.tolist()}")
else:
    print("\n❌ No detections!")

# Try lower threshold
print(f"\n\n{'='*70}")
print(f"Trying with LOWER threshold (0.01):")
print(f"{'='*70}")

results_loose = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.01,
    target_sizes=[image.size[::-1]]
)

print(f"With threshold=0.01: {len(results_loose[0]['boxes'])} detections")
if len(results_loose[0]['boxes']) > 0:
    for idx, (box, score, label) in enumerate(zip(results_loose[0]['boxes'], results_loose[0]['scores'], results_loose[0]['labels']), 1):
        print(f"  [{idx}] score={score:.4f}")
