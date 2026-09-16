from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import AutoProcessor

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_PATH = (
    PROJECT_DIR
    / "images"
    / "baggage_claim"
    / "baggage_claim_001.jpg"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "aeroassist_preprocessing_evidence.png"
)

# Load the real AeroAssist image
raw_image = Image.open(IMAGE_PATH)

original_mode = raw_image.mode
original_size = raw_image.size

rgb_image = ImageOps.exif_transpose(
    raw_image
).convert("RGB")

# Use the same Hugging Face CLIP processor as vision_pipeline.py
print("Loading AeroAssist CLIP processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

processed = processor(
    images=rgb_image,
    return_tensors="pt",
)

pixel_values = processed["pixel_values"]

# Terminal evidence
print("\nAEROASSIST PREPROCESSING EVIDENCE")
print("--------------------------------")
print("Source image:", IMAGE_PATH.name)
print("Original size:", original_size)
print("Original mode:", original_mode)
print("Converted mode:", rgb_image.mode)
print("Processor: CLIP ViT-B/32")
print("Pixel tensor shape:", tuple(pixel_values.shape))
print("Tensor dtype:", pixel_values.dtype)
print(
    "Tensor value range:",
    f"{pixel_values.min().item():.3f}",
    "to",
    f"{pixel_values.max().item():.3f}",
)
print("Model-ready: True")

# Report figure
fig, ax = plt.subplots(figsize=(11, 7))

ax.imshow(rgb_image)
ax.axis("off")

ax.set_title(
    "AeroAssist Image Preprocessing Evidence",
    fontsize=18,
    fontweight="bold",
    pad=16,
)

details = (
    f"Input: {IMAGE_PATH.name}\n"
    f"Original image: {original_size[0]} x {original_size[1]} px | "
    f"{original_mode}\n"
    f"Converted colour mode: RGB\n"
    "Processor: CLIP ViT-B/32\n"
    f"Model-ready tensor: {tuple(pixel_values.shape)}\n"
    f"Pipeline: raw image -> EXIF correction -> RGB -> "
    f"CLIP processor -> pixel tensor"
)

fig.text(
    0.5,
    0.04,
    details,
    ha="center",
    va="bottom",
    fontsize=11,
)

plt.subplots_adjust(
    bottom=0.23,
    top=0.88,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print("\nFigure saved to:")
print(OUTPUT_PATH)
