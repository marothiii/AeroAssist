from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "aeroassist_dataset_samples.png"

# Fixed seed means the evidence figure can be reproduced.
random.seed(42)

# Friendly names used only for the report figure.
DISPLAY_NAMES = {
    "gate_sign": "Gate Sign",
    "baggage_claim": "Baggage Claim",
    "security": "Security",
    "check_in": "Check-in",
    "checkin": "Check-in",
    "information": "Information",
    "accessibility": "Accessibility",
    "family": "Family Facilities",
    "lounge": "Lounge",
    "restaurant": "Restaurant",
    "transport": "Transport",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

class_folders = sorted(
    folder
    for folder in IMAGE_DIR.iterdir()
    if folder.is_dir()
)

print("Classes found:", len(class_folders))

for folder in class_folders:
    count = len([
        path for path in folder.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    ])
    print(f"{folder.name}: {count} images")

# One reproducibly selected image from every class.
selected = []

for folder in class_folders:
    images = sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        continue

    selected.append(
        (folder.name, random.choice(images))
    )

if not selected:
    raise RuntimeError("No images were found.")

cols = 5
rows = (len(selected) + cols - 1) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(15, 3.6 * rows),
)

axes = axes.flatten()

for ax, (class_name, image_path) in zip(axes, selected):
    image = Image.open(image_path).convert("RGB")

    # Standardise the display crop without modifying the dataset file.
    image = ImageOps.fit(
        image,
        (700, 450),
        method=Image.Resampling.LANCZOS,
    )

    ax.imshow(image)
    ax.axis("off")

    display_name = DISPLAY_NAMES.get(
        class_name,
        class_name.replace("_", " ").title(),
    )

    ax.set_title(
        display_name,
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # Small provenance label makes this look like genuine
    # project evidence rather than a stock-image collage.
    ax.text(
        0.5,
        -0.06,
        image_path.name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

for ax in axes[len(selected):]:
    ax.axis("off")

fig.suptitle(
    "AeroAssist Visual Dataset",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

fig.text(
    0.5,
    0.015,
    "One reproducibly selected example from each airport scene category",
    ha="center",
    fontsize=10,
)

plt.tight_layout(
    rect=[0.02, 0.04, 0.98, 0.94]
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nFigure saved to:")
print(OUTPUT_PATH)
