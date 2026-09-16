import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_DIR = os.path.join(BASE_DIR, "images")

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "image_manifest.csv"
)


CLASSES = [
    "gate_sign",
    "baggage_claim",
    "security",
    "check_in",
    "lounge",
    "restaurant",
    "information",
    "transport",
    "accessibility",
    "family_facility",
]


IMAGES_PER_CLASS = 20


print("AeroAssist visual dataset")
print("-------------------------")
print("Classes:", len(CLASSES))
print("Images per class:", IMAGES_PER_CLASS)
print("Target images:", len(CLASSES) * IMAGES_PER_CLASS)

for class_name in CLASSES:
    class_path = os.path.join(
        IMAGE_DIR,
        class_name
    )

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    print(
        f"{class_name}: {len(image_files)}/{IMAGES_PER_CLASS}"
    )