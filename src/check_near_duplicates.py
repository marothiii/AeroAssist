from pathlib import Path
from itertools import combinations

import imagehash
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Smaller distance = more visually similar.
# We use a conservative threshold so we flag likely near-duplicates
# without treating every similar airport scene as a duplicate.
PHASH_THRESHOLD = 6


def get_images():
    return sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def calculate_phash(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return imagehash.phash(image)


def main():
    print("AeroAssist near-duplicate image check")
    print("------------------------------------")

    image_paths = get_images()
    print(f"Images found: {len(image_paths)}")

    hashes = {}

    for path in image_paths:
        try:
            hashes[path] = calculate_phash(path)
        except Exception as error:
            print(f"Could not hash {path}: {error}")

    flagged_pairs = []

    for path_a, path_b in combinations(hashes.keys(), 2):
        distance = hashes[path_a] - hashes[path_b]

        if distance <= PHASH_THRESHOLD:
            flagged_pairs.append((distance, path_a, path_b))

    flagged_pairs.sort(key=lambda item: item[0])

    print(f"\nPairs with pHash distance <= {PHASH_THRESHOLD}: {len(flagged_pairs)}")
    print("------------------------------------")

    if not flagged_pairs:
        print("No likely near-duplicates detected.")
    else:
        for distance, path_a, path_b in flagged_pairs:
            relative_a = path_a.relative_to(PROJECT_DIR)
            relative_b = path_b.relative_to(PROJECT_DIR)

            print(f"\nDistance: {distance}")
            print(f"  {relative_a}")
            print(f"  {relative_b}")

    print("\n------------------------------------")
    print("Near-duplicate check complete.")


if __name__ == "__main__":
    main()