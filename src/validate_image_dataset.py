from pathlib import Path
from collections import Counter
import hashlib

import pandas as pd
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
MANIFEST_PATH = PROJECT_DIR / "data" / "image_manifest.csv"

EXPECTED_CLASSES = [
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

EXPECTED_PER_CLASS = 20
EXPECTED_TOTAL = 200


def get_image_path(row):
    return IMAGE_DIR / row["class_label"] / row["filename"]


def file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():
    print("AeroAssist image dataset validation")
    print("-----------------------------------")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Manifest rows: {len(manifest)}")

    # --------------------------------------------------
    # 1. Manifest checks
    # --------------------------------------------------

    duplicate_ids = manifest[
        manifest["image_id"].duplicated(keep=False)
    ]

    duplicate_filenames = manifest[
        manifest["filename"].duplicated(keep=False)
    ]

    duplicate_sources = manifest[
        manifest["source_reference"].duplicated(keep=False)
    ]

    print("\nManifest integrity")
    print("------------------")
    print(f"Duplicate image IDs: {len(duplicate_ids)}")
    print(f"Duplicate filenames: {len(duplicate_filenames)}")
    print(f"Duplicate source references: {len(duplicate_sources)}")

    # --------------------------------------------------
    # 2. Class balance
    # --------------------------------------------------

    print("\nClass distribution")
    print("------------------")

    class_counts = Counter(manifest["class_label"])

    for class_name in EXPECTED_CLASSES:
        count = class_counts.get(class_name, 0)
        print(f"{class_name}: {count}/{EXPECTED_PER_CLASS}")

    # --------------------------------------------------
    # 3. Check every manifest image
    # --------------------------------------------------

    missing_files = []
    corrupt_files = []
    dimensions = []
    formats = []
    hashes = {}

    for _, row in manifest.iterrows():
        path = get_image_path(row)

        if not path.exists():
            missing_files.append(str(path))
            continue

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)

                width, height = image.size

                dimensions.append(
                    {
                        "image_id": row["image_id"],
                        "filename": row["filename"],
                        "width": width,
                        "height": height,
                    }
                )

                formats.append(image.format)

                # Force Pillow to actually decode the image.
                image.load()

            image_hash = file_hash(path)

            if image_hash not in hashes:
                hashes[image_hash] = []

            hashes[image_hash].append(row["filename"])

        except Exception as error:
            corrupt_files.append(
                (row["filename"], str(error))
            )

    # --------------------------------------------------
    # 4. Exact duplicate files
    # --------------------------------------------------

    exact_duplicates = {
        hash_value: filenames
        for hash_value, filenames in hashes.items()
        if len(filenames) > 1
    }

    # --------------------------------------------------
    # 5. Results
    # --------------------------------------------------

    print("\nFile integrity")
    print("--------------")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupt/unreadable files: {len(corrupt_files)}")
    print(f"Exact duplicate image files: {len(exact_duplicates)}")

    if formats:
        print(f"Detected formats: {sorted(set(formats))}")

    if dimensions:
        dimension_df = pd.DataFrame(dimensions)

        print(
            "Width range:",
            f"{dimension_df['width'].min()}–"
            f"{dimension_df['width'].max()} px"
        )

        print(
            "Height range:",
            f"{dimension_df['height'].min()}–"
            f"{dimension_df['height'].max()} px"
        )

    if missing_files:
        print("\nMissing:")
        for path in missing_files:
            print(" -", path)

    if corrupt_files:
        print("\nCorrupt/unreadable:")
        for filename, error in corrupt_files:
            print(f" - {filename}: {error}")

    if exact_duplicates:
        print("\nExact duplicates:")
        for filenames in exact_duplicates.values():
            print(" -", " <-> ".join(filenames))

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    class_balance_ok = all(
        class_counts.get(class_name, 0) == EXPECTED_PER_CLASS
        for class_name in EXPECTED_CLASSES
    )

    passed = (
        len(manifest) == EXPECTED_TOTAL
        and len(duplicate_ids) == 0
        and len(duplicate_filenames) == 0
        and len(duplicate_sources) == 0
        and class_balance_ok
        and len(missing_files) == 0
        and len(corrupt_files) == 0
        and len(exact_duplicates) == 0
    )

    print("\n-----------------------------------")

    if passed:
        print("RESULT: BASIC DATASET VALIDATION PASSED")
    else:
        print("RESULT: DATASET VALIDATION NEEDS ATTENTION")


if __name__ == "__main__":
    main()