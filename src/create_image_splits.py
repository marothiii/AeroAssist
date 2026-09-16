from pathlib import Path
import random
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "data" / "image_manifest.csv"

RANDOM_SEED = 42

# Different photographs that come from the same airport/scene.
# Keeping each group in one split reduces evaluation leakage.
RELATED_GROUPS = [
    # Baggage claim
    ["IMG021", "IMG028"],          # Zurich
    ["IMG023", "IMG039"],          # Port Columbus
    ["IMG027", "IMG038"],          # Taoyuan

    # Security
    ["IMG051", "IMG052"],          # Brisbane
    ["IMG058", "IMG059", "IMG060"],  # TSA-related set

    # Transport
    ["IMG141", "IMG156"],          # Brisbane railway
    ["IMG146", "IMG157", "IMG158"],  # Southend railway
    ["IMG142", "IMG160"],          # Brisbane taxi

    # Accessibility
    ["IMG161", "IMG162"],          # Brisbane Changing Places

    # Family facilities
    ["IMG183", "IMG184", "IMG185"],  # Zurich playground
    ["IMG193", "IMG194"],          # Hello Kitty nursery
]

TARGETS = {
    "development": 14,
    "validation": 3,
    "test": 3,
}


def build_group_lookup():
    lookup = {}

    for group_number, group in enumerate(RELATED_GROUPS, start=1):
        group_name = f"group_{group_number:02d}"

        for image_id in group:
            if image_id in lookup:
                raise ValueError(
                    f"{image_id} appears in more than one related group."
                )

            lookup[image_id] = group_name

    return lookup


def split_class(class_df, group_lookup, rng):
    items = []
    used_ids = set()

    # Add manually related groups first.
    for group_name in sorted(set(group_lookup.values())):
        group_ids = [
            image_id
            for image_id, mapped_group in group_lookup.items()
            if mapped_group == group_name
        ]

        rows = class_df[class_df["image_id"].isin(group_ids)]

        if len(rows) > 0:
            ids = rows["image_id"].tolist()
            items.append(ids)
            used_ids.update(ids)

    # Every unrelated image acts as its own group.
    for image_id in class_df["image_id"]:
        if image_id not in used_ids:
            items.append([image_id])

    rng.shuffle(items)

    assignments = {}
    counts = {
        "development": 0,
        "validation": 0,
        "test": 0,
    }

    # Put larger related groups first so they remain intact.
    items.sort(key=len, reverse=True)

    for item in items:
        size = len(item)

        possible_splits = [
            split_name
            for split_name, target in TARGETS.items()
            if counts[split_name] + size <= target
        ]

        if not possible_splits:
            raise RuntimeError(
                f"Could not create exact split for class "
                f"{class_df['class_label'].iloc[0]}."
            )

        # Prefer the split with the largest remaining capacity.
        chosen_split = max(
            possible_splits,
            key=lambda split_name: (
                TARGETS[split_name] - counts[split_name]
            ),
        )

        for image_id in item:
            assignments[image_id] = chosen_split

        counts[chosen_split] += size

    if counts != TARGETS:
        raise RuntimeError(
            f"Unexpected split counts for "
            f"{class_df['class_label'].iloc[0]}: {counts}"
        )

    return assignments


def main():
    print("AeroAssist image split creation")
    print("-------------------------------")

    manifest = pd.read_csv(MANIFEST_PATH)

    if len(manifest) != 200:
        raise ValueError(
            f"Expected 200 manifest rows, found {len(manifest)}."
        )

    group_lookup = build_group_lookup()
    rng = random.Random(RANDOM_SEED)

    all_assignments = {}

    for class_name in sorted(manifest["class_label"].unique()):
        class_df = manifest[
            manifest["class_label"] == class_name
        ].copy()

        if len(class_df) != 20:
            raise ValueError(
                f"{class_name} has {len(class_df)} images instead of 20."
            )

        assignments = split_class(
            class_df,
            group_lookup,
            rng,
        )

        all_assignments.update(assignments)

    manifest["split"] = manifest["image_id"].map(all_assignments)

    if manifest["split"].isna().any():
        raise RuntimeError("Some images were not assigned a split.")

    manifest.to_csv(MANIFEST_PATH, index=False)

    print("\nOverall split")
    print("-------------")
    print(manifest["split"].value_counts().to_string())

    print("\nPer-class split")
    print("---------------")

    table = pd.crosstab(
        manifest["class_label"],
        manifest["split"],
    )

    table = table.reindex(
        columns=["development", "validation", "test"]
    )

    print(table.to_string())

    # Verify that every related group stayed together.
    print("\nRelated-group leakage check")
    print("---------------------------")

    leakage_found = False

    for group_name in sorted(set(group_lookup.values())):
        ids = [
            image_id
            for image_id, mapped_group in group_lookup.items()
            if mapped_group == group_name
        ]

        rows = manifest[manifest["image_id"].isin(ids)]

        if len(rows) == 0:
            continue

        splits = rows["split"].unique()

        if len(splits) != 1:
            leakage_found = True
            print(
                f"LEAKAGE: {group_name} -> "
                f"{rows[['image_id', 'split']].values.tolist()}"
            )

    if leakage_found:
        raise RuntimeError("Related-group leakage detected.")

    print("No related groups cross dataset splits.")

    print("\n-------------------------------")
    print("IMAGE SPLIT CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()