from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "aeroassist_class_distribution.png"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DISPLAY_NAMES = {
    "accessibility": "Accessibility",
    "baggage_claim": "Baggage Claim",
    "check_in": "Check-in",
    "family_facility": "Family Facility",
    "gate_sign": "Gate Sign",
    "information": "Information",
    "lounge": "Lounge",
    "restaurant": "Restaurant",
    "security": "Security",
    "transport": "Transport",
}

classes = []
counts = []

for folder in sorted(IMAGE_DIR.iterdir()):
    if not folder.is_dir():
        continue

    count = sum(
        1
        for file in folder.iterdir()
        if file.suffix.lower() in VALID_EXTENSIONS
    )

    classes.append(
        DISPLAY_NAMES.get(
            folder.name,
            folder.name.replace("_", " ").title(),
        )
    )
    counts.append(count)


total_images = sum(counts)

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(classes, counts)

ax.set_title(
    "AeroAssist Visual Dataset Class Distribution",
    fontsize=16,
    fontweight="bold",
    pad=16,
)

ax.set_ylabel("Number of Images")
ax.set_xlabel("Airport Scene Category")

ax.set_ylim(0, max(counts) + 5)

ax.tick_params(
    axis="x",
    rotation=35,
)

# Show the exact number above each category.
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax.text(
    0.99,
    0.96,
    f"Total images: {total_images}\nClasses: {len(classes)}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Classes:", len(classes))
print("Total images:", total_images)

for name, count in zip(classes, counts):
    print(f"{name}: {count}")

print("\nFigure saved to:")
print(OUTPUT_PATH)
