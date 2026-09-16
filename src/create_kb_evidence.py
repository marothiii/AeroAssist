from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "airport_knowledge_base.json"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "aeroassist_knowledge_base.png"

SELECTED_IDS = [
    "NIA006",
    "NIA017",
    "NIA019",
    "NIA024",
]

with open(KB_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

records = {
    record["id"]: record
    for record in data
}

selected = [
    records[record_id]
    for record_id in SELECTED_IDS
]


def wrap(value, width=43):
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
        )
    )


def yes_no(value):
    return "Yes" if value else "No"


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

axes = axes.flatten()

for ax, record in zip(axes, selected):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Card background
    card = FancyBboxPatch(
        (0.03, 0.04),
        0.94,
        0.90,
        boxstyle="round,pad=0.018",
        linewidth=1.2,
        facecolor="white",
        edgecolor="0.65",
    )

    ax.add_patch(card)

    ax.text(
        0.07,
        0.88,
        record["name"],
        fontsize=16,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.07,
        0.81,
        f'{record["id"]}  |  {record["category"].upper()}',
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    location = (
        f'{record["terminal"]}, '
        f'{record["zone"]}, '
        f'{record["floor"]}'
    )

    fields = [
        ("Location", location),
        ("Description", record["description"]),
        ("Directions", record["directions"]),
        (
            "Accessible route",
            record["accessible_route"],
        ),
        (
            "Walking time",
            f'{record["walking_minutes"]} min '
            f'(accessible: {record["accessible_minutes"]} min)',
        ),
        (
            "Criticality",
            record["criticality"],
        ),
        (
            "Live data required",
            yes_no(record["live_data_required"]),
        ),
    ]

    # The final field changes according to the record,
    # showing why different records contain different
    # passenger-safety information.
    if record["id"] == "NIA019":
        fields.append(
            (
                "Family facilities",
                record["family_facilities"],
            )
        )
    else:
        fields.append(
            (
                "Verification",
                record["verification_message"]
                or "No additional verification message",
            )
        )

    y = 0.74

    for label, value in fields:

        ax.text(
            0.07,
            y,
            f"{label}:",
            fontsize=9.5,
            fontweight="bold",
            va="top",
        )

        wrapped = wrap(value)

        ax.text(
            0.31,
            y,
            wrapped,
            fontsize=9.3,
            va="top",
        )

        line_count = wrapped.count("\n") + 1
        y -= 0.055 + (line_count - 1) * 0.038


fig.suptitle(
    "AeroAssist Structured Airport Knowledge Base",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

fig.text(
    0.5,
    0.025,
    (
        f"Selected evidence from {len(data)} JSON records: "
        "wayfinding, accessibility, family support and critical assistance"
    ),
    ha="center",
    fontsize=10.5,
)

plt.tight_layout(
    rect=[0.02, 0.05, 0.98, 0.94]
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Knowledge-base records:", len(data))
print("Schema fields:", len(data[0]))
print("Selected records:", ", ".join(SELECTED_IDS))

print("\nFigure saved to:")
print(OUTPUT_PATH)
