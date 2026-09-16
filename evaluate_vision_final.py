from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from src.vision_inference import (
    AeroAssistVisionInference,
    DEVICE,
    vision_confidence_label,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_DIR / "images"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# LOAD EXACT AEROASSIST VISION MODEL
# ============================================================

print("\n==========================================")
print("AEROASSIST FINAL VISION EVALUATION")
print("==========================================\n")

vision = AeroAssistVisionInference()

class_names = vision.class_names

print("\nClasses:")
for i, class_name in enumerate(
    class_names,
    start=1,
):
    print(
        f"{i:02d}. {class_name}"
    )


# ============================================================
# FULL CLASS PREDICTION
# ============================================================

def predict_full(image_path):
    """
    Uses the exact same frozen CLIP processor,
    model, prompt ensemble and class embeddings
    as AeroAssistVisionInference.

    Unlike predict(), this retains all class
    probabilities so Top-3 can be evaluated.
    """

    image_path = Path(image_path)

    with Image.open(
        image_path
    ) as raw_image:

        image = ImageOps.exif_transpose(
            raw_image
        ).convert("RGB")

    image_inputs = vision.processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(DEVICE)
        for key, value
        in image_inputs.items()
    }

    with torch.no_grad():

        output = (
            vision.model
            .get_image_features(
                **image_inputs
            )
        )

        if isinstance(
            output,
            torch.Tensor,
        ):
            image_features = output

        elif hasattr(
            output,
            "pooler_output",
        ):
            image_features = (
                output.pooler_output
            )

        else:
            raise TypeError(
                "Unexpected CLIP "
                f"output type: {type(output)}"
            )

        image_features = F.normalize(
            image_features,
            dim=-1,
        )

        similarities = (
            image_features
            @ vision.class_embeddings.T
        )[0]

        probabilities = torch.softmax(
            similarities * 100.0,
            dim=0,
        )

    ranked_indices = torch.argsort(
        probabilities,
        descending=True,
    )

    ranked_indices = (
        ranked_indices
        .detach()
        .cpu()
        .tolist()
    )

    probability_values = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    similarity_values = (
        similarities
        .detach()
        .cpu()
        .numpy()
    )

    ranking = []

    for index in ranked_indices:

        ranking.append(
            {
                "class": class_names[index],
                "score": float(
                    probability_values[index]
                ),
                "similarity": float(
                    similarity_values[index]
                ),
            }
        )

    top1 = ranking[0]
    top2 = ranking[1]

    probability_margin = (
        top1["score"]
        - top2["score"]
    )

    similarity_margin = (
        top1["similarity"]
        - top2["similarity"]
    )

    confidence = (
        vision_confidence_label(
            top1["score"],
            probability_margin,
        )
    )

    return {
        "ranking": ranking,
        "predicted_class": (
            top1["class"]
        ),
        "score": top1["score"],
        "similarity": (
            top1["similarity"]
        ),
        "second_class": (
            top2["class"]
        ),
        "second_score": (
            top2["score"]
        ),
        "margin": (
            probability_margin
        ),
        "similarity_margin": (
            similarity_margin
        ),
        "confidence": confidence,
    }


# ============================================================
# DISCOVER DATASET
# ============================================================

dataset = []

for class_name in class_names:

    class_dir = (
        IMAGE_DIR
        / class_name
    )

    if not class_dir.exists():

        print(
            f"WARNING: missing folder "
            f"{class_dir}"
        )

        continue

    image_paths = sorted(
        [
            path
            for path
            in class_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in VALID_EXTENSIONS
            )
        ]
    )

    print(
        f"{class_name:20s}: "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        dataset.append(
            (
                class_name,
                image_path,
            )
        )


if not dataset:

    raise RuntimeError(
        "No evaluation images found."
    )


print(
    "\nTotal images:",
    len(dataset),
)


# ============================================================
# RUN EVALUATION
# ============================================================

rows = []

true_labels = []
predicted_labels = []

print("\nRunning frozen CLIP evaluation...\n")


for index, (
    true_class,
    image_path,
) in enumerate(
    dataset,
    start=1,
):

    result = predict_full(
        image_path
    )

    ranking = result[
        "ranking"
    ]

    top3 = [
        item["class"]
        for item
        in ranking[:3]
    ]

    top1_correct = (
        result["predicted_class"]
        == true_class
    )

    top3_correct = (
        true_class in top3
    )

    # Vision-level operational policy:
    # confident/caution = usable evidence
    # uncertain = abstain
    accepted = (
        result["confidence"]
        != "uncertain"
    )

    unsafe_accepted_error = (
        accepted
        and not top1_correct
    )

    abstained = (
        not accepted
    )

    abstained_error = (
        abstained
        and not top1_correct
    )

    abstained_correct = (
        abstained
        and top1_correct
    )

    row = {
        "image": str(
            image_path.relative_to(
                PROJECT_DIR
            )
        ),
        "true_class": (
            true_class
        ),
        "predicted_class": (
            result[
                "predicted_class"
            ]
        ),
        "top1_correct": (
            top1_correct
        ),
        "top3_correct": (
            top3_correct
        ),
        "top1_score": (
            result["score"]
        ),
        "top1_similarity": (
            result[
                "similarity"
            ]
        ),
        "second_class": (
            result[
                "second_class"
            ]
        ),
        "second_score": (
            result[
                "second_score"
            ]
        ),
        "probability_margin": (
            result["margin"]
        ),
        "similarity_margin": (
            result[
                "similarity_margin"
            ]
        ),
        "confidence": (
            result[
                "confidence"
            ]
        ),
        "accepted": accepted,
        "abstained": abstained,
        "unsafe_accepted_error": (
            unsafe_accepted_error
        ),
        "abstained_error": (
            abstained_error
        ),
        "abstained_correct": (
            abstained_correct
        ),
        "top2_class": (
            ranking[1]["class"]
        ),
        "top3_class": (
            ranking[2]["class"]
        ),
    }

    rows.append(
        row
    )

    true_labels.append(
        true_class
    )

    predicted_labels.append(
        result[
            "predicted_class"
        ]
    )

    print(
        f"[{index:03d}/{len(dataset):03d}] "
        f"{image_path.name:35s} "
        f"TRUE={true_class:18s} "
        f"PRED={result['predicted_class']:18s} "
        f"SCORE={result['score']:.3f} "
        f"{result['confidence'].upper()}"
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


predictions_path = (
    RESULTS_DIR
    / "vision_predictions.csv"
)

df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# OVERALL METRICS
# ============================================================

n = len(df)

top1_accuracy = (
    df["top1_correct"]
    .mean()
)

top3_accuracy = (
    df["top3_correct"]
    .mean()
)

mean_score = (
    df["top1_score"]
    .mean()
)

mean_similarity = (
    df["top1_similarity"]
    .mean()
)

mean_margin = (
    df["probability_margin"]
    .mean()
)

mean_similarity_margin = (
    df["similarity_margin"]
    .mean()
)

acceptance_rate = (
    df["accepted"]
    .mean()
)

abstention_rate = (
    df["abstained"]
    .mean()
)


accepted_df = df[
    df["accepted"]
]

if len(accepted_df) > 0:

    accepted_accuracy = (
        accepted_df[
            "top1_correct"
        ].mean()
    )

else:

    accepted_accuracy = np.nan


unsafe_errors = int(
    df[
        "unsafe_accepted_error"
    ].sum()
)

abstained_errors = int(
    df[
        "abstained_error"
    ].sum()
)

abstained_correct = int(
    df[
        "abstained_correct"
    ].sum()
)


# ============================================================
# CONFIDENCE COUNTS
# ============================================================

confident_count = int(
    (
        df["confidence"]
        == "confident"
    ).sum()
)

caution_count = int(
    (
        df["confidence"]
        == "caution"
    ).sum()
)

uncertain_count = int(
    (
        df["confidence"]
        == "uncertain"
    ).sum()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n\n"
    "=========================================="
)

print(
    "FINAL VISION RESULTS"
)

print(
    "=========================================="
)

print(
    f"Images evaluated:              {n}"
)

print(
    f"Top-1 accuracy:               "
    f"{top1_accuracy:.3%}"
)

print(
    f"Top-3 accuracy:               "
    f"{top3_accuracy:.3%}"
)

print(
    f"Mean Top-1 probability:       "
    f"{mean_score:.4f}"
)

print(
    f"Mean cosine similarity:       "
    f"{mean_similarity:.4f}"
)

print(
    f"Mean probability margin:      "
    f"{mean_margin:.4f}"
)

print(
    f"Mean similarity margin:       "
    f"{mean_similarity_margin:.4f}"
)

print(
    "\n--- Confidence policy ---"
)

print(
    f"Confident predictions:        "
    f"{confident_count}"
)

print(
    f"Caution predictions:          "
    f"{caution_count}"
)

print(
    f"Uncertain predictions:        "
    f"{uncertain_count}"
)

print(
    f"Operational acceptance rate: "
    f"{acceptance_rate:.3%}"
)

print(
    f"Abstention rate:              "
    f"{abstention_rate:.3%}"
)

print(
    f"Accuracy when accepted:       "
    f"{accepted_accuracy:.3%}"
)

print(
    f"Incorrect accepted cases:     "
    f"{unsafe_errors}"
)

print(
    f"Incorrect cases abstained:    "
    f"{abstained_errors}"
)

print(
    f"Correct cases abstained:      "
    f"{abstained_correct}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

per_class_rows = []

for class_name in class_names:

    class_df = df[
        df["true_class"]
        == class_name
    ]

    if class_df.empty:
        continue

    class_accepted = class_df[
        class_df["accepted"]
    ]

    if len(
        class_accepted
    ) > 0:

        selective_accuracy = (
            class_accepted[
                "top1_correct"
            ].mean()
        )

    else:

        selective_accuracy = (
            np.nan
        )

    per_class_rows.append(
        {
            "class": class_name,
            "images": len(
                class_df
            ),
            "top1_accuracy": (
                class_df[
                    "top1_correct"
                ].mean()
            ),
            "top3_accuracy": (
                class_df[
                    "top3_correct"
                ].mean()
            ),
            "mean_score": (
                class_df[
                    "top1_score"
                ].mean()
            ),
            "mean_margin": (
                class_df[
                    "probability_margin"
                ].mean()
            ),
            "acceptance_rate": (
                class_df[
                    "accepted"
                ].mean()
            ),
            "accepted_accuracy": (
                selective_accuracy
            ),
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)


per_class_path = (
    RESULTS_DIR
    / "vision_per_class.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False,
)


print(
    "\n--- Per-class performance ---\n"
)

print(
    per_class_df.to_string(
        index=False,
        formatters={
            "top1_accuracy":
                lambda x: f"{x:.1%}",
            "top3_accuracy":
                lambda x: f"{x:.1%}",
            "mean_score":
                lambda x: f"{x:.3f}",
            "mean_margin":
                lambda x: f"{x:.3f}",
            "acceptance_rate":
                lambda x: f"{x:.1%}",
            "accepted_accuracy":
                lambda x: (
                    "—"
                    if pd.isna(x)
                    else f"{x:.1%}"
                ),
        },
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    labels=class_names,
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)


report_df = (
    pd.DataFrame(
        report
    )
    .transpose()
)


report_path = (
    RESULTS_DIR
    / "vision_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=class_names,
)


fig, ax = plt.subplots(
    figsize=(11, 9)
)

image = ax.imshow(
    cm,
    interpolation="nearest",
)

ax.set_title(
    "AeroAssist CLIP Vision Confusion Matrix"
)

ax.set_xlabel(
    "Predicted class"
)

ax.set_ylabel(
    "True class"
)

ax.set_xticks(
    np.arange(
        len(class_names)
    )
)

ax.set_yticks(
    np.arange(
        len(class_names)
    )
)

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(
    class_names
)


threshold = (
    cm.max() / 2
    if cm.size
    else 0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
        )


fig.tight_layout()


confusion_path = (
    RESULTS_DIR
    / "vision_confusion_matrix.png"
)

fig.savefig(
    confusion_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    df["top1_score"],
    bins=15,
)

ax.set_title(
    "Distribution of CLIP Top-1 Prediction Scores"
)

ax.set_xlabel(
    "Top-1 probability"
)

ax.set_ylabel(
    "Number of images"
)

fig.tight_layout()


confidence_plot_path = (
    RESULTS_DIR
    / "vision_confidence_distribution.png"
)

fig.savefig(
    confidence_plot_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# FAILURES
# ============================================================

failure_df = df[
    ~df["top1_correct"]
].copy()


failure_df = failure_df.sort_values(
    by="top1_score",
    ascending=False,
)


failure_path = (
    RESULTS_DIR
    / "vision_failure_cases.csv"
)

failure_df.to_csv(
    failure_path,
    index=False,
)


# ============================================================
# LOW-CONFIDENCE CASES
# ============================================================

low_confidence_df = df[
    df["confidence"]
    == "uncertain"
].copy()


low_confidence_df = (
    low_confidence_df
    .sort_values(
        by="top1_score",
        ascending=False,
    )
)


low_confidence_path = (
    RESULTS_DIR
    / "vision_abstention_cases.csv"
)

low_confidence_df.to_csv(
    low_confidence_path,
    index=False,
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = [
    {
        "metric": "images_evaluated",
        "value": n,
    },
    {
        "metric": "top1_accuracy",
        "value": top1_accuracy,
    },
    {
        "metric": "top3_accuracy",
        "value": top3_accuracy,
    },
    {
        "metric": "mean_top1_probability",
        "value": mean_score,
    },
    {
        "metric": "mean_cosine_similarity",
        "value": mean_similarity,
    },
    {
        "metric": "mean_probability_margin",
        "value": mean_margin,
    },
    {
        "metric": "mean_similarity_margin",
        "value": mean_similarity_margin,
    },
    {
        "metric": "acceptance_rate",
        "value": acceptance_rate,
    },
    {
        "metric": "abstention_rate",
        "value": abstention_rate,
    },
    {
        "metric": "accuracy_when_accepted",
        "value": accepted_accuracy,
    },
    {
        "metric": "unsafe_accepted_errors",
        "value": unsafe_errors,
    },
    {
        "metric": "abstained_incorrect_predictions",
        "value": abstained_errors,
    },
    {
        "metric": "abstained_correct_predictions",
        "value": abstained_correct,
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (
    RESULTS_DIR
    / "vision_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n=========================================="
)

print(
    "FILES SAVED"
)

print(
    "=========================================="
)

for path in [
    predictions_path,
    per_class_path,
    report_path,
    confusion_path,
    confidence_plot_path,
    failure_path,
    low_confidence_path,
    summary_path,
]:

    print(
        path.relative_to(
            PROJECT_DIR
        )
    )


print(
    "\nVision evaluation complete."
)