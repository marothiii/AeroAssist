from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoProcessor, CLIPModel


# ==================================================
# AeroAssist - FINAL FROZEN VISION PIPELINE
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "data" / "image_manifest.csv"
IMAGE_DIR = PROJECT_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ==================================================
# FROZEN PROMPT ENSEMBLE
#
# Selected using development data.
# Do NOT modify using test results.
# ==================================================

CLASS_PROMPTS = {
    "gate_sign": [
        "an airport boarding gate with a gate number sign",
        "an airport departure gate sign showing a gate number",
        "a numbered boarding gate inside an airport terminal",
        "airport signage directing passengers to boarding gates",
    ],

    "baggage_claim": [
        "an airport baggage claim carousel with luggage",
        "an airport baggage reclaim conveyor belt",
        "passengers collecting luggage at an airport baggage carousel",
        "an airport arrivals baggage claim area",
    ],

    "security": [
        "an airport passenger security screening checkpoint",
        "airport security screening with metal detectors and x-ray machines",
        "passengers going through an airport security checkpoint",
        "an airport security inspection and screening area",
    ],

    "check_in": [
        "an airport airline check-in counter",
        "airport check-in desks with airline counters",
        "passengers checking in for a flight at an airport counter",
        "an airport departure check-in hall with check-in desks",
    ],

    "lounge": [
        "an airport passenger lounge with seating",
        "an airline airport lounge waiting area",
        "an airport VIP lounge with chairs and tables",
        "an airport lounge interior for waiting passengers",
    ],

    "restaurant": [
        "an airport restaurant or cafe",
        "an airport food court with restaurants",
        "a cafe or dining area inside an airport terminal",
        "airport food and beverage shops with tables and seating",
    ],

    "information": [
        "an airport information desk with an information sign",
        "an airport passenger information counter",
        "an airport help desk for passenger information",
        "an airport information service counter inside a terminal",
    ],

    "transport": [
        "an airport train station or railway platform",
        "an airport taxi rank or taxi pickup area",
        "an airport bus stop or shuttle transport area",
        "airport ground transportation including trains buses and taxis",
    ],

    "accessibility": [
        "an airport accessibility facility for passengers with reduced mobility",
        "wheelchair assistance and accessible facilities at an airport",
        "an airport accessibility sign or accessible passenger service",
        "special assistance equipment for disabled airport passengers",
    ],

    "family_facility": [
        "an airport children's play area",
        "an airport baby care or nursing room",
        "an airport family facility for parents and children",
        "a baby changing room or children's facility inside an airport",
    ],
}


# ==================================================
# FROZEN CONFIDENCE POLICY
#
# Confident threshold selected on validation data.
# ==================================================

VISION_CONFIDENT_SCORE = 0.70
VISION_CONFIDENT_MARGIN = 0.05

VISION_CAUTION_SCORE = 0.50
VISION_CAUTION_MARGIN = 0.10


def vision_confidence_label(score, margin):
    if (
        score >= VISION_CONFIDENT_SCORE
        and margin >= VISION_CONFIDENT_MARGIN
    ):
        return "confident"

    if (
        score >= VISION_CAUTION_SCORE
        and margin >= VISION_CAUTION_MARGIN
    ):
        return "caution"

    return "uncertain"


# ==================================================
# Dataset
# ==================================================

def load_manifest():
    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "image_id",
        "filename",
        "class_label",
        "source_type",
        "source_reference",
        "split",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    return manifest


# ==================================================
# CLIP
# ==================================================

def load_clip():
    print(f"Loading CLIP: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    return processor, model


def extract_feature_tensor(output):
    """
    Compatibility helper for the installed
    Transformers version.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        "Unexpected CLIP output type: "
        f"{type(output)}"
    )


# ==================================================
# Frozen prompt embeddings
# ==================================================

def build_class_embeddings(processor, model):
    class_names = list(CLASS_PROMPTS.keys())
    class_embeddings = []

    print("\nBuilding frozen prompt-ensemble embeddings...")

    with torch.no_grad():

        for class_name in class_names:

            prompts = CLASS_PROMPTS[class_name]

            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )

            text_inputs = {
                key: value.to(DEVICE)
                for key, value in text_inputs.items()
            }

            output = model.get_text_features(
                **text_inputs
            )

            features = extract_feature_tensor(
                output
            )

            features = F.normalize(
                features,
                dim=-1,
            )

            class_feature = features.mean(
                dim=0,
                keepdim=True,
            )

            class_feature = F.normalize(
                class_feature,
                dim=-1,
            )

            class_embeddings.append(
                class_feature
            )

    class_embeddings = torch.cat(
        class_embeddings,
        dim=0,
    )

    return class_names, class_embeddings


# ==================================================
# Image classification
# ==================================================

def classify_image(
    image_path,
    processor,
    model,
    class_names,
    class_embeddings,
):
    with Image.open(image_path) as raw_image:

        image = ImageOps.exif_transpose(
            raw_image
        ).convert("RGB")

    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(DEVICE)
        for key, value in image_inputs.items()
    }

    with torch.no_grad():

        output = model.get_image_features(
            **image_inputs
        )

        image_features = extract_feature_tensor(
            output
        )

        image_features = F.normalize(
            image_features,
            dim=-1,
        )

        similarities = (
            image_features
            @ class_embeddings.T
        )[0]

        probabilities = torch.softmax(
            similarities * 100.0,
            dim=0,
        )

    ranked_indices = torch.argsort(
        probabilities,
        descending=True,
    )

    results = []

    for index in ranked_indices:

        index = index.item()

        results.append(
            {
                "class": class_names[index],
                "score": probabilities[index].item(),
                "similarity": similarities[index].item(),
            }
        )

    return results


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

def evaluate_test(
    manifest,
    processor,
    model,
    class_names,
    class_embeddings,
):
    test = manifest[
        manifest["split"] == "test"
    ].copy()

    if len(test) != 30:
        raise ValueError(
            f"Expected exactly 30 test images, "
            f"found {len(test)}."
        )

    rows = []

    print("\nEvaluating FINAL held-out test set...")
    print("------------------------------------")

    for number, (_, row) in enumerate(
        test.iterrows(),
        start=1,
    ):

        image_path = (
            IMAGE_DIR
            / row["class_label"]
            / row["filename"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        results = classify_image(
            image_path,
            processor,
            model,
            class_names,
            class_embeddings,
        )

        top1 = results[0]
        second = results[1]
        third = results[2]

        true_class = row["class_label"]

        top3_classes = [
            result["class"]
            for result in results[:3]
        ]

        confidence_margin = (
            top1["score"] - second["score"]
        )

        similarity_margin = (
            top1["similarity"]
            - second["similarity"]
        )

        confidence_label = vision_confidence_label(
            top1["score"],
            confidence_margin,
        )

        correct = (
            top1["class"] == true_class
        )

        rows.append(
            {
                "image_id": row["image_id"],
                "filename": row["filename"],
                "true_class": true_class,

                "predicted_class": top1["class"],

                "top1_score": top1["score"],
                "top1_similarity": top1["similarity"],

                "second_class": second["class"],
                "second_score": second["score"],
                "second_similarity": second["similarity"],

                "third_class": third["class"],
                "third_score": third["score"],
                "third_similarity": third["similarity"],

                "confidence_margin": confidence_margin,
                "similarity_margin": similarity_margin,

                "confidence_label": confidence_label,

                "top1_correct": correct,

                "top3_correct": (
                    true_class in top3_classes
                ),
            }
        )

        status = (
            "OK"
            if correct
            else "WRONG"
        )

        print(
            f"[{number:02d}/30] "
            f"{row['image_id']} | "
            f"{status:5s} | "
            f"true={true_class} | "
            f"pred={top1['class']} | "
            f"score={top1['score']:.4f} | "
            f"margin={confidence_margin:.4f} | "
            f"{confidence_label}"
        )

    return pd.DataFrame(rows)


# ==================================================
# Final metrics
# ==================================================

def report_test_metrics(predictions):

    total = len(predictions)

    top1_accuracy = (
        predictions["top1_correct"].mean()
    )

    top3_accuracy = (
        predictions["top3_correct"].mean()
    )

    print("\nFINAL HELD-OUT VISION TEST")
    print("==========================")

    print(f"Images evaluated: {total}")
    print(
        f"Top-1 accuracy: {top1_accuracy:.3f}"
    )
    print(
        f"Top-3 accuracy: {top3_accuracy:.3f}"
    )

    # ----------------------------------------------
    # Per-class accuracy
    # ----------------------------------------------

    print("\nPer-class Top-1 accuracy")
    print("------------------------")

    per_class = (
        predictions
        .groupby("true_class")["top1_correct"]
        .agg(["sum", "count", "mean"])
        .sort_index()
    )

    for class_name, row in per_class.iterrows():

        correct = int(row["sum"])
        count = int(row["count"])

        print(
            f"{class_name:16s} "
            f"{correct}/{count} "
            f"({row['mean']:.3f})"
        )

    # ----------------------------------------------
    # Confidence distribution
    # ----------------------------------------------

    print("\nConfidence distribution")
    print("-----------------------")

    confidence_counts = (
        predictions["confidence_label"]
        .value_counts()
    )

    for label in [
        "confident",
        "caution",
        "uncertain",
    ]:

        count = int(
            confidence_counts.get(
                label,
                0,
            )
        )

        print(
            f"{label:10s}: "
            f"{count}/{total} "
            f"({count / total:.3f})"
        )

    # ----------------------------------------------
    # Selective accuracy
    # ----------------------------------------------

    confident = predictions[
        predictions["confidence_label"]
        == "confident"
    ]

    print("\nConfidence-sensitive evaluation")
    print("-------------------------------")

    coverage = (
        len(confident) / total
    )

    print(
        f"Confident coverage: "
        f"{len(confident)}/{total} "
        f"({coverage:.3f})"
    )

    if len(confident) > 0:

        selective_accuracy = (
            confident["top1_correct"].mean()
        )

        confident_errors = int(
            (~confident["top1_correct"]).sum()
        )

        print(
            f"Selective accuracy: "
            f"{selective_accuracy:.3f}"
        )

        print(
            f"Confident errors: "
            f"{confident_errors}"
        )

    else:

        print(
            "Selective accuracy: N/A"
        )

        print(
            "Confident errors: 0"
        )

    # ----------------------------------------------
    # Accuracy by confidence label
    # ----------------------------------------------

    print("\nAccuracy by confidence label")
    print("----------------------------")

    for label in [
        "confident",
        "caution",
        "uncertain",
    ]:

        subset = predictions[
            predictions["confidence_label"]
            == label
        ]

        if len(subset) == 0:

            print(
                f"{label:10s}: "
                "no predictions"
            )

        else:

            accuracy = (
                subset["top1_correct"].mean()
            )

            print(
                f"{label:10s}: "
                f"{accuracy:.3f} "
                f"({int(subset['top1_correct'].sum())}"
                f"/{len(subset)})"
            )

    # ----------------------------------------------
    # Errors
    # ----------------------------------------------

    errors = predictions[
        ~predictions["top1_correct"]
    ]

    print("\nTest errors")
    print("-----------")

    print(
        f"Incorrect Top-1 predictions: "
        f"{len(errors)}/{total}"
    )

    for _, row in errors.iterrows():

        print(
            f"{row['image_id']} | "
            f"{row['true_class']} -> "
            f"{row['predicted_class']} | "
            f"score={row['top1_score']:.4f} | "
            f"margin={row['confidence_margin']:.4f} | "
            f"{row['confidence_label']}"
        )


# ==================================================
# Main
# ==================================================

def main():

    print(
        "AeroAssist FINAL frozen vision evaluation"
    )

    print(
        "========================================="
    )

    manifest = load_manifest()

    print(
        f"Manifest images: {len(manifest)}"
    )

    print("\nSplit distribution:")

    print(
        manifest["split"]
        .value_counts()
        .to_string()
    )

    processor, model = load_clip()

    print("\nCLIP loaded successfully.")

    print(
        "Frozen confident threshold: "
        f"score >= {VISION_CONFIDENT_SCORE:.2f}, "
        f"margin >= {VISION_CONFIDENT_MARGIN:.2f}"
    )

    print(
        "Frozen caution threshold: "
        f"score >= {VISION_CAUTION_SCORE:.2f}, "
        f"margin >= {VISION_CAUTION_MARGIN:.2f}"
    )

    class_names, class_embeddings = (
        build_class_embeddings(
            processor,
            model,
        )
    )

    print(
        f"Class embedding shape: "
        f"{tuple(class_embeddings.shape)}"
    )

    predictions = evaluate_test(
        manifest,
        processor,
        model,
        class_names,
        class_embeddings,
    )

    report_test_metrics(
        predictions
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "vision_final_test_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print("\nFinal predictions saved to:")
    print(output_path)

    print("\n=================================")
    print("FINAL VISION TEST COMPLETE")
    print("DO NOT TUNE ON THESE TEST RESULTS")


if __name__ == "__main__":
    main()