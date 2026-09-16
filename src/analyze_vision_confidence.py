from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

VALIDATION_PATH = (
    PROJECT_DIR
    / "results"
    / "vision_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "vision_confidence_threshold_analysis.csv"
)


# --------------------------------------------------
# Load validation predictions
# --------------------------------------------------

def load_predictions():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: "
            f"{VALIDATION_PATH}"
        )

    predictions = pd.read_csv(VALIDATION_PATH)

    required_columns = {
        "image_id",
        "true_class",
        "predicted_class",
        "top1_score",
        "confidence_margin",
        "top1_correct",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(predictions) != 30:
        raise ValueError(
            f"Expected 30 validation predictions, "
            f"found {len(predictions)}."
        )

    return predictions


# --------------------------------------------------
# Evaluate threshold combinations
# --------------------------------------------------

def evaluate_thresholds(predictions):
    """
    'Accepted' means the vision prediction is considered
    confident enough to use without requesting additional
    evidence.

    We evaluate score AND margin together.
    """

    score_thresholds = [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    margin_thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    rows = []

    total = len(predictions)

    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:

            accepted_mask = (
                (
                    predictions["top1_score"]
                    >= score_threshold
                )
                &
                (
                    predictions["confidence_margin"]
                    >= margin_threshold
                )
            )

            accepted = predictions[accepted_mask]
            rejected = predictions[~accepted_mask]

            accepted_count = len(accepted)
            rejected_count = len(rejected)

            coverage = accepted_count / total
            abstention_rate = rejected_count / total

            if accepted_count > 0:
                selective_accuracy = (
                    accepted["top1_correct"].mean()
                )

                accepted_errors = (
                    ~accepted["top1_correct"]
                ).sum()

            else:
                selective_accuracy = 0.0
                accepted_errors = 0

            rows.append(
                {
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                    "accepted": accepted_count,
                    "rejected": rejected_count,
                    "coverage": coverage,
                    "abstention_rate": abstention_rate,
                    "selective_accuracy": selective_accuracy,
                    "accepted_errors": int(
                        accepted_errors
                    ),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Display useful candidates
# --------------------------------------------------

def report_candidates(results):
    print("\nVision confidence threshold analysis")
    print("------------------------------------")
    print(
        "Thresholds are derived ONLY from "
        "the validation split."
    )

    # Useful policies need reasonable coverage.
    useful = results[
        results["coverage"] >= 0.30
    ].copy()

    useful = useful.sort_values(
        by=[
            "selective_accuracy",
            "coverage",
            "score_threshold",
            "margin_threshold",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    )

    print(
        "\nBest candidates with at least "
        "30% coverage:"
    )

    print(
        useful.head(15).to_string(
            index=False,
            formatters={
                "coverage": "{:.3f}".format,
                "abstention_rate": "{:.3f}".format,
                "selective_accuracy": "{:.3f}".format,
            },
        )
    )

    print("\nZero-error candidates")
    print("---------------------")

    zero_error = useful[
        useful["accepted_errors"] == 0
    ].copy()

    if zero_error.empty:
        print(
            "No zero-error threshold combination "
            "with at least 30% coverage."
        )
    else:
        zero_error = zero_error.sort_values(
            by="coverage",
            ascending=False,
        )

        print(
            zero_error.head(15).to_string(
                index=False,
                formatters={
                    "coverage": "{:.3f}".format,
                    "abstention_rate": "{:.3f}".format,
                    "selective_accuracy": "{:.3f}".format,
                },
            )
        )


# --------------------------------------------------
# Inspect individual predictions
# --------------------------------------------------

def report_ranked_predictions(predictions):
    ranked = predictions.sort_values(
        by=[
            "top1_score",
            "confidence_margin",
        ],
        ascending=False,
    )

    print("\nPredictions ranked by confidence")
    print("--------------------------------")

    for _, row in ranked.iterrows():

        status = (
            "CORRECT"
            if row["top1_correct"]
            else "WRONG"
        )

        print(
            f"{row['image_id']} | "
            f"{status:7s} | "
            f"{row['true_class']} -> "
            f"{row['predicted_class']} | "
            f"score={row['top1_score']:.4f} | "
            f"margin={row['confidence_margin']:.4f}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("AeroAssist vision confidence analysis")
    print("-------------------------------------")

    predictions = load_predictions()

    print(
        f"Validation predictions loaded: "
        f"{len(predictions)}"
    )

    print(
        f"Overall Top-1 accuracy: "
        f"{predictions['top1_correct'].mean():.3f}"
    )

    results = evaluate_thresholds(
        predictions
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    report_candidates(
        results
    )

    report_ranked_predictions(
        predictions
    )

    print("\nFull threshold analysis saved to:")
    print(OUTPUT_PATH)

    print("\n-----------------------------------")
    print("CONFIDENCE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()