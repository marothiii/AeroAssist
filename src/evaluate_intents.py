import os

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from text_pipeline import detect_intent


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "passenger_queries.csv"
)


def evaluate_intents():

    print("Loading passenger query dataset...")

    df = pd.read_csv(DATA_PATH)

    # Evaluate only on the held-out test set
    test_df = df[df["split"] == "test"].copy()

    print(f"Test samples: {len(test_df)}")

    # Run AeroAssist intent detector
    test_df["predicted_intent"] = test_df["text"].apply(
        detect_intent
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        test_df["intent"],
        test_df["predicted_intent"]
    )

    print("\n==============================")
    print("AEROASSIST INTENT EVALUATION")
    print("==============================")

    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_df["intent"],
            test_df["predicted_intent"],
            zero_division=0
        )
    )

    # Show incorrect predictions
    errors = test_df[
        test_df["intent"] != test_df["predicted_intent"]
    ]

    print("\n==============================")
    print("INCORRECT PREDICTIONS")
    print("==============================")

    if len(errors) == 0:
        print("No incorrect predictions.")
    else:
        for _, row in errors.iterrows():
            print("\nPassenger query:")
            print(row["text"])

            print("Expected:")
            print(row["intent"])

            print("Predicted:")
            print(row["predicted_intent"])

    # Confusion matrix
    labels = sorted(test_df["intent"].unique())

    matrix = confusion_matrix(
        test_df["intent"],
        test_df["predicted_intent"],
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(confusion_df)


if __name__ == "__main__":
    evaluate_intents()