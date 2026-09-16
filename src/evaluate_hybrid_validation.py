import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from text_pipeline import (
    detect_intent,
    hybrid_intent_decision
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "passenger_queries.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "intent_classifier.joblib"
)


def evaluate_validation():

    df = pd.read_csv(DATA_PATH)

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(f"Validation samples: {len(validation_df)}")

    bundle = joblib.load(MODEL_PATH)

    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    # Rule-based predictions
    validation_df["rule_intent"] = (
        validation_df["text"].apply(detect_intent)
    )

    # ML predictions
    X_validation = vectorizer.transform(
        validation_df["text"]
    )

    validation_df["ml_intent"] = (
        classifier.predict(X_validation)
    )

    # Hybrid V2 predictions
    validation_df["hybrid_intent"] = (
        validation_df.apply(
            lambda row: hybrid_intent_decision(
                row["text"],
                row["rule_intent"],
                row["ml_intent"]
            ),
            axis=1
        )
    )

    rule_accuracy = accuracy_score(
        validation_df["intent"],
        validation_df["rule_intent"]
    )

    ml_accuracy = accuracy_score(
        validation_df["intent"],
        validation_df["ml_intent"]
    )

    hybrid_accuracy = accuracy_score(
        validation_df["intent"],
        validation_df["hybrid_intent"]
    )

    print("\n================================")
    print("VALIDATION METHOD COMPARISON")
    print("================================")

    print(f"\nRule accuracy:   {rule_accuracy:.3f}")
    print(f"ML accuracy:     {ml_accuracy:.3f}")
    print(f"Hybrid accuracy: {hybrid_accuracy:.3f}")

    print("\n================================")
    print("HYBRID VALIDATION REPORT")
    print("================================\n")

    print(
        classification_report(
            validation_df["intent"],
            validation_df["hybrid_intent"],
            zero_division=0
        )
    )

    errors = validation_df[
        validation_df["intent"]
        != validation_df["hybrid_intent"]
    ]

    print("\n================================")
    print("HYBRID VALIDATION ERRORS")
    print("================================")

    if errors.empty:
        print("No validation errors.")
    else:
        for _, row in errors.iterrows():

            print("\nPassenger query:")
            print(row["text"])

            print("Expected:")
            print(row["intent"])

            print("Rule:")
            print(row["rule_intent"])

            print("ML:")
            print(row["ml_intent"])

            print("Hybrid:")
            print(row["hybrid_intent"])


if __name__ == "__main__":
    evaluate_validation()