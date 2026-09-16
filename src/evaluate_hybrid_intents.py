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


def evaluate_hybrid():

    print("Loading passenger query dataset...")

    df = pd.read_csv(DATA_PATH)

    test_df = df[df["split"] == "test"].copy()

    print(f"Test samples: {len(test_df)}")

    print("Loading trained ML intent classifier...")

    bundle = joblib.load(MODEL_PATH)

    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    # -----------------------------------
    # Rule-based predictions
    # -----------------------------------

    test_df["rule_intent"] = test_df["text"].apply(
        detect_intent
    )

    # -----------------------------------
    # ML predictions
    # -----------------------------------

    X_test = vectorizer.transform(
        test_df["text"]
    )

    test_df["ml_intent"] = classifier.predict(
        X_test
    )

    # -----------------------------------
    # Hybrid predictions
    # -----------------------------------

    test_df["hybrid_intent"] = test_df.apply(
        lambda row: hybrid_intent_decision(
            row["text"],
            row["rule_intent"],
            row["ml_intent"]
        ),
        axis=1
    )

    # -----------------------------------
    # Accuracy comparison
    # -----------------------------------

    rule_accuracy = accuracy_score(
        test_df["intent"],
        test_df["rule_intent"]
    )

    ml_accuracy = accuracy_score(
        test_df["intent"],
        test_df["ml_intent"]
    )

    hybrid_accuracy = accuracy_score(
        test_df["intent"],
        test_df["hybrid_intent"]
    )

    print("\n===================================")
    print("AEROASSIST INTENT METHOD COMPARISON")
    print("===================================")

    print(f"\nRule-based accuracy: {rule_accuracy:.3f}")
    print(f"ML accuracy:         {ml_accuracy:.3f}")
    print(f"Hybrid accuracy:     {hybrid_accuracy:.3f}")

    print("\n===================================")
    print("HYBRID CLASSIFICATION REPORT")
    print("===================================\n")

    print(
        classification_report(
            test_df["intent"],
            test_df["hybrid_intent"],
            zero_division=0
        )
    )

    # -----------------------------------
    # Cases corrected by hybrid system
    # -----------------------------------

    corrected = test_df[
        (test_df["ml_intent"] != test_df["intent"])
        &
        (test_df["hybrid_intent"] == test_df["intent"])
    ]

    print("\n===================================")
    print("ML ERRORS CORRECTED BY HYBRID")
    print("===================================")

    if corrected.empty:
        print("No ML errors were corrected.")
    else:
        for _, row in corrected.iterrows():

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

    # -----------------------------------
    # Hybrid errors
    # -----------------------------------

    errors = test_df[
        test_df["hybrid_intent"] != test_df["intent"]
    ]

    print("\n===================================")
    print("REMAINING HYBRID ERRORS")
    print("===================================")

    if errors.empty:
        print("No hybrid errors.")
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
    evaluate_hybrid()