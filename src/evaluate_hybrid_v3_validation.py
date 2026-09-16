import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from text_pipeline import AeroAssistTextPipeline


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "passenger_queries.csv"
)


def main():
    print("Loading validation dataset...")

    df = pd.read_csv(DATA_PATH)

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(
        f"Validation samples: {len(validation_df)}"
    )

    print("\nLoading AeroAssist Hybrid V3...")
    pipeline = AeroAssistTextPipeline()

    expected = []

    rule_predictions = []
    tfidf_predictions = []
    semantic_predictions = []
    hybrid_predictions = []

    errors = []

    for _, row in validation_df.iterrows():

        query = row["text"]
        true_intent = row["intent"]

        result = pipeline.retrieve(query)

        rule_pred = result["rule_intent"]
        tfidf_pred = result["tfidf_intent"]
        semantic_pred = result["semantic_intent"]
        hybrid_pred = result["intent"]

        expected.append(true_intent)

        rule_predictions.append(rule_pred)
        tfidf_predictions.append(tfidf_pred)
        semantic_predictions.append(semantic_pred)
        hybrid_predictions.append(hybrid_pred)

        if hybrid_pred != true_intent:
            errors.append(
                {
                    "query": query,
                    "expected": true_intent,
                    "rule": rule_pred,
                    "tfidf": tfidf_pred,
                    "semantic": semantic_pred,
                    "hybrid_v3": hybrid_pred
                }
            )

    print("\n==============================")
    print("VALIDATION ACCURACY")
    print("==============================")

    print(
        "Rule:",
        round(
            accuracy_score(
                expected,
                rule_predictions
            ),
            3
        )
    )

    print(
        "TF-IDF:",
        round(
            accuracy_score(
                expected,
                tfidf_predictions
            ),
            3
        )
    )

    print(
        "Semantic:",
        round(
            accuracy_score(
                expected,
                semantic_predictions
            ),
            3
        )
    )

    print(
        "Hybrid V3:",
        round(
            accuracy_score(
                expected,
                hybrid_predictions
            ),
            3
        )
    )

    print("\n==============================")
    print("HYBRID V3 CLASSIFICATION REPORT")
    print("==============================\n")

    print(
        classification_report(
            expected,
            hybrid_predictions,
            zero_division=0
        )
    )

    print("\n==============================")
    print("HYBRID V3 ERRORS")
    print("==============================")

    if not errors:
        print("No validation errors.")
    else:
        for error in errors:
            print("\nQuery:", error["query"])
            print("Expected:", error["expected"])
            print("Rule:", error["rule"])
            print("TF-IDF:", error["tfidf"])
            print("Semantic:", error["semantic"])
            print("Hybrid V3:", error["hybrid_v3"])


if __name__ == "__main__":
    main()