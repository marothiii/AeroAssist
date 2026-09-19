import os
import sys
import pandas as pd

# Allow this script to import text_pipeline.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_pipeline import AeroAssistTextPipeline


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "passenger_queries.csv"
)


print("Loading validation dataset...")

df = pd.read_csv(DATA_PATH)

# We use validation because it is our development set.
validation_df = df[df["split"] == "validation"].copy()

print(f"Validation examples: {len(validation_df)}")


print("\nLoading AeroAssist...")
pipeline = AeroAssistTextPipeline()


results = []

for _, row in validation_df.iterrows():

    query = row["text"]
    expected_intent = row["intent"]

    result = pipeline.retrieve(query)

    predicted_intent = result["intent"]

    correct = predicted_intent == expected_intent

    rule_semantic_agreement = (
        result["rule_intent"] != "unknown"
        and result["rule_intent"]
        == result["semantic_intent"]
    )

    results.append(
        {
            "query": query,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "correct": correct,
            "semantic_confidence":
                result["semantic_confidence"],
            "retrieval_similarity":
                result["semantic_similarity"],
            "rule_semantic_agreement":
                rule_semantic_agreement,
            "entity_match":
                result["entity_match"],
            "model_disagreement":
                result["model_disagreement"],
            "confidence":
                result["confidence"],
            "confidence_reason":
                result["confidence_reason"],
                
        }
    )


results_df = pd.DataFrame(results)


print("\n======================================")
print("CONFIDENCE SIGNAL ANALYSIS")
print("======================================")

correct_df = results_df[results_df["correct"]]
incorrect_df = results_df[~results_df["correct"]]

print("\nCorrect predictions:", len(correct_df))
print("Incorrect predictions:", len(incorrect_df))


print("\n--- Correct predictions: averages ---")

print(
    "Semantic confidence:",
    round(correct_df["semantic_confidence"].mean(), 3)
)

print(
    "Retrieval similarity:",
    round(correct_df["retrieval_similarity"].mean(), 3)
)

print(
    "Rule-semantic agreement rate:",
    round(correct_df["rule_semantic_agreement"].mean(), 3)
)

print(
    "Entity match rate:",
    round(correct_df["entity_match"].mean(), 3)
)

print(
    "Model disagreement rate:",
    round(correct_df["model_disagreement"].mean(), 3)
)


print("\n--- Incorrect predictions: averages ---")

print(
    "Semantic confidence:",
    round(incorrect_df["semantic_confidence"].mean(), 3)
)

print(
    "Retrieval similarity:",
    round(incorrect_df["retrieval_similarity"].mean(), 3)
)

print(
    "Rule-semantic agreement rate:",
    round(incorrect_df["rule_semantic_agreement"].mean(), 3)
)

print(
    "Entity match rate:",
    round(incorrect_df["entity_match"].mean(), 3)
)

print(
    "Model disagreement rate:",
    round(incorrect_df["model_disagreement"].mean(), 3)
)


print("\n--- Incorrect predictions ---")

columns_to_show = [
    "query",
    "expected_intent",
    "predicted_intent",
    "semantic_confidence",
    "retrieval_similarity",
    "rule_semantic_agreement",
    "entity_match",
    "model_disagreement",
    "confidence",
    "confidence_reason",
]

print(
    incorrect_df[columns_to_show].to_string(
        index=False
    )
)




print("\n--- Signal ranges ---")

for label, subset in [
    ("Correct", correct_df),
    ("Incorrect", incorrect_df)
]:
    print(f"\n{label} predictions")

    print(
        "Semantic confidence:",
        "min =", round(subset["semantic_confidence"].min(), 3),
        "median =", round(subset["semantic_confidence"].median(), 3),
        "max =", round(subset["semantic_confidence"].max(), 3)
    )

    print(
        "Retrieval similarity:",
        "min =", round(subset["retrieval_similarity"].min(), 3),
        "median =", round(subset["retrieval_similarity"].median(), 3),
        "max =", round(subset["retrieval_similarity"].max(), 3)
    )


print("\n======================================")
print("FINAL CONFIDENCE POLICY EVALUATION")
print("======================================")

print("\nAll predictions by confidence:")

print(
    results_df.groupby(
        ["correct", "confidence"]
    ).size()
)


print("\nCorrect predictions by confidence:")

print(
    correct_df["confidence"].value_counts()
)


print("\nIncorrect predictions by confidence:")

print(
    incorrect_df["confidence"].value_counts()
)


accepted_df = results_df[
    results_df["confidence"].isin(
        ["confident", "caution"]
    )
]

uncertain_df = results_df[
    results_df["confidence"] == "uncertain"
]


coverage = len(accepted_df) / len(results_df)

selective_accuracy = (
    accepted_df["correct"].mean()
    if len(accepted_df) > 0
    else 0
)

abstention_rate = (
    len(uncertain_df) / len(results_df)
)

unsafe_accepted_errors = len(
    accepted_df[~accepted_df["correct"]]
)


print("\n--- Selective prediction metrics ---")

print(
    "Coverage:",
    round(coverage, 3)
)

print(
    "Selective accuracy:",
    round(selective_accuracy, 3)
)

print(
    "Abstention rate:",
    round(abstention_rate, 3)
)

print(
    "Unsafe accepted errors:",
    unsafe_accepted_errors
)


print("\n--- Correct but uncertain cases ---")

correct_uncertain = results_df[
    (results_df["correct"])
    & (results_df["confidence"] == "uncertain")
]

print(
    correct_uncertain[
        [
            "query",
            "expected_intent",
            "predicted_intent",
            "semantic_confidence",
            "retrieval_similarity",
            "rule_semantic_agreement",
            "entity_match",
            "model_disagreement",
            "confidence_reason",
        ]
    ].to_string(index=False)
)
