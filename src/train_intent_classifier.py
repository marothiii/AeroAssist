import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
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


def train_classifier():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000
    )

    X_train = vectorizer.fit_transform(
        train_df["text"]
    )

    X_test = vectorizer.transform(
        test_df["text"]
    )

    y_train = train_df["intent"]
    y_test = test_df["intent"]

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    print("\nTraining intent classifier...")

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n================================")
    print("ML INTENT CLASSIFIER EVALUATION")
    print("================================")

    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nIncorrect predictions:")

    errors = test_df.copy()
    errors["predicted_intent"] = predictions

    errors = errors[
        errors["intent"]
        != errors["predicted_intent"]
    ]

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

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier
        },
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train_classifier()