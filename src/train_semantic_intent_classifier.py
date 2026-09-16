import os
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer

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
    "semantic_intent_classifier.joblib"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def train_semantic_classifier():

    print("Loading passenger dataset...")

    df = pd.read_csv(DATA_PATH)

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    print("\nLoading sentence-transformer model...")

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    print("Creating training embeddings...")

    X_train = embedding_model.encode(
        train_df["text"].tolist(),
        show_progress_bar=True
    )

    print("Creating validation embeddings...")

    X_validation = embedding_model.encode(
        validation_df["text"].tolist(),
        show_progress_bar=True
    )

    y_train = train_df["intent"]
    y_validation = validation_df["intent"]

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    print("\nTraining semantic intent classifier...")

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    print("\n====================================")
    print("SEMANTIC INTENT VALIDATION")
    print("====================================")

    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0
        )
    )

    errors = validation_df.copy()

    errors["predicted_intent"] = predictions

    errors = errors[
        errors["intent"]
        != errors["predicted_intent"]
    ]

    print("\n====================================")
    print("SEMANTIC CLASSIFIER ERRORS")
    print("====================================")

    if errors.empty:
        print("No validation errors.")

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
            "classifier": classifier,
            "model_name": MODEL_NAME
        },
        MODEL_PATH
    )

    print(
        f"\nSemantic classifier saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    train_semantic_classifier()