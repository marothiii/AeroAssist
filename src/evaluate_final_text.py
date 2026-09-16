import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from text_pipeline import AeroAssistTextPipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "final_text_test.csv",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

DETAILS_PATH = os.path.join(
    RESULTS_DIR,
    "text_final_results.csv",
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "text_summary.csv",
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "text_classification_report.csv",
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "text_confusion_matrix.png",
)

RETRIEVAL_ERRORS_PATH = os.path.join(
    RESULTS_DIR,
    "text_retrieval_errors.csv",
)


# ============================================================
# LOAD TEST SET
# ============================================================

print(
    "Loading final unseen text test set..."
)

df = pd.read_csv(
    DATA_PATH
)

print(
    "Final test examples:",
    len(df),
)


# ============================================================
# LOAD AEROASSIST
# ============================================================

print(
    "\nLoading AeroAssist..."
)

pipeline = (
    AeroAssistTextPipeline()
)


# ============================================================
# EVALUATION
# ============================================================

results = []


for _, row in df.iterrows():

    query = row["text"]

    expected_intent = (
        row["intent"]
    )

    result = pipeline.retrieve(
        query
    )

    predicted_intent = (
        result["intent"]
    )

    correct = (
        predicted_intent
        == expected_intent
    )

    record = result.get(
        "record"
    )

    if isinstance(
        record,
        dict,
    ):
        retrieved_record = (
            record.get("id")
        )
    else:
        retrieved_record = None

    expected_record = None

    # Supports either column name if you already have one.
    if (
        "expected_record"
        in df.columns
    ):
        expected_record = row[
            "expected_record"
        ]

    elif (
        "record_id"
        in df.columns
    ):
        expected_record = row[
            "record_id"
        ]

    # Convert NaN to None.
    if pd.isna(
        expected_record
    ):
        expected_record = None

    if expected_record is not None:

        acceptable_records = [
            record.strip()
            for record in str(expected_record).split("|")
        ]

        retrieval_correct = (
            str(retrieved_record)
            in acceptable_records
        )

    else:

        retrieval_correct = None

    





    results.append(
        {
            "sample_id": (
                row["sample_id"]
            ),
            "query": query,
            "expected_intent": (
                expected_intent
            ),
            "predicted_intent": (
                predicted_intent
            ),
            "correct": correct,
            "confidence": (
                result["confidence"]
            ),
            "confidence_reason": (
                result[
                    "confidence_reason"
                ]
            ),
            "semantic_confidence": (
                result[
                    "semantic_confidence"
                ]
            ),
            "retrieval_similarity": (
                result[
                    "semantic_similarity"
                ]
            ),
            "entity_match": (
                result[
                    "entity_match"
                ]
            ),
            "model_disagreement": (
                result[
                    "model_disagreement"
                ]
            ),
            "retrieved_record": (
                retrieved_record
            ),
            "expected_record": (
                expected_record
            ),
            "retrieval_correct": (
                retrieval_correct
            ),
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# CORE METRICS
# ============================================================

print(
    "\n======================================"
)

print(
    "FINAL TEXT EVALUATION"
)

print(
    "======================================"
)


y_true = (
    results_df[
        "expected_intent"
    ]
)

y_pred = (
    results_df[
        "predicted_intent"
    ]
)


accuracy = accuracy_score(
    y_true,
    y_pred,
)


macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# SELECTIVE / ABSTENTION METRICS
# ============================================================

accepted_df = results_df[
    results_df[
        "confidence"
    ].isin(
        [
            "confident",
            "caution",
        ]
    )
]


uncertain_df = results_df[
    results_df[
        "confidence"
    ]
    == "uncertain"
]


coverage = (
    len(accepted_df)
    / len(results_df)
)


selective_accuracy = (
    accepted_df[
        "correct"
    ].mean()
    if len(
        accepted_df
    ) > 0
    else 0
)


abstention_rate = (
    len(uncertain_df)
    / len(results_df)
)


unsafe_accepted_errors = len(
    accepted_df[
        ~accepted_df[
            "correct"
        ]
    ]
)


# ============================================================
# RETRIEVAL ACCURACY
# ============================================================

retrieval_eval_df = (
    results_df[
        results_df[
            "retrieval_correct"
        ].notna()
    ]
)


if len(
    retrieval_eval_df
) > 0:

    retrieval_accuracy = (
        retrieval_eval_df[
            "retrieval_correct"
        ].mean()
    )

else:

    retrieval_accuracy = (
        np.nan
    )


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print(
    "Intent accuracy:",
    round(
        accuracy,
        3,
    ),
)

print(
    "Macro precision:",
    round(
        macro_precision,
        3,
    ),
)

print(
    "Macro recall:",
    round(
        macro_recall,
        3,
    ),
)

print(
    "Macro F1:",
    round(
        macro_f1,
        3,
    ),
)

print(
    "Weighted precision:",
    round(
        weighted_precision,
        3,
    ),
)

print(
    "Weighted recall:",
    round(
        weighted_recall,
        3,
    ),
)

print(
    "Weighted F1:",
    round(
        weighted_f1,
        3,
    ),
)

print(
    "Coverage:",
    round(
        coverage,
        3,
    ),
)

print(
    "Selective accuracy:",
    round(
        selective_accuracy,
        3,
    ),
)

print(
    "Abstention rate:",
    round(
        abstention_rate,
        3,
    ),
)

print(
    "Unsafe accepted errors:",
    unsafe_accepted_errors,
)


if not np.isnan(
    retrieval_accuracy
):

    print(
        "Retrieval accuracy:",
        round(
            retrieval_accuracy,
            3,
        ),
    )

else:

    print(
        "Retrieval accuracy: "
        "not calculated "
        "(no expected record column)"
    )


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print(
    "\n--- Confidence distribution ---"
)

print(
    results_df[
        "confidence"
    ].value_counts()
)


# ============================================================
# INCORRECT PREDICTIONS
# ============================================================

print(
    "\n--- Incorrect predictions ---"
)

incorrect_df = results_df[
    ~results_df[
        "correct"
    ]
]


if len(
    incorrect_df
) == 0:

    print(
        "None"
    )

else:

    print(
        incorrect_df[
            [
                "sample_id",
                "query",
                "expected_intent",
                "predicted_intent",
                "confidence",
                "confidence_reason",
                "semantic_confidence",
                "retrieval_similarity",
                "model_disagreement",
                "retrieved_record",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# PER-INTENT ACCURACY
# ============================================================

print(
    "\n--- Accuracy by intent ---"
)


intent_accuracy = (
    results_df
    .groupby(
        "expected_intent"
    )["correct"]
    .mean()
    .sort_values(
        ascending=False
    )
)


print(
    intent_accuracy
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    zero_division=0,
    output_dict=True,
)


report_df = (
    pd.DataFrame(
        report
    )
    .transpose()
)


report_df.to_csv(
    REPORT_PATH
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    set(
        y_true.tolist()
    )
    | set(
        y_pred.tolist()
    )
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
)


fig, ax = plt.subplots(
    figsize=(12, 10)
)

ax.imshow(
    cm,
    interpolation="nearest",
)

ax.set_title(
    "AeroAssist Text Intent Confusion Matrix"
)

ax.set_xlabel(
    "Predicted intent"
)

ax.set_ylabel(
    "Expected intent"
)

ax.set_xticks(
    np.arange(
        len(labels)
    )
)

ax.set_yticks(
    np.arange(
        len(labels)
    )
)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(
    labels
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


fig.savefig(
    CONFUSION_PATH,
    dpi=220,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# RETRIEVAL ERRORS
# ============================================================

if len(
    retrieval_eval_df
) > 0:

    retrieval_errors_df = (
        retrieval_eval_df[
            ~retrieval_eval_df[
                "retrieval_correct"
            ]
        ]
    )

else:

    retrieval_errors_df = (
        pd.DataFrame()
    )


retrieval_errors_df.to_csv(
    RETRIEVAL_ERRORS_PATH,
    index=False,
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = [
    {
        "metric": (
            "test_examples"
        ),
        "value": (
            len(results_df)
        ),
    },
    {
        "metric": (
            "intent_accuracy"
        ),
        "value": accuracy,
    },
    {
        "metric": (
            "macro_precision"
        ),
        "value": macro_precision,
    },
    {
        "metric": (
            "macro_recall"
        ),
        "value": macro_recall,
    },
    {
        "metric": (
            "macro_f1"
        ),
        "value": macro_f1,
    },
    {
        "metric": (
            "weighted_precision"
        ),
        "value": weighted_precision,
    },
    {
        "metric": (
            "weighted_recall"
        ),
        "value": weighted_recall,
    },
    {
        "metric": (
            "weighted_f1"
        ),
        "value": weighted_f1,
    },
    {
        "metric": (
            "coverage"
        ),
        "value": coverage,
    },
    {
        "metric": (
            "selective_accuracy"
        ),
        "value": (
            selective_accuracy
        ),
    },
    {
        "metric": (
            "abstention_rate"
        ),
        "value": (
            abstention_rate
        ),
    },
    {
        "metric": (
            "unsafe_accepted_errors"
        ),
        "value": (
            unsafe_accepted_errors
        ),
    },
    {
        "metric": (
            "retrieval_accuracy"
        ),
        "value": (
            retrieval_accuracy
        ),
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df.to_csv(
    DETAILS_PATH,
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n======================================"
)

print(
    "FILES SAVED"
)

print(
    "======================================"
)


for path in [
    DETAILS_PATH,
    SUMMARY_PATH,
    REPORT_PATH,
    CONFUSION_PATH,
    RETRIEVAL_ERRORS_PATH,
]:

    print(
        os.path.relpath(
            path,
            BASE_DIR,
        )
    )


print(
    "\nFinal text evaluation complete."
)