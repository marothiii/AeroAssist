from pathlib import Path
import csv

import whisper
from jiwer import wer

from text_pipeline import AeroAssistTextPipeline


# ==================================================
# Paths
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "audio"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "speech_clean_evaluation.csv"

WHISPER_MODEL_NAME = "base"


# ==================================================
# Frozen clean speech evaluation set
# ==================================================

TEST_CASES = [
    {
        "id": "SPEECH001",
        "filename": "speech_01.m4a",
        "reference": "Where is Gate B12?",
        "expected_intent": "find_gate",
        "expected_gate": "B12",
    },
    {
        "id": "SPEECH002",
        "filename": "speech_02.m4a",
        "reference": "Where is the nearest security checkpoint?",
        "expected_intent": "find_security",
        "expected_gate": None,
    },
    {
        "id": "SPEECH003",
        "filename": "speech_03.m4a",
        "reference": "I lost my passport.",
        "expected_intent": "travel_document_help",
        "expected_gate": None,
    },
    {
        "id": "SPEECH004",
        "filename": "speech_04.m4a",
        "reference": "Where can I change my baby's diaper?",
        "expected_intent": "family_assistance",
        "expected_gate": None,
    },
    {
        "id": "SPEECH005",
        "filename": "speech_05.m4a",
        "reference": "How do I get to the train station?",
        "expected_intent": "find_transport",
        "expected_gate": None,
    },
]


# ==================================================
# Whisper transcription
# ==================================================

def transcribe_audio(audio_path, model):
    result = model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        fp16=False,
        verbose=False,
    )

    return result["text"].strip()


# ==================================================
# Evaluation
# ==================================================

def main():

    print("Loading Whisper...")
    whisper_model = whisper.load_model(
        WHISPER_MODEL_NAME,
        device="cpu",
    )
    print("Whisper loaded.")

    print("\nLoading frozen AeroAssist text pipeline...")
    text_pipeline = AeroAssistTextPipeline()
    print("Text pipeline loaded.")

    results = []

    print("\nEvaluating clean speech...\n")

    for case in TEST_CASES:

        audio_path = AUDIO_DIR / case["filename"]

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing audio file: {audio_path}"
            )

        transcript = transcribe_audio(
            audio_path,
            whisper_model,
        )

        sample_wer = wer(
            case["reference"].lower(),
            transcript.lower(),
        )

        text_result = text_pipeline.retrieve(
            transcript
        )

        predicted_intent = text_result["intent"]
        predicted_gate = text_result["gate"]

        intent_correct = (
            predicted_intent
            == case["expected_intent"]
        )

        if case["expected_gate"] is None:
            entity_correct = True
        else:
            entity_correct = (
                predicted_gate
                == case["expected_gate"]
            )

        downstream_correct = (
            intent_correct
            and entity_correct
        )

        row = {
            "audio_id": case["id"],
            "filename": case["filename"],
            "condition": "clean",
            "reference": case["reference"],
            "transcript": transcript,
            "wer": round(sample_wer, 4),
            "expected_intent": case["expected_intent"],
            "predicted_intent": predicted_intent,
            "intent_correct": intent_correct,
            "expected_gate": case["expected_gate"],
            "predicted_gate": predicted_gate,
            "entity_correct": entity_correct,
            "downstream_correct": downstream_correct,
            "confidence": text_result["confidence"],
            "retrieved_record": text_result[
                "record"
            ]["id"],
        }

        results.append(row)

        print(case["id"])
        print("Reference:", case["reference"])
        print("Transcript:", transcript)
        print(f"WER: {sample_wer:.3f}")
        print(
            "Intent:",
            predicted_intent,
            "| Correct:",
            intent_correct,
        )
        print(
            "Gate:",
            predicted_gate,
            "| Correct:",
            entity_correct,
        )
        print(
            "Confidence:",
            text_result["confidence"],
        )
        print(
            "Downstream correct:",
            downstream_correct,
        )
        print("-" * 50)


    # ==================================================
    # Overall metrics
    # ==================================================

    mean_wer = sum(
        row["wer"]
        for row in results
    ) / len(results)

    intent_accuracy = sum(
        row["intent_correct"]
        for row in results
    ) / len(results)

    downstream_accuracy = sum(
        row["downstream_correct"]
        for row in results
    ) / len(results)


    print("\nClean Speech Evaluation")
    print("=======================")

    print(
        f"Samples: {len(results)}"
    )

    print(
        f"Mean WER: {mean_wer:.3f}"
    )

    print(
        f"Intent accuracy: "
        f"{intent_accuracy:.3f}"
    )

    print(
        f"Downstream accuracy: "
        f"{downstream_accuracy:.3f}"
    )


    # ==================================================
    # Save CSV
    # ==================================================

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()