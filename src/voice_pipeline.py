from pathlib import Path
import argparse

import whisper

try:
    from src.text_pipeline import AeroAssistTextPipeline
except ModuleNotFoundError:
    from text_pipeline import AeroAssistTextPipeline


# ==================================================
# AeroAssist - Voice -> Text -> Passenger Response
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

WHISPER_MODEL_NAME = "base"


# ==================================================
# Whisper
# ==================================================

def load_whisper_model():
    print("Loading Whisper...")
    print(f"Model: {WHISPER_MODEL_NAME}")
    print("Device: cpu")

    model = whisper.load_model(
        WHISPER_MODEL_NAME,
        device="cpu",
    )

    print("Whisper loaded successfully.")

    return model


# ==================================================
# Audio transcription
# ==================================================

def transcribe_audio(audio_path, whisper_model):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("\nTranscribing audio...")
    print(f"File: {audio_path.name}")

    result = whisper_model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        fp16=False,
        verbose=False,
    )

    transcript = result["text"].strip()

    return {
        "transcript": transcript,
        "language": result.get(
            "language",
            "en",
        ),
    }


# ==================================================
# Voice -> Frozen Text Pipeline
# ==================================================

def process_voice_query(
    audio_path,
    whisper_model,
    text_pipeline,
):
    transcription = transcribe_audio(
        audio_path,
        whisper_model,
    )

    transcript = transcription["transcript"]

    if not transcript:
        raise ValueError(
            "Whisper returned an empty transcript."
        )

    text_result = text_pipeline.retrieve(
        transcript
    )

    response = text_pipeline.generate_response(
        text_result
    )

    return {
        "audio_file": str(audio_path),
        "language": transcription["language"],
        "transcript": transcript,
        "text_result": text_result,
        "response": response,
    }


# ==================================================
# Display
# ==================================================

def display_result(result):
    text_result = result["text_result"]

    print("\nAeroAssist Voice Analysis")
    print("=========================")

    print(
        "Detected language:",
        result["language"],
    )

    print(
        "Transcript:",
        result["transcript"],
    )

    print("\n--- Text Pipeline Analysis ---")

    print(
        "Gate:",
        text_result["gate"],
    )

    print(
        "Terminal:",
        text_result["terminal"],
    )

    print(
        "Rule intent:",
        text_result["rule_intent"],
    )

    print(
        "TF-IDF intent:",
        text_result["tfidf_intent"],
    )

    print(
        "Semantic intent:",
        text_result["semantic_intent"],
    )

    print(
        "Semantic intent confidence:",
        text_result["semantic_confidence"],
    )

    print(
        "Model disagreement:",
        text_result["model_disagreement"],
    )

    print(
        "Final intent:",
        text_result["intent"],
    )

    print(
        "Urgency:",
        text_result["urgency"],
    )

    print(
        "Passenger mode:",
        text_result["passenger_mode"],
    )

    print(
        "Retrieved record:",
        text_result["record"]["id"],
    )

    print(
        "Location:",
        text_result["record"]["name"],
    )

    print(
        "Semantic similarity:",
        text_result["semantic_similarity"],
    )

    print(
        "Entity match:",
        text_result["entity_match"],
    )

    print(
        "Intent match:",
        text_result["intent_match"],
    )

    print(
        "Rule-semantic agreement:",
        text_result[
            "rule_semantic_agreement"
        ],
    )

    print(
        "Confidence:",
        text_result["confidence"],
    )

    print(
        "Confidence reason:",
        text_result[
            "confidence_reason"
        ],
    )

    print("\nAeroAssist:")
    print(result["response"])

    print("\n=========================")


# ==================================================
# Main
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a spoken airport passenger "
            "query through Whisper and the frozen "
            "AeroAssist text pipeline."
        )
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to an audio file",
    )

    args = parser.parse_args()

    # Load speech model
    whisper_model = load_whisper_model()

    # Load existing frozen text pipeline
    print("\nLoading frozen AeroAssist text pipeline...")

    text_pipeline = AeroAssistTextPipeline()

    print(
        "Frozen AeroAssist text pipeline "
        "loaded successfully."
    )

    # Process complete voice query
    result = process_voice_query(
        args.audio,
        whisper_model,
        text_pipeline,
    )

    display_result(
        result
    )


if __name__ == "__main__":
    main()