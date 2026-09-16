from pathlib import Path
import csv

from end_to_end_pipeline import AeroAssistEndToEndPipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


CASES = [
    # --------------------------------------------------
    # Voice-only clean
    # --------------------------------------------------
    {
        "case_id": "E2E001",
        "type": "voice_clean",
        "image": None,
        "audio": "audio/speech_01.m4a",
        "text": "",
        "expected_record": "NIA006",
        "expected_behavior": "retrieve",
    },
    {
        "case_id": "E2E002",
        "type": "voice_clean",
        "image": None,
        "audio": "audio/speech_02.m4a",
        "text": "",
        "expected_record": "NIA009",
        "expected_behavior": "retrieve",
    },
    {
        "case_id": "E2E003",
        "type": "voice_clean",
        "image": None,
        "audio": "audio/speech_03.m4a",
        "text": "",
        "expected_record": "NIA024",
        "expected_behavior": "human_handover",
    },

    # --------------------------------------------------
    # Image + text agreement
    # --------------------------------------------------
    {
        "case_id": "E2E004",
        "type": "image_text_agreement",
        "image": "images/gate_sign/gate_sign_001.jpg",
        "audio": None,
        "text": "Where is Gate B12?",
        "expected_record": "NIA006",
        "expected_behavior": "retrieve",
    },

    # --------------------------------------------------
    # Image + text contradiction
    # --------------------------------------------------
    {
        "case_id": "E2E005",
        "type": "image_text_conflict",
        "image": "images/baggage_claim/baggage_claim_001.jpg",
        "audio": None,
        "text": "Is this the way to Gate B12?",
        "expected_record": "NIA006",
        "expected_behavior": "conflict_warning",
    },

    # --------------------------------------------------
    # Low-confidence wrong vision should be withheld
    # --------------------------------------------------
    {
        "case_id": "E2E006",
        "type": "vision_uncertainty",
        "image": "images/baggage_claim/baggage_claim_009.jpg",
        "audio": None,
        "text": "Where is Gate B12?",
        "expected_record": "NIA006",
        "expected_behavior": "retrieve",
    },

    # --------------------------------------------------
    # Image + voice contradiction
    # --------------------------------------------------
    {
        "case_id": "E2E007",
        "type": "image_voice_conflict",
        "image": "images/baggage_claim/baggage_claim_001.jpg",
        "audio": "audio/speech_01.m4a",
        "text": "",
        "expected_record": "NIA006",
        "expected_behavior": "conflict_warning",
    },

    # --------------------------------------------------
    # Moderate-noise voice
    # --------------------------------------------------
    {
        "case_id": "E2E008",
        "type": "voice_moderate_noise",
        "image": None,
        "audio": "audio/speech_01_moderate.m4a",
        "text": "",
        "expected_record": "NIA006",
        "expected_behavior": "retrieve",
    },

    # --------------------------------------------------
    # Severe-noise voice
    # Expected safe abstention / uncertainty.
    # We do not require a specific KB record.
    # --------------------------------------------------
    {
        "case_id": "E2E009",
        "type": "voice_severe_noise",
        "image": None,
        "audio": "audio/speech_01_noisy.m4a",
        "text": "",
        "expected_record": None,
        "expected_behavior": None,
        "expected_confidence": "uncertain",
    },
]


def normalise_record(record):
    if record in ("", None):
        return None
    return record


def main():

    print("Loading AeroAssist end-to-end pipeline...")

    pipeline = AeroAssistEndToEndPipeline(
        use_vision=True,
        use_voice=True,
    )

    rows = []

    print("\nRunning end-to-end evaluation...\n")

    for case in CASES:

        image_path = (
            PROJECT_DIR / case["image"]
            if case["image"]
            else None
        )

        audio_path = (
            PROJECT_DIR / case["audio"]
            if case["audio"]
            else None
        )

        result = pipeline.run(
            text=case["text"],
            image_path=image_path,
            audio_path=audio_path,
        )

        fusion = result["fusion_result"]

        predicted_record = normalise_record(
            fusion["record"]
        )

        expected_record = normalise_record(
            case.get("expected_record")
        )

        expected_behavior = case.get(
            "expected_behavior"
        )

        expected_confidence = case.get(
            "expected_confidence"
        )

        record_correct = (
            predicted_record == expected_record
            if expected_record is not None
            else None
        )

        behavior_correct = (
            fusion["behavior"] == expected_behavior
            if expected_behavior is not None
            else None
        )

        confidence_correct = (
            fusion["confidence"] == expected_confidence
            if expected_confidence is not None
            else None
        )

        if expected_record is not None and expected_behavior is not None:
            exact_correct = (
                record_correct
                and behavior_correct
            )
        elif expected_confidence is not None:
            exact_correct = confidence_correct
        else:
            exact_correct = None

        vision_result = result.get(
            "vision_result"
        )

        transcription = result.get(
            "transcription"
        )

        row = {
            "case_id": case["case_id"],
            "type": case["type"],
            "modalities": "+".join(
                result["modalities"]
            ),
            "transcript": (
                transcription["transcript"]
                if transcription
                else ""
            ),
            "vision_predicted_class": (
                vision_result["predicted_class"]
                if vision_result
                else ""
            ),
            "vision_score": (
                vision_result["score"]
                if vision_result
                else ""
            ),
            "vision_confidence": (
                vision_result["confidence"]
                if vision_result
                else ""
            ),
            "fusion_image_label": (
                result["fusion_image_label"]
            ),
            "expected_record": (
                expected_record
                if expected_record
                else ""
            ),
            "predicted_record": (
                predicted_record
                if predicted_record
                else ""
            ),
            "expected_behavior": (
                expected_behavior
                if expected_behavior
                else ""
            ),
            "predicted_behavior": (
                fusion["behavior"]
            ),
            "fusion_confidence": (
                fusion["confidence"]
            ),
            "conflict": fusion["conflict"],
            "record_correct": record_correct,
            "behavior_correct": behavior_correct,
            "confidence_correct": confidence_correct,
            "exact_correct": exact_correct,
        }

        rows.append(row)

        print(
            f'{case["case_id"]} | '
            f'{case["type"]}'
        )

        print(
            "  Record:",
            predicted_record,
        )

        print(
            "  Behavior:",
            fusion["behavior"],
        )

        print(
            "  Confidence:",
            fusion["confidence"],
        )

        print(
            "  Conflict:",
            fusion["conflict"],
        )

        if transcription:
            print(
                "  Transcript:",
                transcription["transcript"],
            )

        if vision_result:
            print(
                "  Vision:",
                vision_result["predicted_class"],
                "|",
                vision_result["confidence"],
            )

        print(
            "  Exact:",
            exact_correct,
        )

        print(
            "----------------------------------------"
        )

    output_path = (
        RESULTS_DIR
        / "end_to_end_evaluation.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    scored = [
        row
        for row in rows
        if row["exact_correct"] is not None
    ]

    passed = sum(
        bool(row["exact_correct"])
        for row in scored
    )

    print("\nEND-TO-END EVALUATION SUMMARY")
    print("=============================")

    print(
        "Scored cases:",
        len(scored),
    )

    print(
        "Passed:",
        passed,
    )

    print(
        "Exact accuracy:",
        round(
            passed / len(scored),
            3,
        ),
    )

    print(
        "\nSaved results to:",
        output_path,
    )


if __name__ == "__main__":
    main()