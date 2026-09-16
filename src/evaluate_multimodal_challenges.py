from pathlib import Path
import csv
import json

from multimodal_fusion import AeroAssistMultimodalFusion


PROJECT_DIR = Path(__file__).resolve().parent.parent

CHALLENGE_PATH = (
    PROJECT_DIR
    / "data"
    / "multimodal_challenge_set.json"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = (
    RESULTS_DIR
    / "multimodal_challenge_evaluation.csv"
)


def normalise_record(record):
    if record in ("", None):
        return None

    return record


def main():

    print("Loading multimodal challenge set...")

    with open(
        CHALLENGE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        challenge_cases = json.load(file)

    print(
        f"Loaded {len(challenge_cases)} scenarios."
    )

    print("\nLoading fusion controller...")

    fusion = AeroAssistMultimodalFusion()

    print("\nRunning multimodal challenge evaluation...\n")

    results = []

    for case in challenge_cases:

        scenario_id = case["scenario_id"]

        text = case.get("text", "")
        image_label = case.get(
            "image_label",
            "",
        )
        modalities = case.get(
            "modalities",
            [],
        )

        expected_record = normalise_record(
            case.get("expected_record")
        )

        expected_behavior = case.get(
            "expected_behavior"
        )

        # No synthetic session state is injected during this
        # challenge-set evaluation. Journey-memory scenarios
        # test whether the controller recognises that previous
        # context is required rather than inventing a destination.
        session_context = None

        fusion_result = fusion.fuse(
            text=text,
            image_label=image_label,
            modalities=modalities,
            session_context=session_context,
        )

        predicted_record = normalise_record(
            fusion_result.get("record")
        )

        predicted_behavior = (
            fusion_result.get("behavior")
        )

        record_correct = (
            predicted_record
            == expected_record
        )

        behavior_correct = (
            predicted_behavior
            == expected_behavior
        )

        overall_correct = (
            record_correct
            and behavior_correct
        )

        row = {
            "scenario_id": scenario_id,
            "type": case.get("type"),
            "modalities": "+".join(
                modalities
            ),
            "image_label": image_label,
            "text": text,
            "expected_record": (
                expected_record or ""
            ),
            "predicted_record": (
                predicted_record or ""
            ),
            "record_correct": record_correct,
            "expected_behavior": (
                expected_behavior or ""
            ),
            "predicted_behavior": (
                predicted_behavior or ""
            ),
            "behavior_correct": (
                behavior_correct
            ),
            "overall_correct": (
                overall_correct
            ),
            "confidence": fusion_result.get(
                "confidence"
            ),
            "conflict": fusion_result.get(
                "conflict"
            ),
            "reason": fusion_result.get(
                "reason"
            ),
        }

        results.append(row)

        status = (
            "PASS"
            if overall_correct
            else "FAIL"
        )

        print(
            f"{scenario_id} | {status}"
        )

        print(
            "  Expected record:",
            expected_record,
            "| Predicted:",
            predicted_record,
        )

        print(
            "  Expected behavior:",
            expected_behavior,
            "| Predicted:",
            predicted_behavior,
        )

        print(
            "  Confidence:",
            fusion_result.get(
                "confidence"
            ),
        )

        print(
            "  Conflict:",
            fusion_result.get(
                "conflict"
            ),
        )

        print(
            "  Reason:",
            fusion_result.get(
                "reason"
            ),
        )

        print("-" * 65)

    total = len(results)

    record_correct_count = sum(
        row["record_correct"]
        for row in results
    )

    behavior_correct_count = sum(
        row["behavior_correct"]
        for row in results
    )

    overall_correct_count = sum(
        row["overall_correct"]
        for row in results
    )

    record_accuracy = (
        record_correct_count
        / total
    )

    behavior_accuracy = (
        behavior_correct_count
        / total
    )

    overall_accuracy = (
        overall_correct_count
        / total
    )

    confident_count = sum(
        row["confidence"]
        == "confident"
        for row in results
    )

    caution_count = sum(
        row["confidence"]
        == "caution"
        for row in results
    )

    uncertain_count = sum(
        row["confidence"]
        == "uncertain"
        for row in results
    )

    conflict_count = sum(
        bool(row["conflict"])
        for row in results
    )

    failures = [
        row
        for row in results
        if not row["overall_correct"]
    ]

    print(
        "\nMultimodal Challenge Evaluation"
    )
    print(
        "==============================="
    )

    print(
        f"Scenarios: {total}"
    )

    print(
        "Record accuracy:",
        f"{record_accuracy:.3f}",
        f"({record_correct_count}/{total})",
    )

    print(
        "Behavior accuracy:",
        f"{behavior_accuracy:.3f}",
        f"({behavior_correct_count}/{total})",
    )

    print(
        "Exact scenario accuracy:",
        f"{overall_accuracy:.3f}",
        f"({overall_correct_count}/{total})",
    )

    print(
        "\nConfidence counts:"
    )

    print(
        f"  confident: {confident_count}"
    )

    print(
        f"  caution: {caution_count}"
    )

    print(
        f"  uncertain: {uncertain_count}"
    )

    print(
        f"\nDetected conflicts: {conflict_count}"
    )

    print(
        f"Failed scenarios: {len(failures)}"
    )

    if failures:

        print(
            "\nFAILED SCENARIOS"
        )
        print(
            "----------------"
        )

        for row in failures:

            print(
                row["scenario_id"],
                "|",
                row["type"],
            )

            if not row[
                "record_correct"
            ]:
                print(
                    "  Record:",
                    row[
                        "predicted_record"
                    ],
                    "!=",
                    row[
                        "expected_record"
                    ],
                )

            if not row[
                "behavior_correct"
            ]:
                print(
                    "  Behavior:",
                    row[
                        "predicted_behavior"
                    ],
                    "!=",
                    row[
                        "expected_behavior"
                    ],
                )

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
        writer.writerows(
            results
        )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()