from pathlib import Path
import argparse

try:
    from src.vision_inference import AeroAssistVisionInference
    from src.multimodal_fusion import AeroAssistMultimodalFusion
    from src.voice_pipeline import (
        load_whisper_model,
        transcribe_audio,
    )
except ModuleNotFoundError:
    from vision_inference import AeroAssistVisionInference
    from multimodal_fusion import AeroAssistMultimodalFusion
    from voice_pipeline import (
        load_whisper_model,
        transcribe_audio,
    )

try:
    from src.passenger_response import build_passenger_response
except ModuleNotFoundError:
    from passenger_response import build_passenger_response


# ==================================================
# CLIP class -> frozen fusion symbolic label
# ==================================================

VISION_TO_FUSION_LABEL = {
    "gate_sign": "gate",
    "baggage_claim": "baggage_claim",
    "security": "security",
    "restaurant": "restaurant",
    "transport": "train",

    "check_in": None,
    "lounge": None,
    "information": None,
    "accessibility": None,
    "family_facility": "family_facility",
}


class AeroAssistEndToEndPipeline:

    def __init__(
        self,
        use_vision=True,
        use_voice=True,
    ):

        self.vision = None
        self.whisper_model = None

        if use_vision:
            print("Loading real vision inference...")
            self.vision = AeroAssistVisionInference()

        if use_voice:
            print("Loading frozen Whisper model...")
            self.whisper_model = load_whisper_model()

        print("Loading frozen fusion controller...")
        self.fusion = AeroAssistMultimodalFusion()

        print("End-to-end pipeline ready.")

    def adapt_vision_result(
        self,
        vision_result,
    ):

        if not vision_result:
            return ""

        confidence = vision_result.get(
            "confidence",
            "uncertain",
        )

        if confidence == "uncertain":
            return ""

        predicted_class = vision_result.get(
            "predicted_class"
        )

        fusion_label = VISION_TO_FUSION_LABEL.get(
            predicted_class
        )

        if fusion_label is None:
            return ""

        return fusion_label

    def run(
        self,
        text="",
        image_path=None,
        audio_path=None,
        session_context=None,
    ):

        vision_result = None
        transcription = None
        fusion_image_label = ""

        modalities = []

        # ------------------------------------------
        # Real image -> frozen CLIP
        # ------------------------------------------

        if image_path:

            if self.vision is None:
                raise RuntimeError(
                    "Vision model was not loaded."
                )

            modalities.append("image")

            vision_result = self.vision.predict(
                Path(image_path)
            )

            fusion_image_label = (
                self.adapt_vision_result(
                    vision_result
                )
            )

        # ------------------------------------------
        # Real audio -> frozen Whisper
        # ------------------------------------------

        fusion_text = text

        if audio_path:

            if self.whisper_model is None:
                raise RuntimeError(
                    "Whisper model was not loaded."
                )

            modalities.append("voice")

            transcription = transcribe_audio(
                audio_path,
                self.whisper_model,
            )

            transcript = transcription[
                "transcript"
            ]

            if not transcript:
                raise ValueError(
                    "Whisper returned an empty transcript."
                )

            fusion_text = transcript

        # ------------------------------------------
        # Typed text
        # ------------------------------------------

        elif text:
            modalities.append("text")

        # ------------------------------------------
        # Frozen multimodal fusion
        # ------------------------------------------

        fusion_result = self.fusion.fuse(
            text=fusion_text,
            image_label=fusion_image_label,
            modalities=modalities,
            session_context=session_context,
        )

        passenger_response = build_passenger_response(
            fusion_result
        )

        return {
            "input_text": text,
            "audio_path": (
                str(audio_path)
                if audio_path
                else None
            ),
            "transcription": transcription,
            "fusion_text": fusion_text,
            "image_path": (
                str(image_path)
                if image_path
                else None
            ),
            "vision_result": vision_result,
            "fusion_image_label": (
                fusion_image_label
            ),
            "modalities": modalities,
            "passenger_response": passenger_response,
            "fusion_result": fusion_result,
        }


def display_result(result):

    print("\nEND-TO-END RESULT")
    print("=================")

    print(
        "\nModalities:",
        result["modalities"],
    )

    if result["vision_result"] is not None:

        print("\nVision:")
        print(result["vision_result"])

        print(
            "\nFusion image label:",
            result["fusion_image_label"]
            or "WITHHELD",
        )

    if result["transcription"] is not None:

        print("\nVoice:")

        print(
            "Transcript:",
            result["transcription"][
                "transcript"
            ],
        )

        print(
            "Detected language:",
            result["transcription"][
                "language"
            ],
        )

    elif result["input_text"]:

        print(
            "\nText:",
            result["input_text"],
        )

    fusion_result = result["fusion_result"]

    print("\nFusion:")

    print(
        "Record:",
        fusion_result["record"],
    )

    print(
        "Behavior:",
        fusion_result["behavior"],
    )

    print(
        "Confidence:",
        fusion_result["confidence"],
    )

    print(
        "Conflict:",
        fusion_result["conflict"],
    )

    print(
        "Reason:",
        fusion_result["reason"],
    )

    

    print("\nPassenger response:")

    print(
        result["passenger_response"]["message"]
    )

    print(
        "Safe to display record:",
        result["passenger_response"][
            "safe_to_display_record"
        ],
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the AeroAssist real end-to-end "
            "multimodal pipeline."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an airport image.",
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to an audio query.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Typed passenger query.",
    )

    args = parser.parse_args()

    if (
        args.image is None
        and args.audio is None
        and not args.text
    ):
        parser.error(
            "Provide --image, --audio, or --text."
        )

    pipeline = AeroAssistEndToEndPipeline(
        use_vision=(
            args.image is not None
        ),
        use_voice=(
            args.audio is not None
        ),
    )

    result = pipeline.run(
        text=args.text,
        image_path=args.image,
        audio_path=args.audio,
    )

    display_result(result)


if __name__ == "__main__":
    main()