from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

VISION_CONFIDENT_SCORE = 0.70
VISION_CONFIDENT_MARGIN = 0.05

VISION_CAUTION_SCORE = 0.50
VISION_CAUTION_MARGIN = 0.10


CLASS_PROMPTS = {
    "gate_sign": [
        "an airport boarding gate with a gate number sign",
        "an airport departure gate sign showing a gate number",
        "a numbered boarding gate inside an airport terminal",
        "airport signage directing passengers to boarding gates",
    ],
    "baggage_claim": [
        "an airport baggage claim carousel with luggage",
        "an airport baggage reclaim conveyor belt",
        "passengers collecting luggage at an airport baggage carousel",
        "an airport arrivals baggage claim area",
    ],
    "security": [
        "an airport passenger security screening checkpoint",
        "airport security screening with metal detectors and x-ray machines",
        "passengers going through an airport security checkpoint",
        "an airport security inspection and screening area",
    ],
    "check_in": [
        "an airport airline check-in counter",
        "airport check-in desks with airline counters",
        "passengers checking in for a flight at an airport counter",
        "an airport departure check-in hall with check-in desks",
    ],
    "lounge": [
        "an airport passenger lounge with seating",
        "an airline airport lounge waiting area",
        "an airport VIP lounge with chairs and tables",
        "an airport lounge interior for waiting passengers",
    ],
    "restaurant": [
        "an airport restaurant or cafe",
        "an airport food court with restaurants",
        "a cafe or dining area inside an airport terminal",
        "airport food and beverage shops with tables and seating",
    ],
    "information": [
        "an airport information desk with an information sign",
        "an airport passenger information counter",
        "an airport help desk for passenger information",
        "an airport information service counter inside a terminal",
    ],
    "transport": [
        "an airport train station or railway platform",
        "an airport taxi rank or taxi pickup area",
        "an airport bus stop or shuttle transport area",
        "airport ground transportation including trains buses and taxis",
    ],
    "accessibility": [
        "an airport accessibility facility for passengers with reduced mobility",
        "wheelchair assistance and accessible facilities at an airport",
        "an airport accessibility sign or accessible passenger service",
        "special assistance equipment for disabled airport passengers",
    ],
    "family_facility": [
        "an airport children's play area",
        "an airport baby care or nursing room",
        "an airport family facility for parents and children",
        "a baby changing room or children's facility inside an airport",
    ],
}


def vision_confidence_label(score, margin):
    if (
        score >= VISION_CONFIDENT_SCORE
        and margin >= VISION_CONFIDENT_MARGIN
    ):
        return "confident"

    if (
        score >= VISION_CAUTION_SCORE
        and margin >= VISION_CAUTION_MARGIN
    ):
        return "caution"

    return "uncertain"


def extract_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        "Unexpected CLIP output type: "
        f"{type(output)}"
    )


class AeroAssistVisionInference:

    def __init__(self):

        print(f"Loading frozen CLIP model on {DEVICE}...")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = CLIPModel.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)

        self.model.eval()

        self.class_names = list(
            CLASS_PROMPTS.keys()
        )

        print(
            "Creating frozen prompt-ensemble embeddings..."
        )

        self.class_embeddings = (
            self._build_class_embeddings()
        )

        print("Vision inference ready.")

    def _build_class_embeddings(self):

        class_embeddings = []

        with torch.no_grad():

            for class_name in self.class_names:

                prompts = CLASS_PROMPTS[
                    class_name
                ]

                text_inputs = self.processor(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                )

                text_inputs = {
                    key: value.to(DEVICE)
                    for key, value
                    in text_inputs.items()
                }

                output = self.model.get_text_features(
                    **text_inputs
                )

                features = extract_feature_tensor(
                    output
                )

                features = F.normalize(
                    features,
                    dim=-1,
                )

                class_feature = features.mean(
                    dim=0,
                    keepdim=True,
                )

                class_feature = F.normalize(
                    class_feature,
                    dim=-1,
                )

                class_embeddings.append(
                    class_feature
                )

        return torch.cat(
            class_embeddings,
            dim=0,
        )

    def predict(self, image_path):

        image_path = Path(image_path)

        with Image.open(image_path) as raw_image:

            image = ImageOps.exif_transpose(
                raw_image
            ).convert("RGB")

        image_inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        image_inputs = {
            key: value.to(DEVICE)
            for key, value
            in image_inputs.items()
        }

        with torch.no_grad():

            output = self.model.get_image_features(
                **image_inputs
            )

            image_features = extract_feature_tensor(
                output
            )

            image_features = F.normalize(
                image_features,
                dim=-1,
            )

            similarities = (
                image_features
                @ self.class_embeddings.T
            )[0]

            probabilities = torch.softmax(
                similarities * 100.0,
                dim=0,
            )

        ranked_indices = torch.argsort(
            probabilities,
            descending=True,
        )

        top1_index = ranked_indices[0].item()
        second_index = ranked_indices[1].item()

        top1_score = probabilities[
            top1_index
        ].item()

        second_score = probabilities[
            second_index
        ].item()

        top1_similarity = similarities[
            top1_index
        ].item()

        second_similarity = similarities[
            second_index
        ].item()

        confidence_margin = (
            top1_score - second_score
        )

        similarity_margin = (
            top1_similarity
            - second_similarity
        )

        return {
            "predicted_class": (
                self.class_names[
                    top1_index
                ]
            ),
            "score": top1_score,
            "similarity": top1_similarity,
            "second_class": (
                self.class_names[
                    second_index
                ]
            ),
            "second_score": second_score,
            "second_similarity": (
                second_similarity
            ),
            "margin": confidence_margin,
            "similarity_margin": (
                similarity_margin
            ),
            "confidence": (
                vision_confidence_label(
                    top1_score,
                    confidence_margin,
                )
            ),
        }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python "
            "src/vision_inference.py "
            "<image_path>"
        )
        raise SystemExit(1)

    vision = AeroAssistVisionInference()

    result = vision.predict(
        sys.argv[1]
    )

    print("\nVISION RESULT")
    print("=============")

    for key, value in result.items():
        print(f"{key}: {value}")