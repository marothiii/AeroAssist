from pathlib import Path
import re

try:
    from src.text_pipeline import AeroAssistTextPipeline
except ModuleNotFoundError:
    from text_pipeline import AeroAssistTextPipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Frozen modality confidence policy
# ---------------------------------------------------------

TEXT_CONFIDENCE_WEIGHT = {
    "confident": 1.00,
    "caution": 0.65,
    "uncertain": 0.30,
}

IMAGE_CONFIDENCE = {
    "gate": 0.85,
    "gate_b12": 0.95,
    "gate_a8": 0.95,
    "gate_b9": 0.95,
    "baggage_claim": 0.90,
    "baggage_claim_t2": 0.95,
    "security": 0.90,
    "train": 0.90,
    "restaurant": 0.90,
    "boarding_pass": 0.90,

    "family_facility": 0.90,

    # Deliberately weak visual evidence
    "blurry_gate": 0.35,
    "unclear_symbol": 0.25,
    "low_confidence_information": 0.30,
}


# ---------------------------------------------------------
# Image semantics
# ---------------------------------------------------------

IMAGE_CATEGORY = {
    "gate": "gate",
    "gate_b12": "gate",
    "gate_a8": "gate",
    "gate_b9": "gate",
    "blurry_gate": "gate",

    "baggage_claim": "baggage",
    "baggage_claim_t2": "baggage",

    "security": "security",
    "train": "transport",
    "restaurant": "restaurant",
    "boarding_pass": "travel_document",

    "family_facility": "family",

    "unclear_symbol": "unknown",
    "low_confidence_information": "information",
}


IMAGE_RECORD = {
    "gate_b12": "NIA006",

    # Gate A8 is a known KB location.
    "gate_a8": "NIA002",

    "train": "NIA021",

    "family_facility": "NIA019",
}


# ---------------------------------------------------------
# Text helpers
# ---------------------------------------------------------

def normalise_text(text):
    if not text:
        return ""

    return text.strip()


def lower_text(text):
    return normalise_text(text).lower()


def contains_any(text, phrases):
    text = lower_text(text)

    return any(
        phrase in text
        for phrase in phrases
    )


def extract_gate_from_text(text):
    if not text:
        return None

    match = re.search(
        r"\b([AB]\s?\d{1,2})\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).replace(" ", "").upper()


def translate_small_multilingual_queries(text):
    """
    Very small deterministic multilingual compatibility layer.

    This is NOT claimed as a general translation system.
    It only normalises a few airport phrases represented in
    the designed challenge set.
    """

    cleaned = lower_text(text)

    replacements = {
        "wo ist gate": "where is gate",
        "wo ist der bahnhof": "where is the train station",
        "wo ist bahnhof": "where is the train station",
    }

    for source, target in replacements.items():
        if source in cleaned:
            cleaned = cleaned.replace(
                source,
                target,
            )

    return cleaned


# ---------------------------------------------------------
# Semantic categories for conflict detection
# ---------------------------------------------------------

def infer_text_category(text, text_result=None):
    cleaned = lower_text(text)

    if contains_any(
        cleaned,
        [
            "security",
            "checkpoint",
        ],
    ):
        return "security"

    if contains_any(
        cleaned,
        [
            "baggage claim",
            "baggage reclaim",
        ],
    ):
        return "baggage"

    if contains_any(
        cleaned,
        [
            "gate ",
            "gate b",
            "gate a",
        ],
    ):
        return "gate"

    if contains_any(
        cleaned,
        [
            "train",
            "station",
            "bahnhof",
        ],
    ):
        return "transport"

    if contains_any(
        cleaned,
        [
            "restaurant",
            "food",
            "cafe",
        ],
    ):
        return "restaurant"

    if text_result:
        intent = text_result.get("intent")

        intent_to_category = {
            "find_gate": "gate",
            "find_baggage": "baggage",
            "find_security": "security",
            "find_transport": "transport",
            "find_restaurant": "restaurant",
            "find_lounge": "lounge",
            "find_information": "information",
        }

        return intent_to_category.get(intent)

    return None


# ---------------------------------------------------------
# Safety/context detection
# ---------------------------------------------------------

def detect_live_data_request(text):
    return contains_any(
        text,
        [
            "has my gate changed",
            "gate changed",
            "flight delayed",
            "is my flight delayed",
            "flight cancelled",
            "flight canceled",
        ],
    )


def detect_emergency(text):
    return contains_any(
        text,
        [
            "collapsed",
            "unconscious",
            "medical emergency",
            "can't breathe",
            "cannot breathe",
        ],
    )


def detect_passport_loss(text):
    return contains_any(
        text,
        [
            "lost my passport",
            "passport is missing",
            "missing passport",
        ],
    )


def detect_ordinary_lost_property(text):
    cleaned = lower_text(text)

    if detect_passport_loss(cleaned):
        return False

    return contains_any(
        cleaned,
        [
            "lost my headphones",
            "lost my phone",
            "lost my bag",
            "lost my wallet",
            "lost my belongings",
        ],
    )


def detect_accessibility(text):
    return contains_any(
        text,
        [
            "can't use stairs",
            "cannot use stairs",
            "step-free",
            "step free",
            "cannot walk far",
            "can't walk far",
            "wheelchair",
            "reduced mobility",
        ],
    )


def detect_family_need(text):
    return contains_any(
        text,
        [
            "baby",
            "diaper",
            "nappy",
            "family facility",
            "change my baby",
        ],
    )


def detect_urgency(text):
    return contains_any(
        text,
        [
            "boarding closes soon",
            "boards soon",
            "boarding soon",
            "short connection",
            "very short connection",
            "running late",
        ],
    )


def detect_transfer(text):
    return contains_any(
        text,
        [
            "short connection",
            "connection",
            "transfer",
            "connecting flight",
        ],
    )


def detect_opening_hours(text):
    return contains_any(
        text,
        [
            "still open",
            "opening hours",
            "what time does",
            "when does",
            "open?",
        ],
    )


# ---------------------------------------------------------
# Main fusion engine
# ---------------------------------------------------------

class AeroAssistMultimodalFusion:

    def __init__(self):

        print("Loading frozen AeroAssist text pipeline...")

        self.text_pipeline = AeroAssistTextPipeline()

        print("Multimodal fusion controller ready.")

    def analyse_text(self, text):

        if not text:
            return None

        pipeline_text = translate_small_multilingual_queries(
            text
        )

        result = self.text_pipeline.retrieve(
            pipeline_text
        )

        return result

    def analyse_image_label(self, image_label):

        if not image_label:
            return {
                "label": None,
                "category": None,
                "record": None,
                "confidence": 0.0,
            }

        return {
            "label": image_label,
            "category": IMAGE_CATEGORY.get(
                image_label,
                "unknown",
            ),
            "record": IMAGE_RECORD.get(
                image_label
            ),
            "confidence": IMAGE_CONFIDENCE.get(
                image_label,
                0.50,
            ),
        }

    def fuse(
        self,
        text="",
        image_label="",
        modalities=None,
        session_context=None,
    ):

        if modalities is None:
            modalities = []

        text = normalise_text(text)

        text_result = self.analyse_text(
            text
        ) if text else None

        image_result = self.analyse_image_label(
            image_label
        )

        result = {
            "record": None,
            "behavior": None,
            "confidence": "uncertain",
            "conflict": False,
            "reason": None,
            "text_result": text_result,
            "image_result": image_result,
        }

        # =================================================
        # 1. Critical safety always overrides other evidence
        # =================================================

        if detect_emergency(text):

            result.update(
                {
                    "record": "NIA023",
                    "behavior": "emergency_handover",
                    "confidence": "confident",
                    "reason": (
                        "Critical medical language overrides "
                        "ordinary retrieval."
                    ),
                }
            )

            return result

        if detect_passport_loss(text):

            result.update(
                {
                    "record": "NIA024",
                    "behavior": "human_handover",
                    "confidence": "confident",
                    "reason": (
                        "Lost passport requires dedicated "
                        "travel-document human assistance."
                    ),
                }
            )

            return result

        # =================================================
        # 2. Live operational data safety
        # =================================================

        if detect_live_data_request(text):

            result.update(
                {
                    "record": None,
                    "behavior": "official_source_redirect",
                    "confidence": "confident",
                    "reason": (
                        "Live flight or gate status must be "
                        "verified using an official live source."
                    ),
                }
            )

            return result

        # =================================================
        # 3. Privacy-sensitive visual input
        # =================================================

        if image_label == "boarding_pass":

            result.update(
                {
                    "record": None,
                    "behavior": (
                        "privacy_warning_and_minimise"
                    ),
                    "confidence": "caution",
                    "reason": (
                        "Boarding passes may contain personal "
                        "or travel-identifying information."
                    ),
                }
            )

            return result

        # =================================================
        # 4. Journey/session memory
        # =================================================

        if lower_text(text) in {
            "and after that?",
            "what next?",
            "and then?",
        }:

            if session_context:
                result.update(
                    {
                        "record": session_context.get(
                            "next_record"
                        ),
                        "behavior": "use_session_context",
                        "confidence": "caution",
                        "reason": (
                            "Current query depends on previous "
                            "journey context."
                        ),
                    }
                )

            else:
                result.update(
                    {
                        "record": None,
                        "behavior": "use_session_context",
                        "confidence": "uncertain",
                        "reason": (
                            "Previous journey context is required "
                            "to resolve this query."
                        ),
                    }
                )

            return result

        # =================================================
        # 5. Transfer urgency
        # =================================================

        if (
            detect_transfer(text)
            and detect_urgency(text)
        ):

            result.update(
                {
                    "record": "NIA025",
                    "behavior": "urgent_transfer_help",
                    "confidence": "confident",
                    "reason": (
                        "Short-connection language indicates "
                        "urgent transfer assistance."
                    ),
                }
            )

            return result

        # =================================================
        # 6. Family context
        # =================================================

        if detect_family_need(text):

            if detect_urgency(text):

                behavior = (
                    "nearby_family_recommendation"
                )

            else:

                behavior = "family_recommendation"

            result.update(
                {
                    "record": "NIA019",
                    "behavior": behavior,
                    "confidence": (
                        text_result["confidence"]
                        if text_result
                        else "caution"
                    ),
                    "reason": (
                        "Family-care need has priority over "
                        "the surrounding visual location."
                    ),
                }
            )

            return result

        # =================================================
        # 7. Accessibility context
        # =================================================

        if detect_accessibility(text):

            gate = extract_gate_from_text(
                text
            )

            if gate == "B12":

                behavior = (
                    "urgent_accessible_route"
                    if detect_urgency(text)
                    else "accessible_route"
                )

                result.update(
                    {
                        "record": "NIA006",
                        "behavior": behavior,
                        "confidence": "confident",
                        "reason": (
                            "Exact Gate B12 entity combined "
                            "with accessibility context."
                        ),
                    }
                )

                return result

            if contains_any(
                text,
                [
                    "train",
                    "station",
                ],
            ):

                result.update(
                    {
                        "record": "NIA021",
                        "behavior": "accessible_route",
                        "confidence": "confident",
                        "reason": (
                            "Transport request combined with "
                            "step-free accessibility need."
                        ),
                    }
                )

                return result

            # Image supplies destination while text supplies
            # accessibility requirement.
            if image_result["record"]:

                result.update(
                    {
                        "record": image_result["record"],
                        "behavior": "accessible_route",
                        "confidence": "confident",
                        "reason": (
                            "Image supplies destination while "
                            "text supplies accessibility context."
                        ),
                    }
                )

                return result

            if contains_any(
                text,
                [
                    "cannot walk far",
                    "can't walk far",
                    "wheelchair",
                    "reduced mobility",
                ],
            ):

                result.update(
                    {
                        "record": "NIA017",
                        "behavior": "assistance_route",
                        "confidence": "confident",
                        "reason": (
                            "Passenger mobility limitation "
                            "requires airport assistance."
                        ),
                    }
                )

                return result

        # =================================================
        # 8. Ordinary lost property
        # =================================================

        if detect_ordinary_lost_property(text):

            result.update(
                {
                    "record": "NIA016",
                    "behavior": "retrieve",
                    "confidence": "confident",
                    "reason": (
                        "Ordinary lost property is separated "
                        "from lost travel documents."
                    ),
                }
            )

            return result

        # =================================================
        # 9. Lounge opening-hours query
        # =================================================

        if (
            detect_opening_hours(text)
            and contains_any(
                text,
                ["lounge"],
            )
        ):

            result.update(
                {
                    "record": "NIA013",
                    "behavior": "retrieve_hours",
                    "confidence": "confident",
                    "reason": (
                        "Opening-hours request targets the "
                        "airport lounge."
                    ),
                }
            )

            return result

        # =================================================
        # 10. Journey-stage inference
        # =================================================

        if contains_any(
            text,
            [
                "finished international check-in",
                "finished check-in",
                "just checked in",
            ],
        ):

            result.update(
                {
                    "record": "NIA009",
                    "behavior": "infer_next_stage",
                    "confidence": "caution",
                    "reason": (
                        "After check-in, the next modeled "
                        "journey stage is security."
                    ),
                }
            )

            return result

        # Baggage -> taxi next-step context
        if (
            image_label == "baggage_claim_t2"
            and contains_any(
                text,
                [
                    "taxi",
                    "after i get my bag",
                    "after i get my baggage",
                ],
            )
        ):

            result.update(
                {
                    "record": "NIA022",
                    "behavior": "journey_next_step",
                    "confidence": "confident",
                    "reason": (
                        "Image establishes baggage-claim "
                        "journey stage and text requests taxi "
                        "as the next step."
                    ),
                }
            )

            return result

        # =================================================
        # 11. Weak image + ambiguous text
        # =================================================

        if (
            image_result["confidence"] < 0.50
            and (
                not text
                or lower_text(text) in {
                    "help?",
                    "where do i go?",
                }
            )
        ):

            result.update(
                {
                    "record": None,
                    "behavior": (
                        "request_clearer_input"
                        if not text
                        else "clarify"
                    ),
                    "confidence": "uncertain",
                    "reason": (
                        "Neither modality provides sufficient "
                        "reliable evidence."
                    ),
                }
            )

            return result

        # =================================================
        # 12. Strong image resolves ambiguous text
        # =================================================

        if (
            lower_text(text)
            in {
                "where do i go?",
                "where is this?",
                "what is this?",
            }
            and image_result["confidence"] >= 0.70
            and image_result["record"]
        ):

            result.update(
                {
                    "record": image_result["record"],
                    "behavior": "weight_image_more",
                    "confidence": "confident",
                    "reason": (
                        "Text is ambiguous but visual evidence "
                        "is specific and strong."
                    ),
                }
            )

            return result

        # =================================================
        # 13. Contextual service near an image location
        # =================================================

        if (
            image_label == "gate_a8"
            and contains_any(
                text,
                [
                    "somewhere quiet",
                    "quiet near here",
                    "quiet place",
                ],
            )
        ):

            result.update(
                {
                    "record": "NIA013",
                    "behavior": (
                        "contextual_nearby_service"
                    ),
                    "confidence": "caution",
                    "reason": (
                        "Image provides current gate context "
                        "while voice/text requests a nearby "
                        "quiet service."
                    ),
                }
            )

            return result

        # =================================================
        # 14. Cross-modal contradiction
        # =================================================

        text_category = infer_text_category(
            text,
            text_result,
        )

        image_category = image_result[
            "category"
        ]

        image_is_strong = (
            image_result["confidence"] >= 0.70
        )

        categories_conflict = (
            image_is_strong
            and text_category is not None
            and image_category not in {
                None,
                "unknown",
                "travel_document",
            }
            and image_category != text_category
        )

        confirmation_question = (
            lower_text(text).startswith("this is ")
            or (
                lower_text(text).startswith("is this ")
                and not lower_text(text).startswith("is this the way")
            )
        )

        if categories_conflict:

            selected_record = None

            # A specific textual destination/service may
            # still be retained, but the contradiction is
            # surfaced to the passenger.
            if text_result and not confirmation_question:

                selected_record = (
                    text_result["record"]["id"]
                    if text_result.get("record")
                    else None
                )

            result.update(
                {
                    "record": selected_record,
                    "behavior": "conflict_warning",
                    "confidence": "caution",
                    "conflict": True,
                    "reason": (
                        f"Visual evidence suggests "
                        f"{image_category}, while the "
                        f"language request suggests "
                        f"{text_category}."
                    ),
                }
            )

            return result

        # =================================================
        # 15. Weak image but specific language
        # =================================================


        if (
            image_label
            and image_result["confidence"] < 0.50
            and text_result
        ):

            selected_record = (
                text_result["record"]["id"]
                if text_result.get("record")
                else None
            )

            result.update(
                {
                    "record": selected_record,
                    "behavior": "weight_text_more",
                    "confidence": text_result[
                        "confidence"
                    ],
                    "reason": (
                        "Visual evidence is weak, so stronger "
                        "language evidence receives greater "
                        "fusion weight."
                    ),
                }
            )

            return result

        # =================================================
        # 16. Agreement between modalities
        # =================================================

        if (
            image_result["record"]
            and text_result
        ):

            text_record = (
                text_result["record"]["id"]
                if text_result.get("record")
                else None
            )

            if (
                text_record
                == image_result["record"]
            ):

                result.update(
                    {
                        "record": text_record,
                        "behavior": "retrieve",
                        "confidence": "confident",
                        "reason": (
                            "Image and language evidence agree."
                        ),
                    }
                )

                return result

        # =================================================
        # 17. Text/voice retrieval fallback
        # =================================================

        if text_result:

            selected_record = (
                text_result["record"]["id"]
                if text_result.get("record")
                else None
            )

            result.update(
                {
                    "record": selected_record,
                    "behavior": "retrieve",
                    "confidence": text_result[
                        "confidence"
                    ],
                    "reason": (
                        "Response primarily supported by "
                        "language evidence."
                    ),
                }
            )

            return result

        # =================================================
        # 18. Image-only fallback
        # =================================================

        if (
            image_result["record"]
            and image_result["confidence"] >= 0.70
        ):

            result.update(
                {
                    "record": image_result["record"],
                    "behavior": "retrieve",
                    "confidence": "caution",
                    "reason": (
                        "Only visual evidence is available."
                    ),
                }
            )

            return result

        # =================================================
        # 19. Final abstention
        # =================================================

        result.update(
            {
                "record": None,
                "behavior": "clarify",
                "confidence": "uncertain",
                "reason": (
                    "Available multimodal evidence is "
                    "insufficient for safe retrieval."
                ),
            }
        )

        return result