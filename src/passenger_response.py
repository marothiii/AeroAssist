from pathlib import Path
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "airport_knowledge_base.json"


def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


KNOWLEDGE_BASE = load_knowledge_base()
KB_BY_ID = {record["id"]: record for record in KNOWLEDGE_BASE}


def get_record(record_id):
    if not record_id:
        return None

    return KB_BY_ID.get(record_id)


def build_standard_record_message(record):
    """
    Build a useful passenger-facing response from a safe KB record.
    """

    name = record.get("name", "Airport service")
    terminal = record.get("terminal")
    zone = record.get("zone")
    floor = record.get("floor")
    description = record.get("description")
    directions = record.get("directions")
    walking_minutes = record.get("walking_minutes")
    opening_hours = record.get("opening_hours")
    verification_message = record.get("verification_message")

    parts = [f"**{name}**"]

    location_parts = [
        value
        for value in [terminal, zone, floor]
        if value
    ]

    if location_parts:
        parts.append("Location: " + " • ".join(location_parts))

    if description:
        parts.append(description)

    if directions:
        parts.append(f"Directions: {directions}")

    if walking_minutes is not None:
        parts.append(
            f"Estimated walking time: approximately {walking_minutes} minutes."
        )

    if opening_hours:
        parts.append(f"Availability: {opening_hours}.")

    if verification_message:
        parts.append(f"Please verify: {verification_message}")

    return "\n\n".join(parts)


def build_accessible_record_message(record):
    """
    Build an accessibility-focused passenger response.
    """

    name = record.get("name", "Airport service")
    terminal = record.get("terminal")
    zone = record.get("zone")
    accessible_route = record.get("accessible_route")
    accessible_minutes = record.get("accessible_minutes")
    nearest_lift = record.get("nearest_lift")
    verification_message = record.get("verification_message")

    parts = [f"**{name} — step-free route**"]

    location_parts = [
        value
        for value in [terminal, zone]
        if value
    ]

    if location_parts:
        parts.append("Location: " + " • ".join(location_parts))

    if accessible_route:
        parts.append(f"Accessible directions: {accessible_route}")

    if nearest_lift:
        parts.append(f"Nearest lift: {nearest_lift}.")

    if accessible_minutes is not None:
        parts.append(
            f"Estimated accessible walking time: approximately "
            f"{accessible_minutes} minutes."
        )

    if verification_message:
        parts.append(f"Please verify: {verification_message}")

    return "\n\n".join(parts)


def build_family_record_message(record):
    """
    Build a family-focused passenger response.
    """

    base_message = build_standard_record_message(record)
    family_facilities = record.get("family_facilities")

    if family_facilities:
        base_message += (
            f"\n\nNearby family facility: {family_facilities}."
        )

    return base_message


def build_passenger_response(fusion_result):
    """
    Convert the internal fusion result into a safe passenger-facing response.

    Important:
    - uncertain internal retrievals are never exposed;
    - live-data questions are redirected to official sources;
    - multimodal conflicts are clearly disclosed;
    - safe KB matches provide useful directions and next-step information.
    """

    if not fusion_result:
        return {
            "message": (
                "I couldn't reliably understand that request. "
                "Please try again or ask airport staff."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    confidence = fusion_result.get("confidence", "uncertain")
    behavior = fusion_result.get("behavior", "")
    record_id = fusion_result.get("record")
    record = get_record(record_id)

    # ---------------------------------------------------------
    # 1. UNCERTAINTY SAFETY GATE
    # ---------------------------------------------------------

    if confidence == "uncertain":
        return {
            "message": (
                "I'm not confident enough to give you directions from that "
                "input. Please repeat your request, type it instead, provide "
                "a clearer image, or ask airport staff."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 2. LIVE / DYNAMIC INFORMATION
    # ---------------------------------------------------------

    if behavior == "official_source_redirect":
        return {
            "message": (
                "That information may change in real time. Please check the "
                "official airport departure displays, airline information, "
                "or an airport information desk before acting on it."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 3. EMERGENCY / HUMAN HANDOVER
    # ---------------------------------------------------------

    if behavior == "emergency_handover":
        return {
            "message": (
                "This requires immediate human assistance. Please contact "
                "airport staff or emergency services straight away."
            ),
            "safe_to_display_record": True,
            "record": record,
        }

    if behavior == "human_handover":
        if record:
            name = record.get(
                "name",
                "the appropriate airport assistance desk",
            )

            message = (
                f"Please go directly to **{name}** for help."
            )

            verification_message = record.get("verification_message")

            if verification_message:
                message += f"\n\n{verification_message}"

            return {
                "message": message,
                "safe_to_display_record": True,
                "record": record,
            }

        return {
            "message": (
                "This request requires assistance from airport staff. "
                "Please go to the nearest information or passenger-service desk."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 4. PRIVACY
    # ---------------------------------------------------------

    if behavior == "privacy_warning_and_minimise":
        return {
            "message": (
                "Please avoid sharing personal travel documents or sensitive "
                "information unnecessarily. Use the official airport displays "
                "or ask airport staff to confirm your travel details."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 5. SESSION CONTEXT / CLARIFICATION
    # ---------------------------------------------------------

    if behavior == "use_session_context":
        return {
            "message": (
                "I need the previous journey step before I can safely tell you "
                "what comes next."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    if behavior in {"clarify", "request_clearer_input"}:
        return {
            "message": (
                "I need a little more information before I can guide you. "
                "Please provide the gate, terminal, service, destination, "
                "or a clearer image."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 6. MULTIMODAL CONFLICT
    # ---------------------------------------------------------

    if behavior == "conflict_warning":
        if record:
            useful_details = build_standard_record_message(record)

            return {
                "message": (
                    "Your image and request appear to disagree. I can use the "
                    "destination you explicitly asked for, but please verify "
                    "the surrounding signs before following the route."
                    f"\n\n{useful_details}"
                ),
                "safe_to_display_record": True,
                "record": record,
            }

        return {
            "message": (
                "Your image and request appear to disagree, so I can't safely "
                "choose a destination. Please check the signs or clarify your "
                "request."
            ),
            "safe_to_display_record": False,
            "record": None,
        }

    # ---------------------------------------------------------
    # 7. SAFE KB RECORD RESPONSES
    # ---------------------------------------------------------

    if record:
        if behavior in {
            "accessible_route",
            "urgent_accessible_route",
            "assistance_route",
        }:
            message = build_accessible_record_message(record)

        elif behavior in {
            "family_recommendation",
            "nearby_family_recommendation",
        }:
            message = build_family_record_message(record)

        else:
            message = build_standard_record_message(record)

        if confidence == "caution":
            message = (
                "I found a possible match. Please verify the nearby signs "
                "before relying on this guidance.\n\n"
                + message
            )

        return {
            "message": message,
            "safe_to_display_record": True,
            "record": record,
        }

    # ---------------------------------------------------------
    # 8. FINAL SAFE FALLBACK
    # ---------------------------------------------------------

    return {
        "message": (
            "I couldn't safely identify a matching airport location or service. "
            "Please provide more detail or ask airport staff."
        ),
        "safe_to_display_record": False,
        "record": None,
    }