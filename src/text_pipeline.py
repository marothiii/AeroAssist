import json
import os
import re
import joblib

from sentence_transformers import SentenceTransformer, util


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "data", "airport_knowledge_base.json")
INTENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "intent_classifier.joblib"
)

SEMANTIC_INTENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "semantic_intent_classifier.joblib"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_record_text(record):
    keywords = " ".join(record.get("keywords", []))

    return (
        f"{record['name']}. "
        f"Category: {record['category']}. "
        f"Terminal: {record['terminal']}. "
        f"Zone: {record['zone']}. "
        f"Description: {record['description']}. "
        f"Keywords: {keywords}."
    )


def extract_gate(text):
    match = re.search(r"\b([AB]\d{1,2})\b", text.upper())

    if match:
        return match.group(1)

    return None


def extract_terminal(text):
    match = re.search(
        r"\b(?:TERMINAL\s*|T)([12])\b",
        text.upper()
    )

    if match:
        return f"Terminal {match.group(1)}"

    return None


def detect_context(text):
    text_lower = text.lower()

    urgency = "normal"
    passenger_mode = "standard"

    if any(
        phrase in text_lower
        for phrase in [
            "boarding closes",
            "boarding soon",
            "boards soon",
            "flight boards",
            "late",
            "in 10 minutes",
            "in 15 minutes",
            "short connection"
        ]
    ):
        urgency = "time_sensitive"

    if any(
        phrase in text_lower
        for phrase in [
            "can't use stairs",
            "cannot use stairs",
            "wheelchair",
            "limited mobility",
            "step-free",
            "accessible"
        ]
    ):
        passenger_mode = "accessibility"

    if any(
        phrase in text_lower
        for phrase in [
            "baby",
            "child",
            "family",
            "diaper",
            "feeding"
        ]
    ):
        passenger_mode = "family"

    if any(
        phrase in text_lower
        for phrase in [
            "lost my passport",
            "passport is missing",
            "someone has collapsed",
            "emergency"
        ]
    ):
        urgency = "critical"

    return urgency, passenger_mode

        


def detect_intent(text):
    text_lower = text.lower()

    if any(word in text_lower for word in ["gate", "boarding gate"]):
        return "find_gate"

    if any(word in text_lower for word in ["check-in", "check in", "checkin"]):
        return "find_checkin"

    if any(word in text_lower for word in ["baggage", "luggage", "carousel"]):
        return "find_baggage"

    if any(word in text_lower for word in ["security", "security control"]):
        return "find_security"

    if any(
        word in text_lower
        for word in [
            "lounge",
            "relax",
            "relaxing",
            "rest area"
        ]
    ):
        return "find_lounge"

    if any(word in text_lower for word in ["restaurant", "food", "eat"]):
        return "find_restaurant"

    if any(
        word in text_lower
        for word in [
            "train",
            "station",
            "taxi",
            "cab",
            "uber",
            "rideshare",
            "transport"
        ]
    ):
        return "find_transport"

    if any(word in text_lower for word in ["information desk", "information"]):
        return "find_information"

    if any(word in text_lower for word in ["lost my passport", "passport is missing", "passport"]):
        return "travel_document_help"

    if any(word in text_lower for word in ["lost", "missing bag", "lost property"]):
        return "lost_property"

    if any(word in text_lower for word in ["wheelchair", "accessible", "stairs", "mobility"]):
        return "accessibility_help"

    if any(word in text_lower for word in ["baby", "child", "family", "diaper", "feeding"]):
        return "family_assistance"

    if any(word in text_lower for word in ["transfer", "connection", "connecting flight"]):
        return "transfer_help"

    if any(word in text_lower for word in ["open", "opening", "closing", "hours"]):
        return "opening_hours"

    return "unknown"


    

def intent_matches_record(intent, record):
    intent_record_map = {
        "find_gate": {
            "NIA001", "NIA002", "NIA003",
            "NIA004", "NIA005", "NIA006"
        },
        "find_checkin": {"NIA007", "NIA008"},
        "find_security": {"NIA009", "NIA010"},
        "find_baggage": {"NIA011", "NIA012"},
        "find_lounge": {"NIA013"},
        "find_information": {"NIA014", "NIA015"},
        "lost_property": {"NIA016"},
        "accessibility_help": {"NIA017"},
        "family_assistance": {"NIA019"},
        "find_restaurant": {"NIA020"},
        "find_transport": {"NIA021", "NIA022"},
        "travel_document_help": {"NIA024"},
        "transfer_help": {"NIA025"}
    }

    # Opening-hours questions can refer to many facilities
    if intent == "opening_hours":
        return bool(record.get("opening_hours"))

    allowed_records = intent_record_map.get(intent, set())

    return record["id"] in allowed_records


def hybrid_intent_decision(query, rule_intent, ml_intent):
    text_lower = query.lower()
    gate = extract_gate(query)

    # 1. High-priority question-purpose signals
    if any(
        phrase in text_lower
        for phrase in [
            "what time",
            "opening time",
            "closing time",
            "when does",
            "when is",
            "hours"
        ]
    ):
        return "opening_hours"

    # 2. Explicit airport-service signals
    if any(
        phrase in text_lower
        for phrase in [
            "security checkpoint",
            "security control"
        ]
    ):
        return "find_security"

    # 3. Safety / assistance signals
    if rule_intent == "travel_document_help":
        return "travel_document_help"

    if rule_intent in [
        "accessibility_help",
        "family_assistance"
    ]:
        return rule_intent

    # 4. Exact gate entity evidence
    if gate:
        return "find_gate"

    # 5. Agreement between rule and ML
    if (
        rule_intent != "unknown"
        and rule_intent == ml_intent
    ):
        return rule_intent

    # 6. Rule fallback
    if rule_intent != "unknown":
        return rule_intent

    # 7. ML fallback
    return ml_intent




class AeroAssistTextPipeline:

    def __init__(self):
        print("Loading AeroAssist knowledge base...")
        self.knowledge_base = load_knowledge_base()

        print("Loading sentence-transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        self.record_texts = [
            build_record_text(record)
            for record in self.knowledge_base
        ]

        print("Creating knowledge-base embeddings...")
        self.record_embeddings = self.model.encode(
            self.record_texts,
            convert_to_tensor=True
        )
        
        print("Loading trained intent classifier...")

        intent_bundle = joblib.load(INTENT_MODEL_PATH)

        self.intent_vectorizer = intent_bundle["vectorizer"]
        self.intent_classifier = intent_bundle["classifier"]

        print("Loading semantic intent classifier...")

        semantic_bundle = joblib.load(
            SEMANTIC_INTENT_MODEL_PATH
        )

        self.semantic_intent_classifier = (
            semantic_bundle["classifier"]
        )


    def retrieve(self, query):
        gate = extract_gate(query)
        terminal = extract_terminal(query)

        rule_intent = detect_intent(query)

        query_features = self.intent_vectorizer.transform([query])

        tfidf_intent = self.intent_classifier.predict(
            query_features
        )[0]

        semantic_query_embedding = self.model.encode(
            [query]
        )

        semantic_intent = self.semantic_intent_classifier.predict(
            semantic_query_embedding
        )[0]

        semantic_probabilities = (
            self.semantic_intent_classifier.predict_proba(
                semantic_query_embedding
            )[0]
        )

        semantic_confidence = float(
            max(semantic_probabilities)
        )

        intent = hybrid_intent_decision(
            query,
            rule_intent,
            semantic_intent
        )

        urgency, passenger_mode = detect_context(query)

        assistance_intents = {
            "accessibility_help",
            "family_assistance"
        }

        complementary_context = (
            passenger_mode in {"accessibility", "family"}
            and (
                rule_intent in assistance_intents
                or semantic_intent in assistance_intents
            )
        )

        model_disagreement = (
            rule_intent != "unknown"
            and semantic_intent != rule_intent
            and not complementary_context
        )


        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            query_embedding,
            self.record_embeddings
        )[0]

        # -------------------------------------------------
        # Intent-aware knowledge-base retrieval
        # -------------------------------------------------

        candidate_indices = [
            index
            for index, record in enumerate(self.knowledge_base)
            if intent_matches_record(intent, record)
        ]

        # Search within records that match the detected intent
        if candidate_indices:
            best_index = max(
                candidate_indices,
                key=lambda index: float(scores[index])
            )

        # If the intent has no mapped records, use global retrieval
        else:
            best_index = int(scores.argmax())

        best_record = self.knowledge_base[best_index]
        semantic_score = float(scores[best_index])



        # Track whether an exact gate entity was matched
        entity_match = False

        # Strong deterministic gate override
        if gate:
            for index, record in enumerate(self.knowledge_base):
                if record["name"].lower() == f"gate {gate}".lower():
                    best_record = record
                    best_index = index
                    entity_match = True

                    # Use the real semantic similarity for the selected gate record
                    semantic_score = float(scores[index])
                    break

        # Check whether the retrieved record matches the final intent
        intent_match = intent_matches_record(intent, best_record)

        # Agreement between independent intent methods
        rule_semantic_agreement = (
            rule_intent != "unknown"
            and rule_intent == semantic_intent
        )

        # Validation-tuned confidence policy
        if model_disagreement:
            confidence = "uncertain"
            confidence_reason = "intent models disagree"

        elif entity_match:
            confidence = "confident"
            confidence_reason = "exact entity match"

        elif rule_semantic_agreement:
            if semantic_score >= 0.30:
                confidence = "confident"
                confidence_reason = (
                    "intent models agree with sufficient retrieval similarity"
                )
            else:
                confidence = "caution"
                confidence_reason = (
                    "intent models agree but retrieval similarity is limited"
                )

        elif (
            semantic_confidence >= 0.25
            and semantic_score >= 0.35
        ):
            confidence = "caution"
            confidence_reason = (
                "semantic intent and retrieval provide moderate evidence"
            )

        else:
            confidence = "uncertain"
            confidence_reason = "insufficient supporting evidence"


        

        return {
            "query": query,
            "gate": gate,
            "terminal": terminal,
            "rule_intent": rule_intent,
            "tfidf_intent": tfidf_intent,
            "semantic_intent": semantic_intent,
            "semantic_confidence": round(semantic_confidence, 3),
            "model_disagreement": model_disagreement,
            "intent": intent,
            "urgency": urgency,
            "passenger_mode": passenger_mode,
            "record": best_record,
            "semantic_similarity": round(semantic_score, 3),
            "entity_match": entity_match,
            "intent_match": intent_match,
            "rule_semantic_agreement": rule_semantic_agreement,
            "confidence": confidence,
            "confidence_reason": confidence_reason
        }


    def generate_response(self, result):
        record = result["record"]
        passenger_mode = result["passenger_mode"]



        if result["urgency"] == "critical":

            return (
                f"This requires human assistance. "
                f"Please go directly to {record['name']}. "
                f"{record['verification_message']}"
            )
            


        
        if result["confidence"] == "uncertain":
            return (
                "I'm not confident enough to give a reliable direction. "
                "Please provide a more specific location, gate, or service."
            )

        if passenger_mode == "accessibility":
            route = record.get(
                "accessible_route",
                record["directions"]
            )

            minutes = record.get(
                "accessible_minutes",
                record["walking_minutes"]
            )

            return (
                f"{record['name']}: {route} "
                f"Estimated accessible walking time: "
                f"{minutes} minutes."
            )

        if passenger_mode == "family":
            family_info = record.get(
                "family_facilities",
                "No specific family facility information available."
            )

            return (
                f"{record['name']}: {record['directions']} "
                f"Estimated walking time: "
                f"{record['walking_minutes']} minutes. "
                f"Family information: {family_info}"
            )

        return (
            f"{record['name']}: {record['directions']} "
            f"Estimated walking time: "
            f"{record['walking_minutes']} minutes."
        )


if __name__ == "__main__":

    pipeline = AeroAssistTextPipeline()

    print("Type a passenger query.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Passenger: ").strip()

        if query.lower() == "exit":
            print("AeroAssist closed.")
            break

        if not query:
            continue

        result = pipeline.retrieve(query)

        print("\n--- AeroAssist Analysis ---")
        print("Gate:", result["gate"])
        print("Terminal:", result["terminal"])
        print("Rule intent:", result["rule_intent"])
        print("TF-IDF intent:", result["tfidf_intent"])
        print("Semantic intent:", result["semantic_intent"])
        print(
            "Semantic intent confidence:",
            result["semantic_confidence"]
        )
        print(
            "Model disagreement:",
            result["model_disagreement"]
        )
        print("Final intent:", result["intent"])
        print("Urgency:", result["urgency"])
        print("Passenger mode:", result["passenger_mode"])
        print("Retrieved record:", result["record"]["id"])
        print("Location:", result["record"]["name"])
        print("Semantic similarity:", result["semantic_similarity"])
        print("Entity match:", result["entity_match"])
        print("Intent match:", result["intent_match"])
        print("Confidence:", result["confidence"])
        print("Confidence reason:", result["confidence_reason"])

        print("\nAeroAssist:")
        print(pipeline.generate_response(result))

        print("\n" + "-" * 50 + "\n")