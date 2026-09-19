from pathlib import Path
import tempfile
import html
import re

import streamlit as st

from src.end_to_end_pipeline import AeroAssistEndToEndPipeline


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AeroAssist AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# FROZEN BACKEND
# ============================================================

@st.cache_resource
def load_pipeline():
    return AeroAssistEndToEndPipeline(
        use_vision=True,
        use_voice=True,
    )


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    return Path(temp_file.name)


# ============================================================
# HELPERS
# ============================================================

def esc(value):
    return html.escape(
        "" if value is None else str(value)
    )


def pretty(value):
    if not value:
        return "—"

    return str(value).replace(
        "_",
        " ",
    ).title()


def confidence_label(value):
    if value == "confident":
        return "HIGH"

    if value == "caution":
        return "CAUTION"

    return "UNCERTAIN"


def get_record(result):
    record = (
        result
        .get("passenger_response", {})
        .get("record")
    )

    if isinstance(record, dict):
        return record

    return None


def active_language(result):
    transcription = result.get("transcription")

    if transcription:
        return transcription.get(
            "transcript",
            "",
        )

    return result.get(
        "input_text",
        "",
    )


def ambiguous_gate_query(text):
    """
    Detect requests such as 'Gate 12' where the
    airport concourse letter is missing.

    Explicit Gate A12 / Gate B12 remains valid.
    """

    if not text:
        return None

    text = text.strip()

    explicit_gate = re.search(
        r"\b(?:gate\s*)?([AB])\s?(\d{1,2})\b",
        text,
        flags=re.IGNORECASE,
    )

    if explicit_gate:
        return None

    ambiguous = re.search(
        r"\bgate\s+(\d{1,2})\b",
        text,
        flags=re.IGNORECASE,
    )

    if ambiguous:
        return ambiguous.group(1)

    return None


# ============================================================
# FUSION CORE
# ============================================================

def build_core(result=None):

    if result is None:

        language_title = "Waiting for request"
        language_sub = "LANGUAGE READY"
        language_class = "idle"

        vision_title = "No visual evidence"
        vision_sub = "CLIP READY"
        vision_class = "idle"

        voice_title = "No speech evidence"
        voice_sub = "WHISPER READY"
        voice_class = "idle"

        state = "STANDBY"
        confidence = "—"
        safety_gate = "WAITING"
        behavior = "AWAITING EVIDENCE"

        core_class = "standby"

    else:

        fusion = result["fusion_result"]
        response = result["passenger_response"]

        # ------------------------------------------------
        # LANGUAGE
        # ------------------------------------------------

        text = active_language(result)

        if text:

            if len(text) > 46:
                language_title = text[:43] + "..."
            else:
                language_title = text

            if result.get("transcription"):
                language_sub = "VOICE TRANSCRIPT"
            else:
                language_sub = "TEXT ACTIVE"

            language_class = "active"

        else:

            language_title = "No language"
            language_sub = "LANGUAGE IDLE"
            language_class = "idle"

        # ------------------------------------------------
        # VISION
        # ------------------------------------------------

        vision_result = result.get(
            "vision_result"
        )

        if vision_result:

            predicted = pretty(
                vision_result.get(
                    "predicted_class"
                )
            )

            score = vision_result.get(
                "score",
                0,
            )

            margin = vision_result.get(
                "margin"
            )

            vision_title = predicted

            if result.get(
                "fusion_image_label"
            ):

                if margin is None:
                    vision_sub = (
                        f"SCORE {score:.3f}"
                    )

                else:
                    vision_sub = (
                        f"SCORE {score:.3f} "
                        f"· Δ {margin:.3f}"
                    )

                vision_class = "active"

            else:

                vision_sub = (
                    f"WITHHELD · SCORE {score:.3f}"
                )

                vision_class = "withheld"

        else:

            vision_title = "No visual evidence"
            vision_sub = "VISION IDLE"
            vision_class = "idle"

        # ------------------------------------------------
        # VOICE
        # ------------------------------------------------

        transcription = result.get(
            "transcription"
        )

        if transcription:

            voice_title = "Transcript generated"
            voice_sub = "WHISPER ACTIVE"
            voice_class = "active"

        else:

            voice_title = "No speech evidence"
            voice_sub = "WHISPER IDLE"
            voice_class = "idle"

        # ------------------------------------------------
        # FUSION STATE
        # ------------------------------------------------

        safe = bool(
            response.get(
                "safe_to_display_record"
            )
        )

        conflict = bool(
            fusion.get("conflict")
        )

        confidence = confidence_label(
            fusion.get(
                "confidence",
                "uncertain",
            )
        )

        modality_count = len(
            result.get(
                "modalities",
                [],
            )
        )

        if conflict:
            state = "CONFLICT"

        elif modality_count <= 1:
            state = "SINGLE SOURCE"

        elif safe:
            state = "ALIGNED"

        else:
            state = "UNRESOLVED"

        safety_gate = (
            "PASS"
            if safe
            else "BLOCK"
        )

        behavior = pretty(
            fusion.get("behavior")
        ).upper()

        if not safe:
            core_class = "danger"

        elif (
            conflict
            or fusion.get("confidence")
            == "caution"
        ):
            core_class = "warning"

        else:
            core_class = "success"

    return (
        '<div class="fusion-stage">'

        '<div class="gridfx"></div>'

        f'<div class="sensor sensor-text {esc(language_class)}">'
        '<div class="sensor-id">01 / LANGUAGE</div>'
        f'<div class="sensor-title">{esc(language_title)}</div>'
        f'<div class="sensor-sub">{esc(language_sub)}</div>'
        '</div>'

        f'<div class="sensor sensor-vision {esc(vision_class)}">'
        '<div class="sensor-id">02 / VISION</div>'
        f'<div class="sensor-title">{esc(vision_title)}</div>'
        f'<div class="sensor-sub">{esc(vision_sub)}</div>'
        '</div>'

        f'<div class="sensor sensor-voice {esc(voice_class)}">'
        '<div class="sensor-id">03 / VOICE</div>'
        f'<div class="sensor-title">{esc(voice_title)}</div>'
        f'<div class="sensor-sub">{esc(voice_sub)}</div>'
        '</div>'

        '<div class="wire wire-a"></div>'
        '<div class="wire wire-b"></div>'
        '<div class="wire wire-c"></div>'

        f'<div class="fusion-core {esc(core_class)}">'

        '<div class="orbit orbit1"></div>'
        '<div class="orbit orbit2"></div>'

        '<div class="core-inner">'
        '<b>A</b>'
        '<span>FUSION</span>'
        '</div>'

        '</div>'

        '<div class="readout">'

        '<div>'
        '<span>STATE</span>'
        f'<b>{esc(state)}</b>'
        '</div>'

        '<div>'
        '<span>CONFIDENCE</span>'
        f'<b>{esc(confidence)}</b>'
        '</div>'

        '<div>'
        '<span>SAFETY GATE</span>'
        f'<b>{esc(safety_gate)}</b>'
        '</div>'

        '<div>'
        '<span>BEHAVIOUR</span>'
        f'<b>{esc(behavior)}</b>'
        '</div>'

        '</div>'

        '</div>'
    )


# ============================================================
# ROUTE BOARD
# ============================================================

def build_route(
    record,
    behavior,
):

    if not record:
        return ""

    accessible_behaviors = {
        "accessible_route",
        "urgent_accessible_route",
        "assistance_route",
    }

    if (
        behavior in accessible_behaviors
        and record.get("accessible_route")
    ):

        directions = record.get(
            "accessible_route",
            "",
        )

        minutes = record.get(
            "accessible_minutes"
        )

        mode = "STEP-FREE ROUTE"

    else:

        directions = record.get(
            "directions",
            "",
        )

        minutes = record.get(
            "walking_minutes"
        )

        mode = "PASSENGER ROUTE"

    metadata = []

    for value in [
        record.get("terminal"),
        record.get("zone"),
        record.get("floor"),
    ]:

        if value:
            metadata.append(
                str(value)
            )

    meta = " · ".join(
        metadata
    )

    if minutes is None:
        walk = "—"
    else:
        walk = f"{minutes} MIN"

    name = record.get(
        "name",
        "Airport service",
    )

    return (
        '<div class="route-board">'

        '<div class="route-top">'
        f'<span>{esc(mode)}</span>'
        '<span>NIA WAYFINDING</span>'
        '</div>'

        '<div class="route-main">'

        '<div>'
        f'<div class="route-name">{esc(name)}</div>'
        f'<div class="route-meta">{esc(meta)}</div>'
        '</div>'

        '<div class="walk">'
        '<span>EST. WALK</span>'
        f'<b>{esc(walk)}</b>'
        '</div>'

        '</div>'

        '<div class="path">'
        '<span class="dot"></span>'
        '<span class="line"></span>'
        '<span class="arrow">➜</span>'
        '<span class="dot"></span>'
        '</div>'

        f'<div class="directions">{esc(directions)}</div>'

        '</div>'
    )


# ============================================================
# CONFLICT DISPLAY
# ============================================================

def build_conflict(result):

    vision = result.get(
        "vision_result"
    )

    if vision:

        visual_evidence = pretty(
            vision.get(
                "predicted_class"
            )
        )

    else:

        visual_evidence = (
            "Visual evidence"
        )

    language = active_language(
        result
    )

    if len(language) > 58:
        language = (
            language[:55] + "..."
        )

    return (
        '<div class="conflict-banner">'

        '<div class="conflict-evidence">'
        '<span>VISION</span>'
        f'<b>{esc(visual_evidence)}</b>'
        '</div>'

        '<div class="conflict-middle">'
        '<div class="warning-icon">!</div>'
        '<b>EVIDENCE CONFLICT</b>'
        '<span>Modalities disagree</span>'
        '</div>'

        '<div class="conflict-evidence right">'
        '<span>LANGUAGE</span>'
        f'<b>{esc(language)}</b>'
        '</div>'

        '</div>'
    )


# ============================================================
# BLOCKED DISPLAY
# ============================================================

def build_blocked():

    return (
        '<div class="blocked-panel">'

        '<div class="blocked-symbol">×</div>'

        '<div>'

        '<div class="blocked-kicker">'
        'SAFETY INTERVENTION'
        '</div>'

        '<div class="blocked-title">'
        'Route guidance withheld'
        '</div>'

        '<div class="blocked-copy">'
        'The available evidence did not meet the reliability '
        'requirements for passenger guidance.'
        '</div>'

        '</div>'

        '<div class="blocked-state">'
        'BLOCK'
        '</div>'

        '</div>'
    )


# ============================================================
# VISION ANALYSIS
# ============================================================

def build_vision_analysis(result):

    vision = result.get(
        "vision_result"
    )

    if not vision:
        return ""

    predicted = pretty(
        vision.get(
            "predicted_class"
        )
    )

    score = vision.get(
        "score",
        0,
    )

    margin = vision.get(
        "margin"
    )

    accepted = bool(
        result.get(
            "fusion_image_label"
        )
    )

    if accepted:

        status = (
            "ACCEPTED INTO FUSION"
        )

        status_class = (
            "vision-accepted"
        )

        explanation = (
            "Visual evidence met the operational reliability "
            "requirements and was passed to multimodal fusion."
        )

    else:

        status = (
            "WITHHELD FROM FUSION"
        )

        status_class = (
            "vision-withheld"
        )

        explanation = (
            "CLIP produced a scene classification, but the visual "
            "evidence was not accepted as actionable fusion evidence."
        )

    margin_text = (
        f"{margin:.3f}"
        if margin is not None
        else "—"
    )

    return (
        f'<div class="vision-analysis {status_class}">'

        '<div class="vision-analysis-head">'
        '<span>VISION ANALYSIS / CLIP</span>'
        f'<b>{esc(status)}</b>'
        '</div>'

        '<div class="vision-analysis-grid">'

        '<div>'
        '<span>PREDICTED SCENE</span>'
        f'<strong>{esc(predicted)}</strong>'
        '</div>'

        '<div>'
        '<span>CLIP SCORE</span>'
        f'<strong>{score:.3f}</strong>'
        '</div>'

        '<div>'
        '<span>MARGIN</span>'
        f'<strong>{esc(margin_text)}</strong>'
        '</div>'

        '</div>'

        f'<p>{esc(explanation)}</p>'

        '</div>'
    )


# ============================================================
# VERIFICATION
# ============================================================

def build_verification(record):

    if not record:
        return ""

    message = record.get(
        "verification_message"
    )

    if not message:
        return ""

    return (
        '<div class="verification-strip">'
        '<span>VERIFY</span>'
        f'<p>{esc(message)}</p>'
        '</div>'
    )


# ============================================================
# PROCESSING DISPLAY
# ============================================================

def build_loading():

    return (
        '<div class="processing-panel">'

        '<div class="processing-top">'
        '<span>AEROASSIST PROCESSING</span>'
        '<b>LIVE</b>'
        '</div>'

        '<div class="processing-title">'
        'Evaluating passenger evidence'
        '</div>'

        '<div class="processing-flow">'
        '<span>LANGUAGE</span>'
        '<i>→</i>'
        '<span>VISION</span>'
        '<i>→</i>'
        '<span>VOICE</span>'
        '<i>→</i>'
        '<strong>FUSION</strong>'
        '<i>→</i>'
        '<span>SAFETY</span>'
        '</div>'

        '<div class="scanner">'
        '<div class="scanner-line"></div>'
        '</div>'

        '</div>'
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #02060c;
    --panel: #06101b;
    --cyan: #6edcff;
    --cyan2: #35bce9;
    --green: #6cf0b1;
    --amber: #ffc85a;
    --red: #ff7186;
    --white: #f6f8fb;
    --muted: #8296a9;
}


/* ==========================================================
   STREAMLIT HEADER
   ========================================================== */

header[data-testid="stHeader"] {
    display: flex;
    background: rgba(2,6,12,.94);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,.05);
    height: 3.5rem;
}

[data-testid="stToolbar"] {
    display: flex !important;
}

.stDeployButton {
    display: block !important;
}








#MainMenu {
    visibility: visible;
}

footer {
    visibility: hidden;
}


/* ==========================================================
   APP
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 50% -5%,
            rgba(40,140,190,.10),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #030812,
            #02060c
        );
}

.block-container {
    max-width: 1360px;
    padding-top: 4.5rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   NAV
   ========================================================== */

.nia-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 5px 2px 14px;

    border-bottom:
        1px solid
        rgba(255,255,255,.055);

    margin-bottom: 13px;
}

.nia-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nia-mark {
    width: 32px;
    height: 32px;

    display: grid;
    place-items: center;

    border-radius: 8px;

    background:
        linear-gradient(
            135deg,
            #7ee4ff,
            #42bee9
        );

    color: #02111c;
    font-weight: 950;

    box-shadow:
        0 0 24px
        rgba(110,220,255,.18);
}

.nia-brand {
    color: white;
    font-size: .96rem;
    font-weight: 850;
}

.nia-sub {
    color: #6f8396;

    font:
        .61rem
        ui-monospace,
        monospace;

    letter-spacing: .08em;
}

.nia-right {
    display: flex;
    gap: 7px;
    align-items: center;
}

.chip {
    border:
        1px solid
        rgba(255,255,255,.065);

    background:
        rgba(255,255,255,.023);

    color: #758a9d;

    border-radius: 7px;

    padding: 5px 8px;

    font:
        .59rem
        ui-monospace,
        monospace;
}

.live {
    color: var(--green);
}


/* ==========================================================
   TITLE
   ========================================================== */

.prompt {
    text-align: center;
    margin: 5px 0 11px;
}

.prompt-kicker {
    color: #5fcdf6;

    font:
        .65rem
        ui-monospace,
        monospace;

    font-weight: 850;
    letter-spacing: .14em;
}

.prompt-title {
    color: white;

    font-size:
        clamp(
            1.7rem,
            2.8vw,
            2.35rem
        );

    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -.045em;

    margin-top: 5px;
}

.prompt-sub {
    color: #8195a8;
    font-size: .82rem;
    margin-top: 6px;
}


/* ==========================================================
   FUSION STAGE
   ========================================================== */

.fusion-stage {
    position: relative;
    min-height: 365px;

    border:
        1px solid
        rgba(110,220,255,.14);

    border-radius: 21px;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 50% 48%,
            rgba(39,156,205,.095),
            transparent 18%
        ),
        linear-gradient(
            180deg,
            rgba(6,17,29,.96),
            rgba(3,10,18,.99)
        );

    box-shadow:
        0 25px 65px
        rgba(0,0,0,.22);

    margin-bottom: 13px;
}

.gridfx {
    position: absolute;
    inset: 0;

    background-image:
        linear-gradient(
            rgba(110,220,255,.025)
            1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(110,220,255,.025)
            1px,
            transparent 1px
        );

    background-size:
        33px 33px;
}


/* ==========================================================
   SENSOR CARDS
   ========================================================== */

.sensor {
    position: absolute;

    width: 220px;
    min-height: 86px;

    border:
        1px solid
        rgba(255,255,255,.075);

    background:
        rgba(7,18,31,.95);

    border-radius: 14px;

    padding: 12px 13px;

    z-index: 3;

    box-shadow:
        0 12px 30px
        rgba(0,0,0,.13);
}

.sensor-text {
    left: 5%;
    top: 41px;
}

.sensor-vision {
    right: 5%;
    top: 41px;
}

.sensor-voice {
    left: 5%;
    bottom: 41px;
}

.sensor.active {
    border-color:
        rgba(110,220,255,.30);
}

.sensor.withheld {
    border-color:
        rgba(255,200,90,.46);

    background:
        rgba(52,38,8,.23);
}

.sensor-id {
    color: #71889d;

    font:
        .59rem
        ui-monospace,
        monospace;
}

.sensor-title {
    color: white;

    font-size: .81rem;
    line-height: 1.35;
    font-weight: 760;

    margin-top: 8px;

    max-height: 42px;
    overflow: hidden;
}

.sensor-sub {
    color: #58c9f2;

    font:
        .58rem
        ui-monospace,
        monospace;

    margin-top: 6px;
}

.sensor.withheld
.sensor-sub {
    color: var(--amber);
}


/* ==========================================================
   CONNECTORS
   ========================================================== */

.wire {
    position: absolute;
    height: 1px;
    z-index: 2;

    opacity: .68;

    background:
        linear-gradient(
            90deg,
            rgba(110,220,255,.10),
            rgba(110,220,255,.72)
        );

    box-shadow:
        0 0 7px
        rgba(110,220,255,.08);

    transform-origin: left center;
}


/* LANGUAGE → FUSION */

.wire-a {
    left: calc(5% + 220px);
    top: 108px;

    width:
        calc(
            45% - 305px
        );

    transform:
        rotate(9deg);
}


/* VISION → FUSION */

.wire-b {
    right: calc(5% + 220px);
    top: 108px;

    width:
        calc(
            45% - 305px
        );

    transform-origin:
        right center;

    transform:
        rotate(-9deg);
}


/* VOICE → FUSION */

.wire-c {
    left: calc(5% + 220px);
    bottom: 101px;

    width:
        calc(
            45% - 305px
        );

    transform:
        rotate(-9deg);
}


/* ==========================================================
   FUSION CORE
   ========================================================== */

.fusion-core {
    position: absolute;

    left: 50%;
    top: 91px;

    transform:
        translateX(-50%);

    width: 178px;
    height: 178px;

    display: grid;
    place-items: center;

    z-index: 4;
}

.core-inner {
    width: 103px;
    height: 103px;

    border-radius: 50%;

    display: grid;
    place-items: center;
    align-content: center;

    background:
        radial-gradient(
            circle,
            rgba(61,185,232,.25),
            rgba(7,32,52,.97)
        );

    border:
        1px solid
        rgba(110,220,255,.40);

    box-shadow:
        0 0 45px
        rgba(45,183,240,.16);

    z-index: 3;
}

.core-inner b {
    color: white;
    font-size: 1.8rem;
    line-height: 1;
}

.core-inner span {
    color: #63cef5;

    font:
        .59rem
        ui-monospace,
        monospace;

    letter-spacing: .12em;
    margin-top: 5px;
}

.success
.core-inner {
    border-color:
        rgba(108,240,177,.65);

    box-shadow:
        0 0 48px
        rgba(108,240,177,.16);
}

.warning
.core-inner {
    border-color:
        rgba(255,200,90,.72);

    box-shadow:
        0 0 48px
        rgba(255,200,90,.13);
}

.danger
.core-inner {
    border-color:
        rgba(255,113,134,.78);

    box-shadow:
        0 0 52px
        rgba(255,113,134,.16);
}

.orbit {
    position: absolute;

    border:
        1px solid
        rgba(110,220,255,.15);

    border-radius: 50%;

    animation:
        pulse
        3.6s
        ease-in-out
        infinite;
}

.orbit1 {
    width: 140px;
    height: 140px;
}

.orbit2 {
    width: 176px;
    height: 176px;

    animation-delay: .8s;
}

@keyframes pulse {

    0%,
    100% {
        transform: scale(.97);
        opacity: .28;
    }

    50% {
        transform: scale(1.03);
        opacity: .82;
    }
}


/* ==========================================================
   READOUT
   ========================================================== */

.readout {
    position: absolute;

    right: 5%;
    bottom: 40px;

    width: 246px;

    border:
        1px solid
        rgba(255,255,255,.075);

    background:
        rgba(7,18,31,.95);

    border-radius: 14px;

    padding: 9px 12px;

    z-index: 3;
}

.readout div {
    display: flex;
    justify-content: space-between;

    gap: 12px;

    padding: 5px 0;

    border-bottom:
        1px solid
        rgba(255,255,255,.04);
}

.readout div:last-child {
    border-bottom: none;
}

.readout span {
    color: #6f8599;

    font:
        .55rem
        ui-monospace,
        monospace;
}

.readout b {
    color: #e3ecf2;

    font:
        .58rem
        ui-monospace,
        monospace;

    text-align: right;
}


/* ==========================================================
   INPUT
   ========================================================== */

.dock-title {
    color: #5fcdf6;

    font:
        .63rem
        ui-monospace,
        monospace;

    font-weight: 850;
    letter-spacing: .11em;

    margin-top: 12px;
    margin-bottom: 7px;
}

textarea {
    border-radius:
        12px !important;

    min-height:
        68px !important;
}


/* ==========================================================
   UPLOADERS
   ========================================================== */

[data-testid="stFileUploader"] {
    background:
        rgba(6,15,25,.76);

    border:
        1px solid
        rgba(255,255,255,.055);

    border-radius: 13px;

    padding: 0;
}

[data-testid="stFileUploaderDropzone"] {
    min-height:
        74px !important;

    padding:
        10px !important;

    background:
        rgba(7,17,29,.83) !important;

    border:
        1px dashed
        rgba(110,220,255,.18) !important;

    border-radius:
        12px !important;
}

[data-testid="stFileUploaderDropzone"]
small {
    display: none;
}

[data-testid="stFileUploaderDropzone"]
button {
    min-height:
        32px !important;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {
    min-height: 48px;

    border-radius: 12px;

    font-weight: 850;
    letter-spacing: .055em;

    border:
        1px solid
        rgba(110,220,255,.22);

    box-shadow:
        0 0 26px
        rgba(46,183,240,.10);
}


/* ==========================================================
   PROCESSING
   ========================================================== */

.processing-panel {
    border:
        1px solid
        rgba(110,220,255,.20);

    border-radius: 15px;

    padding: 15px 17px;

    margin: 12px 0;

    background:
        linear-gradient(
            120deg,
            rgba(5,24,39,.96),
            rgba(6,17,28,.96)
        );
}

.processing-top {
    display: flex;
    justify-content: space-between;

    font:
        .59rem
        ui-monospace,
        monospace;

    letter-spacing: .10em;

    color: #69d7fc;
}

.processing-top b {
    color: #72efb6;
}

.processing-title {
    color: white;

    font-size: 1.05rem;
    font-weight: 800;

    margin-top: 8px;
}

.processing-flow {
    display: flex;
    gap: 8px;
    align-items: center;

    margin-top: 11px;

    color: #758b9e;

    font:
        .57rem
        ui-monospace,
        monospace;

    flex-wrap: wrap;
}

.processing-flow strong {
    color: #6edcff;
}

.processing-flow i {
    color: #42637a;
}

.scanner {
    position: relative;
    overflow: hidden;

    height: 2px;

    margin-top: 13px;

    background:
        rgba(110,220,255,.08);
}

.scanner-line {
    width: 35%;
    height: 100%;

    background:
        linear-gradient(
            90deg,
            transparent,
            #6edcff,
            transparent
        );

    animation:
        scan
        1.25s
        linear
        infinite;
}

@keyframes scan {

    from {
        transform:
            translateX(-100%);
    }

    to {
        transform:
            translateX(390%);
    }
}


/* ==========================================================
   OUTPUT
   ========================================================== */

.section-k {
    color: #5fcdf6;

    font:
        .63rem
        ui-monospace,
        monospace;

    font-weight: 850;
    letter-spacing: .11em;

    margin-top: 21px;
}

.section-title {
    color: white;

    font-size: 1.48rem;
    font-weight: 900;

    letter-spacing: -.03em;

    margin-top: 2px;
}


/* ==========================================================
   VISION ANALYSIS
   ========================================================== */

.vision-analysis {
    margin-top: 10px;

    padding: 15px 17px;

    border-radius: 15px;

    background:
        rgba(7,17,29,.82);

    border:
        1px solid
        rgba(110,220,255,.14);
}

.vision-analysis-head {
    display: flex;
    justify-content: space-between;

    gap: 15px;
    align-items: center;
}

.vision-analysis-head span {
    color: #62d4fb;

    font:
        .59rem
        ui-monospace,
        monospace;

    letter-spacing: .10em;
}

.vision-analysis-head b {
    font:
        .59rem
        ui-monospace,
        monospace;

    letter-spacing: .06em;
}

.vision-accepted
.vision-analysis-head b {
    color: #6cf0b1;
}

.vision-withheld {
    border-color:
        rgba(255,200,90,.25);

    background:
        rgba(42,31,8,.18);
}

.vision-withheld
.vision-analysis-head b {
    color: #ffc85a;
}

.vision-analysis-grid {
    display: grid;

    grid-template-columns:
        2fr 1fr 1fr;

    gap: 9px;

    margin-top: 13px;
}

.vision-analysis-grid div {
    padding: 10px 11px;

    border-radius: 10px;

    background:
        rgba(255,255,255,.025);

    border:
        1px solid
        rgba(255,255,255,.055);
}

.vision-analysis-grid span {
    display: block;

    color: #6f8599;

    font:
        .54rem
        ui-monospace,
        monospace;
}

.vision-analysis-grid strong {
    display: block;

    color: #edf5fa;

    margin-top: 5px;

    font-size: .86rem;
}

.vision-analysis p {
    margin:
        11px 0 0;

    color: #8fa2b3;

    font-size: .76rem;

    line-height: 1.45;
}


/* ==========================================================
   ROUTE
   ========================================================== */

.route-board {
    margin-top: 9px;

    border:
        1px solid
        rgba(110,220,255,.18);

    border-radius: 19px;

    background:
        linear-gradient(
            112deg,
            rgba(6,31,51,.99),
            rgba(7,56,84,.92)
        );

    padding: 19px 21px;

    box-shadow:
        0 20px 52px
        rgba(0,0,0,.18);
}

.route-top {
    display: flex;
    justify-content: space-between;

    color: #5fcdf4;

    font:
        .59rem
        ui-monospace,
        monospace;

    letter-spacing: .10em;
}

.route-main {
    display: grid;

    grid-template-columns:
        1fr auto;

    gap: 20px;
    align-items: end;

    margin-top: 8px;
}

.route-name {
    color: white;

    font-weight: 950;

    font-size:
        clamp(
            2.5rem,
            5.5vw,
            4.8rem
        );

    letter-spacing: -.06em;

    line-height: .96;
}

.route-meta {
    color: #9ab0c3;

    margin-top: 9px;

    font:
        .69rem
        ui-monospace,
        monospace;
}

.walk {
    text-align: right;
}

.walk span {
    display: block;

    color: #6f899e;

    font:
        .57rem
        ui-monospace,
        monospace;
}

.walk b {
    display: block;

    color: #76ddff;

    font:
        900
        1.78rem
        ui-monospace,
        monospace;

    margin-top: 3px;
}

.path {
    display: flex;
    align-items: center;

    width:
        min(
            650px,
            100%
        );

    margin-top: 17px;
}

.dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #6edcff;

    box-shadow:
        0 0 14px
        rgba(110,220,255,.55);
}

.line {
    flex: 1;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            #6edcff,
            rgba(110,220,255,.15)
        );

    margin: 0 7px;
}

.arrow {
    color: #6edcff;

    font-size: 1.15rem;

    margin-right: 7px;
}

.directions {
    color: #e4edf3;

    line-height: 1.55;

    margin-top: 11px;

    font-size: .89rem;
}


/* ==========================================================
   VERIFICATION
   ========================================================== */

.verification-strip {
    display: flex;
    align-items: center;

    gap: 11px;

    margin-top: 8px;

    padding: 9px 12px;

    border:
        1px solid
        rgba(255,200,90,.16);

    border-radius: 10px;

    background:
        rgba(255,200,90,.035);
}

.verification-strip span {
    color: #ffc85a;

    font:
        .57rem
        ui-monospace,
        monospace;

    letter-spacing: .09em;

    font-weight: 850;
}

.verification-strip p {
    color: #c7b47e;

    font-size: .76rem;

    margin: 0;
}


/* ==========================================================
   CONFLICT
   ========================================================== */

.conflict-banner {
    display: grid;

    grid-template-columns:
        1fr
        190px
        1fr;

    gap: 8px;

    margin-top: 10px;
    margin-bottom: 11px;
}

.conflict-evidence {
    border:
        1px solid
        rgba(255,200,90,.18);

    background:
        rgba(255,200,90,.035);

    border-radius: 13px;

    padding: 12px 13px;
}

.conflict-evidence span {
    color: #a98942;

    font:
        .56rem
        ui-monospace,
        monospace;

    letter-spacing: .09em;
}

.conflict-evidence b {
    display: block;

    color: #ffe19c;

    margin-top: 6px;

    font-size: .81rem;
}

.conflict-evidence.right {
    text-align: right;
}

.conflict-middle {
    border:
        1px solid
        rgba(255,200,90,.30);

    background:
        rgba(255,200,90,.07);

    border-radius: 13px;

    display: grid;
    place-items: center;
    align-content: center;

    text-align: center;

    padding: 8px 10px;
}

.warning-icon {
    width: 29px;
    height: 29px;

    border-radius: 50%;

    display: grid;
    place-items: center;

    background: #ffc85a;

    color: #211800;

    font-weight: 950;

    margin-bottom: 5px;
}

.conflict-middle b {
    color: #ffd77d;

    font:
        .62rem
        ui-monospace,
        monospace;
}

.conflict-middle span {
    color: #a88e56;

    font-size: .61rem;

    margin-top: 3px;
}


/* ==========================================================
   BLOCKED
   ========================================================== */

.blocked-panel {
    display: grid;

    grid-template-columns:
        52px
        1fr
        auto;

    gap: 14px;

    align-items: center;

    margin-top: 10px;

    padding: 17px;

    border:
        1px solid
        rgba(255,113,134,.28);

    border-radius: 16px;

    background:
        linear-gradient(
            110deg,
            rgba(65,17,28,.28),
            rgba(18,7,13,.62)
        );

    box-shadow:
        0 0 42px
        rgba(255,113,134,.05);
}

.blocked-symbol {
    width: 44px;
    height: 44px;

    border-radius: 50%;

    display: grid;
    place-items: center;

    border:
        1px solid
        rgba(255,113,134,.50);

    color: #ff8a9c;

    font-size: 1.5rem;

    font-weight: 700;
}

.blocked-kicker {
    color: #ff8799;

    font:
        .57rem
        ui-monospace,
        monospace;

    letter-spacing: .10em;
}

.blocked-title {
    color: white;

    font-size: 1.12rem;

    font-weight: 850;

    margin-top: 3px;
}

.blocked-copy {
    color: #ad8790;

    font-size: .78rem;

    margin-top: 4px;
}

.blocked-state {
    color: #ff8b9c;

    font:
        850
        .72rem
        ui-monospace,
        monospace;

    border:
        1px solid
        rgba(255,113,134,.27);

    border-radius: 8px;

    padding: 7px 10px;
}


/* ==========================================================
   METRICS
   ========================================================== */

[data-testid="stMetric"] {
    background:
        rgba(7,17,29,.72);

    border:
        1px solid
        rgba(255,255,255,.055);

    border-radius: 12px;

    padding: 10px 12px;
}

[data-testid="stMetricLabel"] {
    font-size:
        .72rem !important;
}

[data-testid="stMetricValue"] {
    font-size:
        1.45rem !important;
}


/* ==========================================================
   EXPANDER
   ========================================================== */

div[data-testid="stExpander"] {
    border-radius: 11px;

    background:
        rgba(6,15,26,.62);

    border:
        1px solid
        rgba(255,255,255,.055);
}


/* ==========================================================
   FOOTER
   ========================================================== */

.foot {
    color: #516779;

    text-align: center;

    padding-top: 22px;

    font:
        .57rem
        ui-monospace,
        monospace;

    letter-spacing: .05em;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media(max-width:900px) {

    .nia-right {
        display: none;
    }

    .fusion-stage {
        min-height: auto;

        padding: 15px;
    }

    .sensor,
    .fusion-core,
    .readout,
    .wire {
        position: relative;

        left: auto;
        right: auto;
        top: auto;
        bottom: auto;

        transform: none;

        width: 100%;

        margin:
            9px 0;
    }

    .fusion-core {
        width: 178px;
        height: 178px;

        margin:
            16px auto;
    }

    .wire {
        display: none;
    }

    .route-main {
        grid-template-columns:
            1fr;
    }

    .walk {
        text-align: left;
    }

    .conflict-banner {
        grid-template-columns:
            1fr;
    }

    .conflict-evidence.right {
        text-align: left;
    }

    .vision-analysis-grid {
        grid-template-columns:
            1fr;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="nia-nav">'
    '<div class="nia-left">'
    '<div class="nia-mark">A</div>'
    '<div>'
    '<div class="nia-brand">AeroAssist AI</div>'
    '<div class="nia-sub">NIA / MULTIMODAL PASSENGER ASSISTANCE</div>'
    '</div>'
    '</div>'
    '<div class="nia-right">'
    '<span class="chip live">● SYSTEM READY</span>'
    '<span class="chip">CLIP</span>'
    '<span class="chip">WHISPER</span>'
    '<span class="chip">FUSION</span>'
    '<span class="chip">SAFETY GATE</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="prompt">'
    '<div class="prompt-kicker">'
    'PASSENGER INTELLIGENCE CORE'
    '</div>'
    '<div class="prompt-title">'
    'What does the passenger need?'
    '</div>'
    '<div class="prompt-sub">'
    'Multimodal evidence is fused before guidance is released.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT CORE
# ============================================================

current_result = st.session_state.get(
    "latest_result"
)


st.markdown(
    build_core(
        current_result
    ),
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="dock-title">'
    'INPUT DOCK / TEXT · VISION · VOICE'
    '</div>',
    unsafe_allow_html=True,
)


text_query = st.text_area(
    "Passenger request",
    placeholder=(
        "Ask AeroAssist — for example: "
        "Where is Gate B12?"
    ),
    height=70,
    label_visibility="collapsed",
)


image_col, audio_col = st.columns(
    2
)


with image_col:

    uploaded_image = st.file_uploader(
        "Visual evidence",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="airport_image",
        label_visibility="collapsed",
    )


with audio_col:

    uploaded_audio = st.file_uploader(
        "Speech evidence",
        type=[
            "m4a",
            "mp3",
            "wav",
        ],
        key="passenger_audio",
        label_visibility="collapsed",
    )


if (
    uploaded_audio is not None
    and text_query.strip()
):

    st.caption(
        "Voice precedence active: the Whisper transcript "
        "will become the language evidence used during fusion."
    )


run_button = st.button(
    "ANALYSE EVIDENCE",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESSING SLOT
# ============================================================

processing_slot = st.empty()


# ============================================================
# RUN
# ============================================================

if run_button:

    ambiguous_gate = ambiguous_gate_query(
        text_query
    )

    if ambiguous_gate:

        st.warning(
            f"Gate {ambiguous_gate} is ambiguous. "
            f"Please specify the concourse letter, for example "
            f"Gate A{ambiguous_gate} or Gate B{ambiguous_gate}."
        )

    elif (
        not text_query.strip()
        and uploaded_image is None
        and uploaded_audio is None
    ):

        st.warning(
            "Add at least one source of passenger evidence."
        )

    else:

        image_path = None
        audio_path = None


        if uploaded_image is not None:

            image_path = save_uploaded_file(
                uploaded_image
            )


        if uploaded_audio is not None:

            audio_path = save_uploaded_file(
                uploaded_audio
            )


        processing_slot.markdown(
            build_loading(),
            unsafe_allow_html=True,
        )


        try:

            pipeline = load_pipeline()


            # =================================================
            # FROZEN BACKEND CALL — DO NOT CHANGE
            # =================================================

            result = pipeline.run(
                text=text_query.strip(),
                image_path=image_path,
                audio_path=audio_path,
            )


            st.session_state[
                "latest_result"
            ] = result


            processing_slot.empty()


            st.rerun()


        except Exception as error:

            processing_slot.empty()


            st.error(
                "AeroAssist could not process this request."
            )


            st.exception(
                error
            )


# ============================================================
# OUTPUT
# ============================================================

if current_result:

    result = current_result

    fusion = result[
        "fusion_result"
    ]

    response = result[
        "passenger_response"
    ]

    record = get_record(
        result
    )

    confidence = fusion.get(
        "confidence",
        "uncertain",
    )

    behavior = fusion.get(
        "behavior",
        "unknown",
    )

    conflict = bool(
        fusion.get(
            "conflict"
        )
    )

    safe = bool(
        response.get(
            "safe_to_display_record"
        )
    )


    st.markdown(
        '<div class="section-k">'
        'DECISION OUTPUT'
        '</div>'
        '<div class="section-title">'
        'Passenger guidance'
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # CONFLICT
    # ========================================================

    if conflict:

        st.markdown(
            build_conflict(
                result
            ),
            unsafe_allow_html=True,
        )


    # ========================================================
    # VISION EXPLAINABILITY
    # ========================================================

    if result.get(
        "vision_result"
    ):

        st.markdown(
            build_vision_analysis(
                result
            ),
            unsafe_allow_html=True,
        )


    # ========================================================
    # SAFE ROUTE
    # ========================================================

    if (
        safe
        and record
    ):

        st.markdown(
            build_route(
                record,
                behavior,
            ),
            unsafe_allow_html=True,
        )


        verification_html = (
            build_verification(
                record
            )
        )


        if verification_html:

            st.markdown(
                verification_html,
                unsafe_allow_html=True,
            )


    # ========================================================
    # ABSTENTION
    # ========================================================

    else:

        st.markdown(
            build_blocked(),
            unsafe_allow_html=True,
        )


        st.error(
            response["message"]
        )


    # ========================================================
    # CAUTION
    # ========================================================

    if (
        safe
        and confidence == "caution"
    ):

        st.warning(
            response["message"]
        )


    # ========================================================
    # STATUS METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(
        4
    )


    with m1:

        st.metric(
            "Confidence",
            confidence_label(
                confidence
            ),
        )


    with m2:

        modality_count = len(
            result.get(
                "modalities",
                [],
            )
        )

        if conflict:
            evidence_state = "CONFLICT"

        elif modality_count <= 1:
            evidence_state = "SINGLE SOURCE"

        else:
            evidence_state = "ALIGNED"

        st.metric(
            "Evidence",
            evidence_state,
        )


    with m3:

        st.metric(
            "Modalities",
            len(
                result.get(
                    "modalities",
                    [],
                )
            ),
        )


    with m4:

        st.metric(
            "Safety gate",
            (
                "PASS"
                if safe
                else "BLOCK"
            ),
        )


    # ========================================================
    # TECHNICAL TRACE
    # ========================================================

    with st.expander(
        "Technical trace"
    ):

        st.write(
            "**Modalities:**",
            result.get(
                "modalities",
                [],
            ),
        )


        st.write(
            "**Behaviour:**",
            fusion.get(
                "behavior"
            ),
        )


        st.write(
            "**Confidence:**",
            fusion.get(
                "confidence"
            ),
        )


        st.write(
            "**Conflict:**",
            fusion.get(
                "conflict"
            ),
        )


        st.write(
            "**Reason:**",
            fusion.get(
                "reason"
            ),
        )


        st.write(
            "**Fusion image label:**",
            (
                result.get(
                    "fusion_image_label"
                )
                or "WITHHELD"
            ),
        )


        st.write(
            "**Passenger safe record:**",
            response.get(
                "safe_to_display_record"
            ),
        )


        st.markdown(
            "#### Passenger-facing response"
        )


        st.write(
            response.get(
                "message"
            )
        )


        if result.get(
            "vision_result"
        ):

            st.markdown(
                "#### CLIP"
            )


            st.json(
                result[
                    "vision_result"
                ]
            )


        if result.get(
            "transcription"
        ):

            st.markdown(
                "#### Whisper"
            )


            st.json(
                result[
                    "transcription"
                ]
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="foot">'
    'AEROASSIST AI · CONTEXT-AWARE MULTIMODAL FUSION · '
    'CONFIDENCE-SENSITIVE PASSENGER ASSISTANCE'
    '</div>',
    unsafe_allow_html=True,
)