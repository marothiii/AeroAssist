from pathlib import Path
import tempfile
import html

import streamlit as st

from src.end_to_end_pipeline import AeroAssistEndToEndPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroAssist AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #050b14;
    --panel: #0a1423;
    --panel-soft: #0d1b2e;
    --panel-blue: #0c2845;
    --line: rgba(255, 255, 255, 0.08);
    --line-blue: rgba(82, 190, 255, 0.22);
    --text: #f8fafc;
    --muted: #91a3b8;
    --cyan: #72d5ff;
    --blue: #1585d8;
    --green: #4ade80;
    --orange: #fbbf24;
    --red: #fb7185;
}

.stApp {
    background:
        radial-gradient(
            circle at 88% 3%,
            rgba(22, 118, 190, 0.17),
            transparent 28%
        ),
        radial-gradient(
            circle at 18% 80%,
            rgba(14, 65, 118, 0.08),
            transparent 26%
        ),
        var(--bg);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.6rem;
    padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #09182a 0%,
            #07111f 100%
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.7rem;
}

[data-testid="stMetric"] {
    background: rgba(11, 25, 42, 0.80);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 16px 13px 16px;
}

[data-testid="stMetricLabel"] {
    color: #91a3b8;
}

[data-testid="stMetricValue"] {
    font-weight: 750;
}

.stButton > button {
    min-height: 54px;
    border-radius: 14px;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 0.01em;
}

[data-testid="stFileUploader"] {
    border-radius: 15px;
}

textarea {
    border-radius: 15px !important;
}

div[data-testid="stExpander"] {
    background: rgba(9, 20, 35, 0.58);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 15px;
}

/* ---------------------------------------------------------
   BRAND
--------------------------------------------------------- */

.sidebar-logo {
    font-size: 1.65rem;
    font-weight: 850;
    color: white;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}

.sidebar-sub {
    color: #8fa3bb;
    font-size: 0.87rem;
    line-height: 1.5;
}

.sidebar-code {
    display: inline-block;
    margin-top: 12px;
    border: 1px solid rgba(114, 213, 255, 0.25);
    background: rgba(114, 213, 255, 0.07);
    color: #72d5ff;
    font-family: monospace;
    font-size: 0.75rem;
    padding: 5px 9px;
    border-radius: 8px;
}

.status-row {
    padding: 9px 11px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: rgba(74, 222, 128, 0.07);
    border: 1px solid rgba(74, 222, 128, 0.18);
    color: #a7f3d0;
    font-size: 0.84rem;
}

/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 32px 38px 31px 38px;
    margin-bottom: 28px;
    border: 1px solid rgba(89, 190, 255, 0.22);
    background:
        linear-gradient(
            110deg,
            rgba(12, 38, 68, 0.98),
            rgba(8, 76, 119, 0.92)
        );
    box-shadow:
        0 24px 65px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero::after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -120px;
    top: -160px;
    border-radius: 50%;
    background: rgba(88, 201, 255, 0.09);
}

.hero-topline {
    color: #7ddcff;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 11px;
}

.hero-title {
    color: white;
    font-size: 3.4rem;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -0.045em;
    margin-bottom: 15px;
}

.hero-subtitle {
    color: #d0deea;
    max-width: 780px;
    font-size: 1.08rem;
    line-height: 1.65;
}

.hero-tags {
    margin-top: 20px;
}

.hero-tag {
    display: inline-block;
    margin-right: 7px;
    margin-bottom: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.13);
    background: rgba(255,255,255,0.07);
    color: #e4eef7;
    font-size: 0.80rem;
}

/* ---------------------------------------------------------
   SECTION TITLES
--------------------------------------------------------- */

.kicker {
    color: #72d5ff;
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.title {
    color: white;
    font-size: 1.75rem;
    font-weight: 850;
    letter-spacing: -0.02em;
    margin-bottom: 5px;
}

.section-sub {
    color: #8fa3b8;
    font-size: 0.93rem;
    margin-bottom: 18px;
}

/* ---------------------------------------------------------
   LIVE AIRPORT BAR
--------------------------------------------------------- */

.ops-bar {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    background: rgba(8, 20, 34, 0.72);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 11px 13px;
    margin-bottom: 23px;
}

.ops-item {
    color: #94a9bf;
    font-family: monospace;
    font-size: 0.77rem;
}

.ops-strong {
    color: #dff6ff;
    font-weight: 700;
}

.ops-live {
    color: #86efac;
    font-weight: 800;
}

/* ---------------------------------------------------------
   JOURNEY STRIP
--------------------------------------------------------- */

.journey-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 14px 0 24px 0;
}

.journey-node {
    text-align: center;
    padding: 11px 6px;
    border-radius: 11px;
    color: #7990a6;
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.05em;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(9, 21, 35, 0.62);
}

.journey-active {
    color: #dff7ff;
    border: 1px solid rgba(77, 201, 255, 0.30);
    background: rgba(14, 112, 173, 0.18);
    box-shadow: inset 0 0 24px rgba(35, 164, 230, 0.05);
}

/* ---------------------------------------------------------
   RESULT BOARD
--------------------------------------------------------- */

.route-board {
    border-radius: 22px;
    padding: 24px 26px;
    background:
        linear-gradient(
            120deg,
            rgba(10, 27, 46, 0.98),
            rgba(9, 38, 62, 0.96)
        );
    border: 1px solid rgba(84, 195, 255, 0.20);
    box-shadow: 0 18px 48px rgba(0,0,0,0.18);
    margin-bottom: 16px;
}

.route-label {
    color: #72d5ff;
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.17em;
    text-transform: uppercase;
}

.route-name {
    color: white;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.035em;
    margin-top: 4px;
}

.route-meta {
    color: #9db3c8;
    font-family: monospace;
    font-size: 0.82rem;
    margin-top: 4px;
}

.route-line {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 18px 0;
}

.route-directions {
    color: #e6eef5;
    line-height: 1.65;
    font-size: 0.96rem;
}

.route-time {
    color: #7ddcff;
    font-weight: 850;
}

/* ---------------------------------------------------------
   FUSION / EVIDENCE
--------------------------------------------------------- */

.fusion-header {
    background: rgba(10, 23, 39, 0.78);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 17px 18px;
    margin-bottom: 12px;
}

.fusion-title {
    color: white;
    font-weight: 850;
    font-size: 1.05rem;
}

.fusion-subtitle {
    color: #8399ae;
    font-size: 0.82rem;
    margin-top: 3px;
}

.decision-pill {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 10px;
    border-radius: 999px;
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 750;
    border: 1px solid rgba(114, 213, 255, 0.23);
    background: rgba(114, 213, 255, 0.07);
    color: #92e3ff;
}

.conflict-banner {
    border-radius: 16px;
    border: 1px solid rgba(251, 191, 36, 0.30);
    background: rgba(251, 191, 36, 0.08);
    padding: 15px 17px;
    color: #fde68a;
    margin: 12px 0;
}

.safety-banner {
    border-radius: 16px;
    border: 1px solid rgba(251, 113, 133, 0.30);
    background: rgba(251, 113, 133, 0.08);
    padding: 15px 17px;
    color: #fecdd3;
    margin: 12px 0;
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    margin-top: 38px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.07);
    text-align: center;
    color: #64788f;
    font-size: 0.78rem;
}

.footer-code {
    font-family: monospace;
    color: #72d5ff;
    font-weight: 800;
}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD FROZEN MODELS
# ============================================================

@st.cache_resource
def load_pipeline():
    return AeroAssistEndToEndPipeline(
        use_vision=True,
        use_voice=True,
    )


# ============================================================
# HELPERS
# ============================================================

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_file.write(uploaded_file.getbuffer())
    temporary_file.close()

    return Path(temporary_file.name)


def confidence_text(confidence):
    if confidence == "confident":
        return "HIGH"
    if confidence == "caution":
        return "CAUTION"
    return "UNCERTAIN"


def confidence_icon(confidence):
    if confidence == "confident":
        return "●"
    if confidence == "caution":
        return "▲"
    return "■"


def clean_behavior(value):
    if not value:
        return "Unknown"
    return value.replace("_", " ").title()


def safe_html(value):
    if value is None:
        return ""
    return html.escape(str(value))


def get_record(result):
    passenger_response = result.get(
        "passenger_response",
        {},
    )

    record = passenger_response.get(
        "record"
    )

    if isinstance(record, dict):
        return record

    return None


def infer_journey_stage(record):
    if not record:
        return "concourse"

    category = str(
        record.get("category", "")
    ).lower()

    if "check" in category:
        return "checkin"

    if "security" in category:
        return "security"

    if "gate" in category:
        return "gate"

    return "concourse"


def journey_strip(active_stage):
    stages = [
        ("checkin", "01  CHECK-IN"),
        ("security", "02  SECURITY"),
        ("concourse", "03  CONCOURSE"),
        ("gate", "04  GATE"),
    ]

    parts = [
        '<div class="journey-strip">'
    ]

    for stage_key, stage_label in stages:

        css_class = "journey-node"

        if stage_key == active_stage:
            css_class += " journey-active"

        parts.append(
            f'<div class="{css_class}">'
            f'{stage_label}'
            f'</div>'
        )

    parts.append("</div>")

    return "".join(parts)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">✈ AeroAssist</div>'
        '<div class="sidebar-sub">'
        'Confidence-sensitive multimodal passenger intelligence'
        '</div>'
        '<div class="sidebar-code">NIA / SYSTEM ONLINE</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### AI stack")

    st.markdown(
        '<div class="status-row">● Semantic language engine</div>'
        '<div class="status-row">● CLIP visual intelligence</div>'
        '<div class="status-row">● Whisper speech recognition</div>'
        '<div class="status-row">● Multimodal fusion controller</div>'
        '<div class="status-row">● Passenger safety gate</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Research focus")

    st.caption(
        "AeroAssist investigates whether confidence-sensitive "
        "multimodal fusion can improve the reliability and usefulness "
        "of airport passenger assistance."
    )

    st.divider()

    st.markdown("#### Safety policy")

    st.caption(
        "Low-confidence evidence may be withheld. Conflicting "
        "modalities trigger an explicit warning. Critical cases "
        "can be redirected to human assistance."
    )

    st.divider()

    st.caption(
        "Nova International Airport is a fictional knowledge base "
        "used for this MSc Artificial Intelligence proof-of-concept."
    )


# ============================================================
# HERO
# ============================================================

hero_html = """
<div class="hero">
<div class="hero-topline">NOVA INTERNATIONAL AIRPORT • INTELLIGENT PASSENGER SYSTEM</div>
<div class="hero-title">AeroAssist AI</div>
<div class="hero-subtitle">
A confidence-sensitive multimodal assistant that combines passenger language,
speech and airport imagery before deciding whether guidance is reliable enough
to show.
</div>
<div class="hero-tags">
<span class="hero-tag">⌨ TEXT</span>
<span class="hero-tag">🎤 WHISPER</span>
<span class="hero-tag">👁 CLIP VISION</span>
<span class="hero-tag">🧠 FUSION</span>
<span class="hero-tag">⚠ CONFLICT DETECTION</span>
<span class="hero-tag">🛡 SAFE ABSTENTION</span>
</div>
</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# ============================================================
# OPERATIONS STRIP
# ============================================================

st.markdown(
    '<div class="ops-bar">'
    '<span class="ops-item ops-live">● SYSTEM READY</span>'
    '<span class="ops-item">|</span>'
    '<span class="ops-item">AIRPORT <span class="ops-strong">NIA</span></span>'
    '<span class="ops-item">|</span>'
    '<span class="ops-item">MODE <span class="ops-strong">MULTIMODAL</span></span>'
    '<span class="ops-item">|</span>'
    '<span class="ops-item">SAFETY <span class="ops-strong">ACTIVE</span></span>'
    '<span class="ops-item">|</span>'
    '<span class="ops-item">KNOWLEDGE BASE <span class="ops-strong">25 SERVICES</span></span>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PASSENGER REQUEST
# ============================================================

st.markdown(
    '<div class="kicker">01 / PASSENGER INPUT</div>'
    '<div class="title">Multimodal request console</div>'
    '<div class="section-sub">'
    'Ask naturally, upload an airport image, speak a request — '
    'or combine modalities so AeroAssist can compare evidence.'
    '</div>',
    unsafe_allow_html=True,
)


text_query = st.text_area(
    "⌨ Passenger language",
    placeholder=(
        "Where is Gate B12?\n"
        "I can't use stairs and my flight boards soon."
    ),
    height=110,
)


input_col1, input_col2 = st.columns(2)


with input_col1:

    st.markdown("### 👁 Visual channel")

    uploaded_image = st.file_uploader(
        "Airport image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="airport_image",
        help=(
            "Upload a gate sign, baggage area, security checkpoint, "
            "transport area or another airport scene."
        ),
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Visual evidence supplied to CLIP",
            use_container_width=True,
        )


with input_col2:

    st.markdown("### 🎤 Speech channel")

    uploaded_audio = st.file_uploader(
        "Passenger voice",
        type=[
            "m4a",
            "mp3",
            "wav",
        ],
        key="passenger_audio",
        help="Upload a spoken passenger request.",
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)


if (
    uploaded_audio is not None
    and text_query.strip()
):

    st.caption(
        "ℹ In the current proof-of-concept, when both typed language "
        "and voice are supplied, the Whisper transcript is used as "
        "the language input for fusion."
    )


st.write("")

run_button = st.button(
    "✈ RUN AEROASSIST ANALYSIS",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN PIPELINE
# ============================================================

if run_button:

    if (
        not text_query.strip()
        and uploaded_image is None
        and uploaded_audio is None
    ):

        st.warning(
            "Provide text, voice, an airport image, "
            "or a combination of inputs."
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

        with st.spinner(
            "Analysing passenger evidence across available modalities..."
        ):

            try:

                pipeline = load_pipeline()

                result = pipeline.run(
                    text=text_query.strip(),
                    image_path=image_path,
                    audio_path=audio_path,
                )

                st.session_state[
                    "latest_result"
                ] = result

            except Exception as error:

                st.error(
                    "AeroAssist could not process this request."
                )

                st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if "latest_result" in st.session_state:

    result = st.session_state[
        "latest_result"
    ]

    fusion = result[
        "fusion_result"
    ]

    passenger_response = result[
        "passenger_response"
    ]

    confidence = fusion.get(
        "confidence",
        "uncertain",
    )

    record = get_record(result)

    st.write("")
    st.divider()

    st.markdown(
        '<div class="kicker">02 / FUSED DECISION</div>'
        '<div class="title">Passenger guidance</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # JOURNEY VISUAL
    # ========================================================

    active_stage = infer_journey_stage(
        record
    )

    st.markdown(
        journey_strip(
            active_stage
        ),
        unsafe_allow_html=True,
    )


    # ========================================================
    # PASSENGER-FACING RESPONSE
    # ========================================================

    if record and passenger_response.get(
        "safe_to_display_record"
    ):

        record_name = safe_html(
            record.get(
                "name",
                fusion.get("record", "Airport service"),
            )
        )

        terminal = safe_html(
            record.get("terminal", "")
        )

        zone = safe_html(
            record.get("zone", "")
        )

        floor = safe_html(
            record.get("floor", "")
        )

        directions = safe_html(
            record.get("directions", "")
        )

        walking_minutes = record.get(
            "walking_minutes"
        )

        meta_parts = [
            value
            for value in [
                terminal,
                zone,
                floor,
            ]
            if value
        ]

        meta_line = " • ".join(
            meta_parts
        )

        route_html = (
            '<div class="route-board">'
            '<div class="route-label">MATCHED AIRPORT DESTINATION</div>'
            f'<div class="route-name">{record_name}</div>'
        )

        if meta_line:

            route_html += (
                f'<div class="route-meta">{meta_line}</div>'
            )

        if directions:

            route_html += (
                '<div class="route-line"></div>'
                f'<div class="route-directions">'
                f'{directions}'
                f'</div>'
            )

        if walking_minutes is not None:

            route_html += (
                '<div class="route-line"></div>'
                f'<div class="route-time">'
                f'→ APPROX. {walking_minutes} MIN WALK'
                f'</div>'
            )

        route_html += "</div>"

        st.markdown(
            route_html,
            unsafe_allow_html=True,
        )


    # ========================================================
    # RICH PASSENGER MESSAGE
    # ========================================================

    if confidence == "confident":

        st.success(
            passenger_response[
                "message"
            ]
        )

    elif confidence == "caution":

        st.warning(
            passenger_response[
                "message"
            ]
        )

    else:

        st.error(
            passenger_response[
                "message"
            ]
        )


    # ========================================================
    # DECISION METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Decision confidence",
            (
                f"{confidence_icon(confidence)} "
                f"{confidence_text(confidence)}"
            ),
        )

    with metric2:

        st.metric(
            "Cross-modal state",
            (
                "CONFLICT"
                if fusion.get("conflict")
                else "ALIGNED"
            ),
        )

    with metric3:

        st.metric(
            "Evidence channels",
            len(
                result.get(
                    "modalities",
                    [],
                )
            ),
        )

    with metric4:

        st.metric(
            "Passenger safety gate",
            (
                "PASSED"
                if passenger_response.get(
                    "safe_to_display_record"
                )
                else "BLOCKED"
            ),
        )


    # ========================================================
    # SPECIAL SAFETY STATES
    # ========================================================

    if fusion.get("conflict"):

        st.markdown(
            '<div class="conflict-banner">'
            '<b>⚠ MULTIMODAL CONFLICT DETECTED</b><br>'
            'Visual and language evidence disagree. '
            'AeroAssist has reduced confidence and exposed the '
            'disagreement instead of silently merging conflicting inputs.'
            '</div>',
            unsafe_allow_html=True,
        )


    if not passenger_response.get(
        "safe_to_display_record"
    ):

        st.markdown(
            '<div class="safety-banner">'
            '<b>🛡 SAFETY GATE ACTIVATED</b><br>'
            'An internal candidate may exist, but AeroAssist judged '
            'the evidence too uncertain to expose location guidance '
            'to the passenger.'
            '</div>',
            unsafe_allow_html=True,
        )


    # ========================================================
    # MULTIMODAL FUSION VISUAL
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="kicker">03 / EXPLAINABLE MULTIMODAL AI</div>'
        '<div class="title">Evidence fusion console</div>'
        '<div class="section-sub">'
        'AeroAssist exposes which evidence entered the decision, '
        'which evidence was trusted, and whether anything was withheld.'
        '</div>',
        unsafe_allow_html=True,
    )

    evidence1, evidence2, evidence3 = st.columns(3)


    # --------------------------------------------------------
    # LANGUAGE EVIDENCE
    # --------------------------------------------------------

    with evidence1:

        st.markdown(
            '<div class="fusion-header">'
            '<div class="fusion-title">🗣 LANGUAGE</div>'
            '<div class="fusion-subtitle">'
            'Typed text or Whisper transcription'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        transcription = result.get(
            "transcription"
        )

        if transcription:

            st.caption(
                "WHISPER TRANSCRIPT"
            )

            st.info(
                transcription.get(
                    "transcript",
                    "",
                )
            )

            st.caption(
                "Speech converted to text before "
                "semantic intent processing."
            )

        elif result.get(
            "input_text"
        ):

            st.caption(
                "TYPED PASSENGER QUERY"
            )

            st.info(
                result[
                    "input_text"
                ]
            )

        else:

            st.caption(
                "No language evidence supplied."
            )


    # --------------------------------------------------------
    # VISION EVIDENCE
    # --------------------------------------------------------

    with evidence2:

        st.markdown(
            '<div class="fusion-header">'
            '<div class="fusion-title">👁 VISION</div>'
            '<div class="fusion-subtitle">'
            'CLIP airport scene classification'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        vision = result.get(
            "vision_result"
        )

        if vision:

            prediction = (
                vision[
                    "predicted_class"
                ]
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            st.caption(
                "CLIP PREDICTION"
            )

            st.info(
                prediction
            )

            st.write(
                f"**Confidence label:** "
                f"{vision['confidence'].upper()}"
            )

            st.write(
                f"**Prediction score:** "
                f"{vision['score']:.3f}"
            )

            st.progress(
                float(
                    max(
                        0.0,
                        min(
                            1.0,
                            vision[
                                "score"
                            ],
                        ),
                    )
                )
            )

            st.write(
                f"**Decision margin:** "
                f"{vision['margin']:.3f}"
            )

            if result.get(
                "fusion_image_label"
            ):

                st.success(
                    "Visual evidence entered fusion."
                )

            else:

                st.warning(
                    "Visual evidence withheld from fusion."
                )

        else:

            st.caption(
                "No image evidence supplied."
            )


    # --------------------------------------------------------
    # FUSION EVIDENCE
    # --------------------------------------------------------

    with evidence3:

        st.markdown(
            '<div class="fusion-header">'
            '<div class="fusion-title">🧠 FUSION</div>'
            '<div class="fusion-subtitle">'
            'Confidence-sensitive decision controller'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        behavior = fusion.get(
            "behavior",
            "unknown",
        )

        st.caption(
            "FINAL BEHAVIOUR"
        )

        st.info(
            clean_behavior(
                behavior
            )
        )

        st.markdown(
            f'<span class="decision-pill">'
            f'{safe_html(behavior).upper()}'
            f'</span>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.write(
            "**Decision rationale**"
        )

        st.write(
            fusion.get(
                "reason",
                "No reason provided.",
            )
        )

        st.write(
            "**Safety permission:** "
            + (
                "PASS"
                if passenger_response.get(
                    "safe_to_display_record"
                )
                else "BLOCK"
            )
        )


    # ========================================================
    # FUSION FLOW
    # ========================================================

    st.write("")

    modalities = result.get(
        "modalities",
        [],
    )

    modality_names = []

    if "text" in modalities:
        modality_names.append("LANGUAGE")

    if "voice" in modalities:
        modality_names.append("VOICE")

    if "image" in modalities:
        modality_names.append("VISION")

    modality_display = (
        " + ".join(
            modality_names
        )
        if modality_names
        else "NO EVIDENCE"
    )

    safety_state = (
        "PASSENGER RESPONSE"
        if passenger_response.get(
            "safe_to_display_record"
        )
        else "SAFE ABSTENTION"
    )

    st.markdown(
        '<div class="ops-bar">'
        f'<span class="ops-item ops-strong">{safe_html(modality_display)}</span>'
        '<span class="ops-item">→</span>'
        '<span class="ops-item ops-strong">SEMANTIC + CONTEXT</span>'
        '<span class="ops-item">→</span>'
        '<span class="ops-item ops-strong">MULTIMODAL FUSION</span>'
        '<span class="ops-item">→</span>'
        '<span class="ops-item ops-strong">CONFIDENCE GATE</span>'
        '<span class="ops-item">→</span>'
        f'<span class="ops-item ops-live">{safe_html(safety_state)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # INTERNAL RECORD
    # ========================================================

    if (
        passenger_response.get(
            "safe_to_display_record"
        )
        and fusion.get(
            "record"
        )
    ):

        st.caption(
            "Knowledge-base match: "
            f"{fusion['record']} • "
            "Nova International Airport"
        )


    # ========================================================
    # TECHNICAL TRACE
    # ========================================================

    st.write("")

    with st.expander(
        "🔬 Research / technical decision trace"
    ):

        st.caption(
            "Developer-facing evidence used for evaluation "
            "and explainability. This section is not required "
            "for normal passenger use."
        )

        trace1, trace2 = st.columns(2)

        with trace1:

            st.write(
                "**Modalities received:**",
                result.get(
                    "modalities",
                    [],
                ),
            )

            st.write(
                "**Fusion image label:**",
                result.get(
                    "fusion_image_label"
                )
                or "WITHHELD",
            )

            st.write(
                "**Internal KB record:**",
                fusion.get(
                    "record"
                ),
            )

            st.write(
                "**Fusion behaviour:**",
                fusion.get(
                    "behavior"
                ),
            )

        with trace2:

            st.write(
                "**Fusion confidence:**",
                fusion.get(
                    "confidence"
                ),
            )

            st.write(
                "**Conflict detected:**",
                fusion.get(
                    "conflict"
                ),
            )

            st.write(
                "**Passenger record exposure:**",
                passenger_response.get(
                    "safe_to_display_record"
                ),
            )

            st.write(
                "**Fusion reason:**",
                fusion.get(
                    "reason"
                ),
            )


        if result.get(
            "vision_result"
        ):

            st.markdown(
                "#### Raw CLIP output"
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
                "#### Raw Whisper output"
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
    '<div class="footer">'
    '<span class="footer-code">NIA / AEROASSIST</span>'
    ' &nbsp;•&nbsp; MSc Artificial Intelligence Proof-of-Concept'
    ' &nbsp;•&nbsp; CLIP + Whisper + Semantic Retrieval'
    ' + Confidence-Sensitive Multimodal Fusion'
    '</div>',
    unsafe_allow_html=True,
)