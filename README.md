# AeroAssist

AeroAssist is a multimodal AI prototype designed to support passenger navigation and assistance in an airport environment. The system processes text, speech and image inputs and combines available evidence through a confidence-aware multimodal fusion controller.

## Core Components

AeroAssist contains the following components:

- Vision classification using CLIP ViT-B/32
- Speech transcription using Whisper Base
- Semantic text processing using all-MiniLM-L6-v2
- Structured airport knowledge-base retrieval
- Confidence-aware multimodal fusion
- Conflict and uncertainty handling
- Passenger-facing response generation
- Streamlit user interface

The system supports:

- Text-only input
- Voice-only input
- Image-only input
- Image + text input
- Image + voice input

## Project Structure

```text
AeroAssist/
├── app.py
├── evaluate_vision_final.py
├── audio/          # Speech samples and noise-test audio
├── data/           # Knowledge base, query datasets and challenge data
├── images/         # Curated airport image dataset
├── results/        # Evaluation outputs, predictions and figures
├── src/            # Pipeline and additional evaluation source code
├── README.md
├── requirements.txt
└── .gitignore
```

## Data

### Visual Data

The curated airport image dataset is stored in `images/`. It contains 200 images across 10 airport-related classes:

- gate sign
- baggage claim
- security
- check-in
- lounge
- restaurant
- information
- transport
- accessibility
- family facility

Image metadata and dataset split information are stored in `data/image_manifest.csv`.

### Speech Data

Speech samples are stored in `audio/`.

The speech evaluation includes clean passenger queries together with controlled synthetic pink-noise variants used to examine transcription robustness under moderate and severe noise conditions.

### Text and Structured Data

Structured experimental data are stored in `data/`. These files include passenger queries, multimodal challenge scenarios and the fictional airport knowledge base used by the retrieval pipeline.

The airport knowledge base contains 25 structured records describing airport locations and passenger services.

## Setup

Python 3 is required.

Create a virtual environment from the project root:

```bash
python3 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg is also required for speech/audio processing.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Run AeroAssist

From the project root directory:

```bash
streamlit run app.py
```

The Streamlit interface provides access to the passenger-facing multimodal prototype.

Users can submit:

- typed passenger queries
- airport images
- recorded or uploaded speech
- combined image and text evidence
- combined image and voice evidence

The interface displays the resulting airport assistance response together with the system confidence state, detected evidence and safety information.

## Pipeline

### Text

Typed passenger queries are processed by the text pipeline for intent recognition, entity extraction and semantic knowledge-base retrieval.

### Speech

Audio is transcribed using Whisper Base. The resulting transcription is passed through the same text-processing and retrieval pipeline used for typed passenger queries.

### Vision

Uploaded airport images are processed using a frozen CLIP ViT-B/32 model. The vision pipeline predicts an airport visual category and produces a class score and prediction margin.

### Multimodal Fusion

Available text, speech and visual evidence is passed to the multimodal fusion controller.

The controller evaluates modality agreement, confidence and passenger context before selecting an appropriate response behaviour.

Possible outcomes include:

- normal knowledge-base retrieval
- caution
- conflict warning
- uncertainty or abstention
- human handover

## Knowledge Base

AeroAssist uses a structured fictional airport knowledge base for retrieval and testing.

The knowledge base contains 25 airport records. Records contain information including:

- service or location name
- category
- terminal
- zone
- floor
- directions
- walking time
- opening hours
- accessibility information
- related facilities
- criticality
- verification information

The knowledge base is static and fictional and should not be interpreted as live airport information.

## Evaluation

Evaluation outputs are stored in `results/`.

### Text Evaluation

Run:

```bash
python src/evaluate_final_text.py
```

This evaluates the text component using accuracy, precision, recall, F1 and confidence-sensitive behaviour.

### Vision Evaluation

Run:

```bash
python evaluate_vision_final.py
```

This evaluates the frozen CLIP ViT-B/32 vision pipeline on the final image test split using Top-1 accuracy, Top-3 accuracy, class-level performance and confidence-sensitive acceptance.

The curated airport images are stored in `images/`, while image metadata and dataset split information are stored in `data/image_manifest.csv`.

### Speech Evaluation

Speech samples and synthetic-noise variants are stored in `audio/`.

Speech evaluation measures Word Error Rate (WER) and downstream passenger-query performance under clean, moderate synthetic pink-noise and severe synthetic pink-noise conditions.

The speech pipeline and evaluation scripts are located in `src/`, with generated outputs stored in `results/`.

### Multimodal Fusion Evaluation

Run:

```bash
python src/evaluate_multimodal_challenges.py
```

The multimodal challenge set evaluates predefined single-modality and combined-modality conditions, including image-text and image-voice scenarios.

These scenarios test fusion behaviour such as modality agreement, conflicting evidence, uncertainty and human handover.

### End-to-End Evaluation

Run:

```bash
python src/evaluate_end_to_end.py
```

This evaluates the integrated AeroAssist pipeline across predefined passenger scenarios using text, speech and image inputs.

The end-to-end evaluation tests whether the complete pipeline produces the expected retrieval, conflict-handling and uncertainty behaviour.

## Confidence and Safety

AeroAssist uses three confidence states:

- confident
- caution
- uncertain

Confidence is used as a safety mechanism rather than simply as a displayed model score.

Low-confidence evidence can be withheld, conflicting modalities can produce a warning, and uncertain cases can abstain from providing unsupported passenger directions.

Critical passenger requests can also trigger human-assistance guidance.

## Reproducibility

The main project resources are organised as follows:

- `images/` — curated airport visual dataset
- `audio/` — clean and synthetic-noise speech samples
- `data/` — knowledge base, passenger queries, image manifest and challenge data
- `results/` — evaluation predictions, metrics and generated figures
- `src/` — text, vision, speech, fusion, response and evaluation code
- `app.py` — Streamlit passenger interface
- `evaluate_vision_final.py` — final vision evaluation script
- `requirements.txt` — Python dependencies

## Safety and Limitations

AeroAssist is a research prototype and has not been validated for real-world airport deployment.

The visual dataset is relatively small, speech testing uses a limited number of utterances from a single speaker, and the airport knowledge base is fictional and static.

Accent robustness, multilingual performance and large-scale real-world airport performance were not empirically evaluated.

The predefined multimodal and end-to-end scenarios demonstrate prototype behaviour on the tested cases and should not be interpreted as general real-world accuracy.

## Academic Project

AeroAssist was developed as an academic proof-of-concept multimodal AI system for airport passenger assistance.