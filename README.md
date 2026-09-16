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
├── audio/
├── data/
├── images/
├── results/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run AeroAssist

From the project root directory:

```bash
streamlit run app.py
```

The Streamlit interface provides access to the passenger-facing multimodal prototype.

## Evaluation

### Text Evaluation

Run:

```bash
python src/evaluate_final_text.py
```

This evaluates the text component using accuracy, precision, recall, F1 and a confusion matrix.

### Multimodal Fusion Evaluation

Run:

```bash
python src/evaluate_multimodal_challenges.py
```

The challenge set evaluates single and combined input conditions, including image-text and image-voice scenarios.

### End-to-End Evaluation

Run:

```bash
python src/evaluate_end_to_end.py
```

This evaluates the complete AeroAssist pipeline across predefined passenger scenarios.

## Knowledge Base

AeroAssist uses a structured fictional airport knowledge base for retrieval and testing. Records contain information including terminal, zone, floor, directions, walking time, accessibility information, keywords, related facilities, criticality and verification information.

The knowledge base is static and fictional and should not be interpreted as live airport information.

## Safety and Limitations

AeroAssist uses confidence states and multimodal agreement checks to reduce unsupported passenger guidance. Low-confidence visual evidence may be withheld, conflicting modalities can trigger warnings, and critical requests can be directed towards human assistance.

The prototype was evaluated using a limited research dataset and predefined scenarios. The results do not demonstrate real-world airport reliability.

## Reproducibility

Evaluation outputs are stored in `results/`, structured project data in `data/`, and implementation modules in `src/`.

The project was developed as an academic multimodal AI prototype.
