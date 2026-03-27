# TextSentinel — AI vs Human Text Detector

A fully offline machine learning web app that classifies text as **Human-written** or **AI-generated** using an ensemble of four models.

## Project Structure

```
ai-text-detector/
├── data/
│   ├── generate_dataset.py   # Generates 22,500-sample dataset
│   └── dataset.csv           # Pre-generated (do not re-run unless needed)
├── model/                    # Saved model artifacts (created by train_model.py)
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   └── index.html
├── app.py                    # Flask backend
├── train_model.py            # Model training script
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

**Step 1 — Train the models** (only needed once, or after regenerating the dataset):
```bash
python train_model.py
```

**Step 2 — Start the web app:**
```bash
python app.py
```

**Step 3 — Open in browser:**
```
http://localhost:5000
```

## Dataset

- **22,500 samples** — balanced
- Human samples: casual conversation, slang, emotional text, storytelling, informal grammar
- AI samples: formal prose, technical explanations, academic tone, structured writing
- Generated via `data/generate_dataset.py` using templates + controlled augmentation

## Models

| Model | Description |
|---|---|
| Logistic Regression (LR) | Fast linear classifier on TF-IDF + hand-crafted features |
| Naive Bayes (NB) | Probabilistic baseline |
| Random Forest (RF) | Ensemble of decision trees |
| SVM | Support Vector Machine with RBF kernel |
| **Ensemble** | Soft-voting combination of LR + RF + SVM (primary) |

### Features used
- TF-IDF (3,000 tokens)
- Slang word count (`lol`, `ngl`, `idk`, `bruh`)
- Exclamation mark count
- Question mark count
- Average word length
- Lexical diversity (unique words / total words)

## Decision Thresholds

| Confidence | Decision |
|---|---|
| ≥ 75% | Acceptable |
| 55–74% | Needs Review |
| < 55% | Likely AI |

## API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the UI |
| `/predict` | POST | Classifies text; body: `{"text": "..."}` |
| `/stats` | GET | Returns per-model accuracy from last training run |
