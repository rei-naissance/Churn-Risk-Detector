---
title: Churn Risk Detector
emoji: 📦
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: mit
---

# 📦 Logistics Churn Risk Detector

A scikit-learn powered model that predicts **customer churn risk** from free-text shipping complaints. Built with a TF-IDF + Random Forest pipeline and a custom keyword-boost post-processor tuned for the logistics domain.

## Features

| Feature | Detail |
|---|---|
| **2 100+ training samples** | Hand-curated seed data + external Hugging Face datasets |
| **TF-IDF Vectorizer** | `ngram_range=(1,2)`, 5 000 features — captures phrases like *"not delivered"* |
| **Random Forest** | 300 estimators, balanced class weights |
| **Keyword Boost** | Post-processing adds risk points for *lost*, *refund*, *damaged*, etc. |
| **External Data Pipeline** | Auto-downloads & caches real complaint datasets on first run |
| **Clean API** | Single function `analyze_complaint(text)` → `dict` |
| **Typed & Logged** | Full type hints + Python `logging` throughout |

## Data Sources

| Source | Samples | What we use |
|---|---|---|
| **Hand-curated seed** | 120 | Trustpilot-inspired FedEx/UPS/DHL complaints (60 pos + 60 neg) |
| **[hblim/customer-complaints](https://huggingface.co/datasets/hblim/customer-complaints)** | ~1 260 | `delivery` → churn, `billing`/`product` → retained |
| **[aciborowska/customers-complaints](https://huggingface.co/datasets/aciborowska/customers-complaints)** | ~730 | CFPB narratives keyword-filtered for logistics themes |
| **Total** | **~2 110** | Merged automatically via `data_loader.py` |

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo (downloads external data on first run)
python churn_model.py
```

## API Usage

```python
from churn_model import analyze_complaint

result = analyze_complaint("Shipment lost in transit, I want a refund")
print(result)
# {
#     "input_text": "Shipment lost in transit, I want a refund",
#     "sentiment_score": 74.05,
#     "detected_keywords": ["lost", "refund"],
#     "keyword_boost_applied": 9.00,
#     "final_churn_risk_pct": 83.05,
#     "risk_level": "Critical"
# }
```

### Return Dictionary

| Key | Type | Description |
|---|---|---|
| `input_text` | `str` | The original complaint |
| `sentiment_score` | `float` | Raw model churn probability (0–100 %) |
| `detected_keywords` | `list[str]` | High-priority logistics keywords found |
| `keyword_boost_applied` | `float` | Total boost in percentage points |
| `final_churn_risk_pct` | `float` | Final risk score clamped to 0–100 |
| `risk_level` | `str` | `Low` / `Medium` / `High` / `Critical` |

## How It Works

1. **Multi-source data pipeline** — 120 hand-curated seed samples merged with ~2 000 external complaints downloaded from Hugging Face. The `data_loader.py` module handles downloading, logistics keyword filtering, label mapping, and caching to `data/external_complaints.npz`.
2. **Pipeline** — `TfidfVectorizer` (5 000 features, uni+bigrams) → `RandomForestClassifier` (300 trees, balanced weights), wrapped in a scikit-learn `Pipeline` for one-call fit/predict.
3. **Post-processing** — Each detected keyword from a list of 40+ high-priority logistics terms (e.g., *lost*, *damaged*, *refund*, *crushed*, *wrong address*, *never arrived*) adds **+4.5 %** to the raw score, capping at 100 %.
4. **Risk bands** — `≥80 %` Critical · `≥55 %` High · `≥30 %` Medium · `<30 %` Low.

## Project Structure

```
Churn-Risk-Detector/
├── app.py                # Gradio web UI (Hugging Face Spaces)
├── churn_model.py        # Model, pipeline, and analyze_complaint()
├── data_loader.py        # External dataset download, filter & cache
├── data/                 # Cached external data (auto-created)
│   └── external_complaints.npz
├── test_churn_model.py   # 35 unit tests (pytest)
├── requirements.txt      # Python dependencies
└── README.md
```

## Running Locally

```bash
# Launch the Gradio web UI
python app.py
# Opens at http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free)
2. Push this repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/Churn-Risk-Detector
   git push hf main
   ```
3. The Space will auto-install from `requirements.txt` and launch `app.py`.

## Running Tests

```bash
pytest test_churn_model.py -v
```
