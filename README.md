<div align="center">

# 🎙️ PoetryDNA
### Neural-Linguistic Fusion Poetry Attribution System

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**PoetryDNA** identifies the author of any poem using a hybrid fusion of deep learning (DistilBERT) and classical machine learning (LightGBM + SBERT), with an interactive "Record Sleeve" UI that visually explains every step of the inference process.

> **Live Website:** [https://poetry-dna.vercel.app](https://poetry-dna.vercel.app)  
> **88.5% fusion accuracy** · 2,495 training stanzas · 6 poets · 3 models fused

</div>

---

## 📑 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Deep Learning Pipeline (DL Brain)](#deep-learning-pipeline)
3. [NLP Linguistic Pipeline (Linguistic Brain)](#nlp-linguistic-pipeline)
4. [Fusion Engine](#fusion-engine)
5. [Backend — FastAPI](#backend)
6. [Frontend — React + Vite](#frontend)
7. [Deployment](#deployment)
8. [Documentation & Study Guides](#documentation--study-guides)
9. [Setup & Running Locally](#setup--running-locally)
10. [Project Structure](#project-structure)
11. [Team](#team)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
│                    (poem / stanza text)                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    FastAPI Backend      │
              │    (Fusion Orchestrator)│
              └─────┬──────────┬────────┘
                    │          │
        ┌───────────▼──┐  ┌────▼──────────────────┐
        │  DL Brain    │  │  Linguistic Brain      │
        │  (Remote)    │  │  (Local)               │
        │              │  │                        │
        │ DistilBERT   │  │ LightGBM + SBERT       │
        │ HF Space     │  │ 32 hand-crafted feats  │
        │ MC Dropout   │  │ TF-IDF + Char n-gram   │
        └──────┬───────┘  └──────────┬─────────────┘
               │                     │
               └──────────┬──────────┘
                          │
              ┌───────────▼───────────┐
              │    Fusion Engine      │
              │  (weighted avg vote)  │
              │  DL: 60% · NLP: 40%  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   React Frontend      │
              │  "Record Sleeve" UI   │
              │  Plotly · Framer      │
              └───────────────────────┘
```

---

## Deep Learning Pipeline

> **Developed by:** Mehtab Singh  
> **Notebook:** `DL-Mehtab/PoetryDNA - DLClassifier.ipynb`  
> **Deployed on:** [Hugging Face Space](https://huggingface.co/spaces/mehtabsingh3711/poetrydna-dl)

### Model: DistilBERT Fine-tuned Classifier

DistilBERT (a 66M-parameter distilled variant of BERT) is fine-tuned for 6-class poetry attribution:

| Stage | Detail |
|-------|--------|
| **Base model** | `distilbert-base-uncased` |
| **Task** | Sequence classification (6 poets) |
| **Input** | Raw poem text, max 128 tokens |
| **Output** | Softmax probability over 6 classes |
| **Training** | Fine-tuned on 2,495 curated stanzas |
| **Accuracy** | ~84% standalone |

### Tokenization
DistilBERT uses WordPiece tokenization. Rare poetic words (e.g., *"thwart"*, *"firmament"*) are split into known subword fragments, preserving stylometric signals even for archaic vocabulary.

```
"Shall I compare" → ["shall", "i", "compare"] ✓
"thunderstruck"   → ["thunder", "##struck"]   (subword split)
```

### CLS Absorption
A special `[CLS]` token at position 0 attends to every other token across 6 transformer layers. Its final hidden state is a 768-dimensional summary vector of the entire verse, passed to a linear classifier head.

### MC Dropout (Uncertainty Quantification)
Instead of one forward pass, we run **10 stochastic passes** with dropout enabled at inference time (Monte Carlo Dropout). This produces:
- **`mean_probs`**: averaged softmax distribution (more reliable than a single pass)
- **`uncertainty`**: standard deviation of predictions across runs — high σ = the model is unsure

```python
# Simplified MC Dropout loop
enable_dropout(model)
probs_runs = [softmax(model(input)) for _ in range(10)]
mean_probs = torch.stack(probs_runs).mean(dim=0)
uncertainty = torch.stack(probs_runs).std(dim=0).mean()
```

### Attention Heatmap
The last attention layer's CLS-row is extracted and mapped back to input tokens, producing per-word attention scores — visualized as a glowing heatmap in the UI.

---

## NLP Linguistic Pipeline

> **Developed by:** Sakshi Verma  
> **Notebook:** `NLP-Sakshi/PoetryDNA - NLPClassifier.ipynb`

### Model: LightGBM Gradient Boosted Classifier

| Stage | Detail |
|-------|--------|
| **Algorithm** | LightGBM (Gradient Boosted Decision Trees) |
| **Features** | 232 total dimensions |
| **Hand-crafted** | 32 stylometric features across 5 categories |
| **TF-IDF** | SVD-reduced to 150 components |
| **Char n-gram** | SVD-reduced to 50 components |
| **Accuracy** | 81.36% standalone |

### 32 Stylometric Features (5 Categories)

#### 1. Prosody
Captures metrical patterns and rhythmic fingerprints:
- Average syllable count per line
- Syllabic variance (consistency of metre)
- Iambic stress ratio (weak-strong foot detection)
- Anapestic ratio, trochaic ratio

#### 2. Rhyme
Detects end-rhyme density and schemes:
- End-rhyme density (proportion of rhyming line-ends)
- ABAB, AABB, ABBA scheme scoring
- Internal rhyme ratio

#### 3. Vocabulary
Keyword density ratios for poet-specific lexicons:
- Archaic word ratio (`thee`, `thine`, `dost`, `hath`…)
- Nature word density
- Dark/mortality word density
- Divine/religious word density
- Sensory adjective density

#### 4. Structure
Line-level structural signals:
- Average line length (words)
- Line-length variance
- Enjambment ratio
- Repetition index
- Flesch readability score

#### 5. Part-of-Speech Density
Syntactic fingerprints via spaCy POS tagging:
- Noun, verb, adjective, adverb, proper-noun density
- Passive-voice ratio

### TF-IDF + Character N-gram
Beyond hand-crafted features, a TF-IDF vectorizer captures vocabulary distinctiveness. SVD (Latent Semantic Analysis) reduces 10,000+ terms to 150 principal components. A separate character-level n-gram model (SVD→50 dims) captures spelling and morphological patterns unique to each poet's era.

### SBERT Semantic Retrieval
Sentence-BERT (`all-MiniLM-L6-v2`) encodes both the query poem and every corpus line into 384-dimensional semantic vectors. Cosine similarity retrieval finds the nearest corpus lines, visualized in an interactive 3D PCA-projected embedding space.

---

## Fusion Engine

The local backend combines both model outputs via **weighted probability averaging**:

```python
FUSION_CONFIG = {
    "dl_weight":  0.60,   # DistilBERT (deep semantic)
    "nlp_weight": 0.40,   # LightGBM  (linguistic fingerprint)
}

fused_probs = {
    poet: dl_weight * dl_prob[poet] + nlp_weight * nlp_prob[poet]
    for poet in poets
}
predicted_poet = max(fused_probs, key=fused_probs.get)
```

If the HF Space is unavailable, the engine gracefully falls back to NLP-only prediction while flagging the DL brain as offline.

---

## Backend

> **Tech:** FastAPI · Python 3.11 · LightGBM · SBERT · scikit-learn

The local backend is a pure **orchestration layer** — it holds the NLP models and coordinates with the remote DL inference service.

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Full fusion prediction |
| `POST` | `/explain` | Detailed per-model explanation |
| `POST` | `/explain-nlp` | NLP features only |
| `GET` | `/corpus-embeddings` | SBERT 3D PCA points for visualization |
| `GET` | `/dataset-stats` | Training dataset statistics |
| `GET` | `/health` | Liveness check |

### AI-Assisted Development
The backend fusion logic, API schema design, robust fallback handling (graceful degradation when HF Space is offline), and the `run_mc_dropout` pipeline were developed with **AI pair-programming assistance** (Google Gemini / Antigravity). Specifically:
- Elimination of `numpy` dependency from the inference path (replaced with pure PyTorch tensor ops)
- Defensive schema normalization for mixed API response formats
- `IndexError`-proof probability vector reconstruction from `top_poets` partial responses

---

## Frontend

> **Tech:** React 18 · Vite · Framer Motion · Plotly.js · TailwindCSS classes · CSS variables

The "Record Sleeve" UI is a dark-mode, glassmorphism-style single-page app with 6 dedicated visualization pages.

### Pages

| Page | Description |
|------|-------------|
| **Home** | Landing with animated particle field |
| **Identify** | Submit poem → full fusion result with confidence bars, heatmap, kindred lines |
| **DistilBERT** | Step-by-step: tokenization → embeddings → transformer layers → CLS → MC dropout |
| **SBERT** | Dual encoding → 3D embedding space → cosine similarity explorer → retrieved lines |
| **Linguistic Brain** | Top-5 stylometric features per category → TF-IDF distinctive terms |
| **About** | Architecture diagram · team · dataset stats |

### Key Components

| Component | Role |
|-----------|------|
| `AttentionHeatmap` | Token-level attention score visualization |
| `ConfidenceBars` | Animated probability bars (poet ranking) |
| `EmbeddingPlot3D` | Interactive 3D Plotly scatter (SBERT space) |
| `HelixMeter` | Combined confidence + uncertainty gauge |
| `SimilarLines` | SBERT nearest-neighbor corpus cards |
| `TransformerLayer` | Per-layer attention head visualization |
| `useScrollReveal` | IntersectionObserver + MutationObserver scroll animations |

### AI-Assisted Development
The entire frontend was built with **AI pair-programming assistance** (Google Gemini / Antigravity), covering:
- "Record Sleeve" design system (CSS variables, glassmorphism panels, gold/crimson palette)
- All 6 page layouts and component architecture
- GSAP → native IntersectionObserver migration to fix invisible-results bug
- MutationObserver integration for dynamically-rendered result sections
- Defensive rendering for DL-offline states (graceful fallback UI)
- React Router v7 future flags

---

## Deployment

### HF Space (DL Brain)
The DistilBERT inference runs on a **Hugging Face Docker Space**:
- `DL-Mehtab/hf_space/Dockerfile` — CPU-only PyTorch image
- `DL-Mehtab/hf_space/app.py` — FastAPI server with MC Dropout
- Endpoint: `https://mehtabsingh3711-poetrydna-dl.hf.space`

### Local Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # development
npm run build      # production bundle
```

---

## Documentation & Study Guides

We have provided detailed guides to help understand the project and the underlying NLP concepts:

- **[Complete NLP Pipeline Guide](docs/PoetryDNA_NLP_Complete_Guide.md)**: A deep dive into every feature, function, and decision made in the NLP pipeline.
- **[Interactive NLP Study Guide](docs/nlp_study_guide.html)**: A standalone interactive guide covering NLP from zero to hero (Prerequisites, History, Core Tasks, and more).

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git LFS (`git lfs install`)

### 1. Clone & install LFS
```bash
git clone https://github.com/MehtabSingh3711/PoetryDNA.git
cd PoetryDNA
git lfs pull          # downloads model weights
```

### 2. Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 4. Environment Variables
Create `backend/.env` (never commit this file):
```env
HF_SPACE_URL=https://mehtabsingh3711-poetrydna-dl.hf.space
```

---

## Project Structure

```
PoetryDNA/
├── DL-Mehtab/                      # Deep Learning pipeline
│   ├── PoetryDNA - DLClassifier.ipynb  # Training notebook (Git LFS)
│   ├── PoetryDNA - DLClassifier.pdf    # Report (Git LFS)
│   ├── export_classes.py           # Exports classes.json from model
│   ├── hf_space/                   # Hugging Face Docker Space
│   │   ├── app.py                  # FastAPI inference server
│   │   ├── Dockerfile              # CPU PyTorch image
│   │   ├── requirements.txt        # HF dependencies
│   │   ├── DistilBERT.pt           # Model weights (Git LFS)
│   │   └── classes.json            # Poet class list
│   └── results/                    # Training metrics & plots
│
├── NLP-Sakshi/                     # NLP + Classical ML pipeline
│   ├── PoetryDNA - NLPClassifier.ipynb # Training notebook (Git LFS)
│   ├── PoetryDNA - NLPClassifier.pdf   # Report (Git LFS)
│   └── results/                    # Confusion matrices, metrics
│
├── backend/                        # FastAPI fusion orchestrator
│   ├── main.py                     # All endpoints + fusion logic
│   ├── nlp_features.py             # 32 stylometric feature extractors
│   ├── requirements.txt            # Python dependencies
│   └── models/                     # Serialized NLP models (Git LFS)
│       ├── lgbm_model.pkl
│       ├── sbert_model/
│       ├── svd_tfidf.pkl
│       └── svd_char.pkl
│
├── frontend/                       # React + Vite UI
│   ├── src/
│   │   ├── pages/                  # 6 visualization pages
│   │   ├── components/             # Reusable UI components
│   │   ├── hooks/                  # useScrollReveal
│   │   └── lib/                    # API client, poet metadata
│   ├── index.html
│   └── package.json
│
├── dataset/                        # Curated poetry corpus
│   └── corpus.json                 # Processed corpus (SBERT retrieval)
│
├── .gitattributes                  # Git LFS rules
├── .gitignore
└── README.md
```

---

## Team

| Name | Role |
|------|------|
| **Mehtab Singh** | DistilBERT fine-tuning, MC Dropout inference, FastAPI backend, React frontend, full-stack integration, AI-assisted development |
| **Sakshi Verma** | LightGBM classifier, 32 stylometric features, SBERT retrieval, TF-IDF/char n-gram pipeline, fusion calibration |

---

## License

MIT © 2026 PoetryDNA Team
