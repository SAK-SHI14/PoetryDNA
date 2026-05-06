from __future__ import annotations

import warnings
import os

# Silence noisy library warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import re
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from nlp_features import extract_all_features


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
DL_MODEL_DIR = WORKSPACE_DIR / "DL-Mehtab" / "results"
NLP_MODEL_DIR = WORKSPACE_DIR / "NLP-Sakshi" / "results"
MAX_LENGTH = 128
MC_DROPOUT_SAMPLES = 50
SIMILAR_LINES_COUNT = 3
NEAREST_NEIGHBORS_COUNT = 8
UMAP_RANDOM_STATE = 42

POET_METADATA: dict[str, dict[str, Any]] = {
    "Shakespeare": {
        "era": "Elizabethan · 1564–1616",
        "color": "#1E3A8A",
        "style_tags": [
            "Dramatic rhetoric",
            "Volta-rich sonnet turns",
            "Psychological intensity",
            "Elevated metaphor",
        ],
    },
    "Keats": {
        "era": "Romantic · 1795–1821",
        "color": "#065F46",
        "style_tags": [
            "Sensuous imagery",
            "Lush musicality",
            "Mythic atmosphere",
            "Meditative longing",
        ],
    },
    "Milton": {
        "era": "Restoration · 1608–1674",
        "color": "#4C1D95",
        "style_tags": [
            "Epic cadence",
            "Biblical grandeur",
            "Elevated syntax",
            "Moral cosmology",
        ],
    },
    "Tennyson": {
        "era": "Victorian · 1809–1892",
        "color": "#92400E",
        "style_tags": [
            "Melancholic tone",
            "Musical rhythm",
            "Nature imagery",
            "Elegiac mood",
        ],
    },
    "Coleridge": {
        "era": "Romantic · 1772–1834",
        "color": "#134E4A",
        "style_tags": [
            "Dreamlike surrealism",
            "Mystic symbolism",
            "Conversational lyricism",
            "Philosophical wonder",
        ],
    },
    "Wordsworth": {
        "era": "Romantic · 1770–1850",
        "color": "#881337",
        "style_tags": [
            "Pastoral reflection",
            "Plainspoken diction",
            "Moral introspection",
            "Landscape memory",
        ],
    },
}


@dataclass
class RuntimeState:
    hf_url: str
    tokenizer: DistilBertTokenizer
    label_encoder: Any
    sbert: SentenceTransformer
    corpus_embeddings: np.ndarray
    corpus_texts: list[str]
    corpus_labels: list[str]
    dataset: pd.DataFrame
    dataset_stats: dict[str, Any]
    reducer_2d: Any
    reducer_3d: Any
    corpus_projection_2d: np.ndarray
    corpus_projection_3d: np.ndarray
    corpus_points: list[dict[str, Any]]
    device: torch.device
    # NLP pipeline artifacts
    nlp_model: Any
    nlp_label_encoder: Any
    nlp_scaler: Any
    nlp_tfidf: Any
    nlp_svd150: Any
    nlp_char_tfidf: Any
    nlp_char_svd: Any
    fusion_config: dict[str, Any]


runtime_state: RuntimeState | None = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="One or more lines of verse.")


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, description="One or more lines of verse.")
    include: list[str] = Field(default_factory=lambda: ["tokens", "distilbert", "sbert", "nlp"])


def resolve_artifact(candidates: Iterable[str]) -> Path:
    search_roots = [MODEL_DIR, DL_MODEL_DIR, NLP_MODEL_DIR, WORKSPACE_DIR]
    for candidate in candidates:
        for root in search_roots:
            path = root / candidate
            if path.exists():
                return path
    raise FileNotFoundError(f"Unable to locate any of: {', '.join(candidates)}")


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_label_encoder(path: Path) -> Any:
    try:
        encoder = safe_torch_load(path)
        if hasattr(encoder, "classes_"):
            return encoder
    except Exception:
        pass

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in ("label_encoder/data.pkl", "data.pkl"):
                if member in archive.namelist():
                    return pickle.loads(archive.read(member))

    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Please provide at least one non-empty line of poetry.")
    return " ".join(lines)


def prepare_inputs(text: str, tokenizer: DistilBertTokenizer, *, padding: str | bool = "max_length") -> tuple[str, dict[str, torch.Tensor]]:
    formatted = normalize_text(text)
    encoded = tokenizer(
        formatted,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=padding,
        return_tensors="pt",
    )
    return formatted, encoded


def enable_dropout(module: nn.Module) -> None:
    if isinstance(module, nn.Dropout):
        module.train()


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper - lower < 1e-9:
        return [1.0 for _ in values]
    return [float((value - lower) / (upper - lower)) for value in values]


def merge_wordpiece_attention(tokens: list[str], scores: list[float]) -> list[dict[str, float | str]]:
    merged_words: list[str] = []
    merged_scores: list[list[float]] = []

    for token, score in zip(tokens, scores, strict=False):
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        cleaned = token.replace("Ġ", "")
        if cleaned.startswith("##") and merged_words:
            merged_words[-1] += cleaned[2:]
            merged_scores[-1].append(float(score))
        else:
            merged_words.append(cleaned)
            merged_scores.append([float(score)])

    averaged = [float(np.mean(chunk)) for chunk in merged_scores]
    normalized = minmax_normalize(averaged)
    return [{"word": word, "score": score} for word, score in zip(merged_words, normalized, strict=False)]


def compute_dataset_stats(dataset: pd.DataFrame) -> dict[str, Any]:
    working = dataset.copy()
    working["char_length"] = working["text"].str.len()
    working["word_count"] = working["text"].str.split().str.len()

    poet_stats = []
    for poet, group in working.groupby("label"):
        tokens = re.findall(r"\b[\w']+\b", " ".join(group["text"]).lower())
        poet_stats.append(
            {
                "poet": poet,
                "count": int(len(group)),
                "avg_char_length": round(float(group["char_length"].mean()), 2),
                "avg_word_count": round(float(group["word_count"].mean()), 2),
                "vocab_size": int(len(set(tokens))),
            }
        )

    all_tokens = re.findall(r"\b[\w']+\b", " ".join(working["text"]).lower())
    return {
        "total_samples": int(len(working)),
        "avg_char_length": round(float(working["char_length"].mean()), 2),
        "avg_word_count": round(float(working["word_count"].mean()), 2),
        "vocab_size": int(len(set(all_tokens))),
        "per_poet": poet_stats,
    }


def build_corpus_points(
    texts: list[str],
    labels: list[str],
    projection_2d: np.ndarray,
    projection_3d: np.ndarray,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for idx, (text, poet) in enumerate(zip(texts, labels, strict=False)):
        points.append(
            {
                "text": text,
                "poet": poet,
                "x": float(projection_3d[idx][0]),
                "y": float(projection_3d[idx][1]),
                "z": float(projection_3d[idx][2]),
                "x2d": float(projection_2d[idx][0]),
                "y2d": float(projection_2d[idx][1]),
            }
        )
    return points


def run_mc_dropout(state: RuntimeState, text: str) -> dict[str, Any]:
    """Calls the remote Hugging Face Space for DL prediction."""
    try:
        import requests
        response = requests.post(state.hf_url, json={"text": text, "runs": 10}, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Format the remote response to match what the local app expects
        return {
            "predicted_poet": data["predicted_poet"],
            "confidence": data["confidence"],
            "uncertainty": data.get("uncertainty", 0.0),
            "top_poets": data["top_poets"],
            "passes": data.get("mc_runs", []),
            "mean_probs": data.get("mean_probs", []),
            "attention": data.get("attention", [])
        }
    except Exception as e:
        print(f"DL Remote Error: {e}")
        return {
            "predicted_poet": "Unknown",
            "confidence": 0.0,
            "uncertainty": 1.0,
            "top_poets": [],
            "passes": [],
            "mean_probs": [],
            "attention": []
        }


def retrieve_similar_lines(state: RuntimeState, query_text: str, predicted_poet: str, *, top_k: int = SIMILAR_LINES_COUNT) -> list[dict[str, Any]]:
    query_embedding = state.sbert.encode([query_text], convert_to_numpy=True)
    poet_indices = [index for index, label in enumerate(state.corpus_labels) if label == predicted_poet]
    poet_embeddings = state.corpus_embeddings[poet_indices]
    poet_texts = [state.corpus_texts[index] for index in poet_indices]
    similarities = cosine_similarity(query_embedding, poet_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for local_index in top_indices:
        results.append(
            {
                "text": poet_texts[int(local_index)],
                "score": round(float(similarities[int(local_index)]), 3),
                "poet": predicted_poet,
            }
        )
    return results


def build_sbert_analysis(
    state: RuntimeState,
    query_text: str,
    predicted_poet: str,
) -> dict[str, Any]:
    query_embedding = state.sbert.encode([query_text], convert_to_numpy=True)
    query_projection_2d = state.reducer_2d.transform(query_embedding)[0]
    query_projection_3d = state.reducer_3d.transform(query_embedding)[0]
    all_scores = cosine_similarity(query_embedding, state.corpus_embeddings)[0]
    nearest_indices = all_scores.argsort()[::-1][:NEAREST_NEIGHBORS_COUNT]

    nearest_neighbors = []
    for index in nearest_indices:
        nearest_neighbors.append(
            {
                "text": state.corpus_texts[int(index)],
                "poet": state.corpus_labels[int(index)],
                "score": round(float(all_scores[int(index)]), 3),
                "x": float(state.corpus_projection_3d[int(index)][0]),
                "y": float(state.corpus_projection_3d[int(index)][1]),
                "z": float(state.corpus_projection_3d[int(index)][2]),
                "x2d": float(state.corpus_projection_2d[int(index)][0]),
                "y2d": float(state.corpus_projection_2d[int(index)][1]),
            }
        )

    filtered_neighbors = [neighbor for neighbor in nearest_neighbors if neighbor["poet"] == predicted_poet][:SIMILAR_LINES_COUNT]

    return {
        "predicted_poet": predicted_poet,
        "query_embedding": query_embedding[0].round(6).tolist(),
        "query_projection_2d": {
            "x": float(query_projection_2d[0]),
            "y": float(query_projection_2d[1]),
        },
        "query_projection_3d": {
            "x": float(query_projection_3d[0]),
            "y": float(query_projection_3d[1]),
            "z": float(query_projection_3d[2]),
        },
        "nearest_neighbors": nearest_neighbors,
        "filtered_neighbors": filtered_neighbors,
    }


def run_nlp_prediction(state: RuntimeState, text: str) -> dict[str, Any]:
    """Run the NLP linguistic brain: hand-crafted features + TF-IDF → LightGBM."""
    # 1) Extract 32 hand-crafted features
    feats = extract_all_features(text)
    feat_names = sorted(feats.keys())
    handcrafted = np.array([[feats[k] for k in feat_names]])

    # 2) TF-IDF word n-grams → SVD-150
    tfidf_matrix = state.nlp_tfidf.transform([text])
    tfidf_reduced = state.nlp_svd150.transform(tfidf_matrix)

    # 3) Char n-grams → SVD-50
    char_matrix = state.nlp_char_tfidf.transform([text])
    char_reduced = state.nlp_char_svd.transform(char_matrix)

    # 4) Concatenate all features → scale → predict
    X_combined = np.hstack([handcrafted, tfidf_reduced, char_reduced])
    X_scaled = state.nlp_scaler.transform(X_combined)

    probs = state.nlp_model.predict_proba(X_scaled)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_poet = str(state.nlp_label_encoder.classes_[predicted_idx])

    # Build top-3
    top_indices = probs.argsort()[::-1][:3]
    top_poets = [
        {"poet": str(state.nlp_label_encoder.classes_[i]), "prob": round(float(probs[i] * 100), 2)}
        for i in top_indices
    ]

    # Feature highlights (top-5 hand-crafted features by absolute value)
    feat_highlights = sorted(feats.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

    return {
        "predicted_poet": predicted_poet,
        "confidence": round(float(probs[predicted_idx] * 100), 2),
        "top_poets": top_poets,
        "probs": probs,
        "feature_highlights": [{"name": k, "value": round(float(v), 4)} for k, v in feat_highlights],
        "all_features": {k: round(float(v), 4) for k, v in feats.items()},
    }


def run_fusion(
    state: RuntimeState,
    dl_result: dict[str, Any],
    nlp_result: dict[str, Any],
) -> dict[str, Any]:
    """Fuse DL + NLP predictions using fusion_config weights."""
    config = state.fusion_config
    dl_weight = float(config.get("dl_weight", 0.6))
    nlp_weight = float(config.get("nlp_weight", 0.4))

    nlp_probs = nlp_result["probs"]

    # Align class orders
    dl_classes = list(state.label_encoder.classes_)
    nlp_classes = list(state.nlp_label_encoder.classes_)

    # ── Build DL probability vector ──────────────────────────────────────
    # Prefer mean_probs (full vector). Fall back to top_poets sparse dict
    # if mean_probs is missing or the wrong length (e.g. old HF Space).
    raw_mean_probs = dl_result.get("mean_probs", [])
    if isinstance(raw_mean_probs, list) and len(raw_mean_probs) == len(dl_classes):
        dl_prob_vec = np.array(raw_mean_probs, dtype=float)
    else:
        # Reconstruct from top_poets sparse data
        dl_prob_vec = np.zeros(len(dl_classes))
        for entry in dl_result.get("top_poets", []):
            poet = entry.get("poet", "")
            prob = float(entry.get("probability", 0.0))
            if poet in dl_classes:
                dl_prob_vec[dl_classes.index(poet)] = prob
        # If still all zeros, give full weight to predicted poet
        if dl_prob_vec.sum() < 1e-9:
            predicted = dl_result.get("predicted_poet", "")
            if predicted in dl_classes:
                dl_prob_vec[dl_classes.index(predicted)] = 1.0

    # ── Fuse ─────────────────────────────────────────────────────────────
    fused_probs = np.zeros(len(dl_classes))
    for i, poet in enumerate(dl_classes):
        dl_p = float(dl_prob_vec[i])
        nlp_idx = nlp_classes.index(poet) if poet in nlp_classes else -1
        nlp_p = float(nlp_probs[nlp_idx]) if nlp_idx >= 0 else 0.0
        fused_probs[i] = dl_weight * dl_p + nlp_weight * nlp_p

    # Normalize
    total = fused_probs.sum()
    if total > 0:
        fused_probs = fused_probs / total

    predicted_idx = int(np.argmax(fused_probs))
    predicted_poet = dl_classes[predicted_idx]

    top_indices = fused_probs.argsort()[::-1][:3]
    top_poets = [
        {"poet": dl_classes[int(i)], "prob": round(float(fused_probs[int(i)] * 100), 2)}
        for i in top_indices
    ]

    return {
        "predicted_poet": predicted_poet,
        "confidence": round(float(fused_probs[predicted_idx] * 100), 2),
        "top_poets": top_poets,
        "dl_weight": dl_weight,
        "nlp_weight": nlp_weight,
    }



def create_runtime_state() -> RuntimeState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_artifact(["DistilBERT.pt", "best_model.pt"])
    label_encoder_path = resolve_artifact(["label_encoder.pt"])
    corpus_embeddings_path = resolve_artifact(["corpus_embeddings.npy"])
    corpus_texts_path = resolve_artifact(["corpus_texts.pkl"])
    corpus_labels_path = resolve_artifact(["corpus_labels.pkl"])
    dataset_path = resolve_artifact(["dataset/poetry_dataset.csv", "poetry_dataset (2).csv"])

    label_encoder = load_label_encoder(label_encoder_path)

    # Tokenizer is light and needed for UI visualizations
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # ── DL REMOTE CONFIG ────────────────────────────────
    # We now offload the heavy DistilBERT work to Hugging Face
    hf_url = "https://mehtabsingh3711-poetrydna-dl.hf.space/predict"
    print(f"DL Engine: Remote (HF Space) 🌐")
    # ────────────────────────────────────────────────────


    with corpus_texts_path.open("rb") as handle:
        corpus_texts = list(pickle.load(handle))
    with corpus_labels_path.open("rb") as handle:
        corpus_labels = list(pickle.load(handle))

    corpus_embeddings = np.load(corpus_embeddings_path)
    dataset = pd.read_csv(dataset_path)
    dataset_stats = compute_dataset_stats(dataset)

    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.08,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
        transform_seed=UMAP_RANDOM_STATE,
    )
    reducer_3d = umap.UMAP(
        n_components=3,
        n_neighbors=25,
        min_dist=0.08,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
        transform_seed=UMAP_RANDOM_STATE,
    )
    corpus_projection_2d = reducer_2d.fit_transform(corpus_embeddings)
    corpus_projection_3d = reducer_3d.fit_transform(corpus_embeddings)
    corpus_points = build_corpus_points(corpus_texts, corpus_labels, corpus_projection_2d, corpus_projection_3d)

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load NLP pipeline artifacts ──────────────────────────────────────
    nlp_model = joblib.load(NLP_MODEL_DIR / "nlp_model.joblib")
    nlp_label_encoder = joblib.load(NLP_MODEL_DIR / "le_nlp.joblib")
    nlp_scaler = joblib.load(NLP_MODEL_DIR / "scaler.joblib")
    nlp_tfidf = joblib.load(NLP_MODEL_DIR / "tfidf_vectorizer.joblib")
    nlp_svd150 = joblib.load(NLP_MODEL_DIR / "svd150.joblib")
    nlp_char_tfidf = joblib.load(NLP_MODEL_DIR / "char_tfidf.joblib")
    nlp_char_svd = joblib.load(NLP_MODEL_DIR / "char_svd.joblib")
    fusion_config = joblib.load(NLP_MODEL_DIR / "fusion_config.joblib")
    # If fusion_config is not a dict, create sensible defaults
    if not isinstance(fusion_config, dict):
        fusion_config = {"dl_weight": 0.6, "nlp_weight": 0.4}

    return RuntimeState(
        hf_url=hf_url,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        sbert=sbert,
        corpus_embeddings=corpus_embeddings,
        corpus_texts=corpus_texts,
        corpus_labels=corpus_labels,
        dataset=dataset,
        dataset_stats=dataset_stats,
        reducer_2d=reducer_2d,
        reducer_3d=reducer_3d,
        corpus_projection_2d=corpus_projection_2d,
        corpus_projection_3d=corpus_projection_3d,
        corpus_points=corpus_points,
        device=device,
        nlp_model=nlp_model,
        nlp_label_encoder=nlp_label_encoder,
        nlp_scaler=nlp_scaler,
        nlp_tfidf=nlp_tfidf,
        nlp_svd150=nlp_svd150,
        nlp_char_tfidf=nlp_char_tfidf,
        nlp_char_svd=nlp_char_svd,
        fusion_config=fusion_config,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime_state
    runtime_state = create_runtime_state()
    try:
        yield
    finally:
        runtime_state = None


app = FastAPI(
    title="PoetryDNA API",
    version="1.0.0",
    description="AI poet authorship attribution and neural explanation backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_state() -> RuntimeState:
    if runtime_state is None:
        raise HTTPException(status_code=503, detail="Model runtime is not ready yet.")
    return runtime_state


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    state = get_state()
    formatted, _ = prepare_inputs(request.text, state.tokenizer)

    # DL branch (Hugging Face Remote)
    dropout_result = run_mc_dropout(state, request.text)
    attention_data = dropout_result.get("attention", [])

    # NLP branch (LightGBM + hand-crafted features)
    nlp_result = run_nlp_prediction(state, formatted)

    # Fusion
    fusion_result = run_fusion(state, dropout_result, nlp_result)

    # Similar lines (using fusion prediction)
    similar_lines = retrieve_similar_lines(state, formatted, fusion_result["predicted_poet"])

    return {
        "predicted_poet": fusion_result["predicted_poet"],
        "confidence": fusion_result["confidence"],
        "uncertainty": dropout_result["uncertainty"],
        "similar_lines": similar_lines,
        "attention": attention_data,
        "fusion_breakdown": {
            "dl_prediction": dropout_result["predicted_poet"],
            "dl_confidence": dropout_result["confidence"],
            "dl_uncertainty": dropout_result["uncertainty"],
            "dl_top_poets": dropout_result["top_poets"],
            "nlp_prediction": nlp_result["predicted_poet"],
            "nlp_confidence": nlp_result["confidence"],
            "nlp_top_poets": nlp_result["top_poets"],
            "nlp_feature_highlights": nlp_result["feature_highlights"],
            "fusion_prediction": fusion_result["predicted_poet"],
            "fusion_confidence": fusion_result["confidence"],
            "fusion_top_poets": fusion_result["top_poets"],
            "dl_weight": fusion_result["dl_weight"],
            "nlp_weight": fusion_result["nlp_weight"],
        },
    }


@app.post("/explain")
def explain(request: ExplainRequest) -> dict[str, Any]:
    state = get_state()
    include = {entry.lower() for entry in request.include}
    unknown = include.difference({"tokens", "distilbert", "sbert", "nlp"})
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown explain sections requested: {', '.join(sorted(unknown))}")

    formatted, encoded = prepare_inputs(request.text, state.tokenizer)
    padded_ids = encoded["input_ids"].to(state.device)
    padded_mask = encoded["attention_mask"].to(state.device)

    variable_text, variable_encoded = prepare_inputs(request.text, state.tokenizer, padding=False)
    variable_ids = variable_encoded["input_ids"].to(state.device)
    variable_mask = variable_encoded["attention_mask"].to(state.device)

    # DL branch (Hugging Face Remote)
    dropout_result = run_mc_dropout(state, request.text)
    attention_data = dropout_result.get("attention", [])

    payload: dict[str, Any] = {
        "formatted_text": formatted,
        "predicted_poet": dropout_result["predicted_poet"],
    }

    if "tokens" in include:
        payload["tokens"] = {
            "raw_tokens": [item["word"] for item in attention_data],
            "attention": attention_data,
        }

    if "distilbert" in include:
        pass_predictions = [
            {
                "run": index + 1,
                "probs": [round(float(prob * 100), 3) for prob in probs],
                "predicted_poet": str(state.label_encoder.classes_[int(np.argmax(probs))]),
            }
            for index, probs in enumerate(dropout_result["passes"])
        ]
        payload["distilbert"] = {
            "classes": [str(label) for label in state.label_encoder.classes_],
            "top_poets": dropout_result["top_poets"],
            "mc_dropout_runs": pass_predictions,
            "cls_summary": {
                "predicted_poet": dropout_result["predicted_poet"],
                "confidence": dropout_result["confidence"],
                "uncertainty": dropout_result["uncertainty"],
                "era": POET_METADATA.get(dropout_result["predicted_poet"], {}).get("era"),
            },
        }

    if "sbert" in include:
        payload["sbert"] = build_sbert_analysis(state, variable_text, dropout_result["predicted_poet"])

    if "nlp" in include:
        nlp_result = run_nlp_prediction(state, formatted)
        all_features = nlp_result["all_features"]

        # Organize features by category for the frontend
        prosody_keys = ["avg_syllables_per_line", "std_syllables_per_line", "iambic_score"]
        rhyme_keys = ["rhyme_density", "is_abab", "is_aabb", "is_abba"]
        vocab_keys = [
            "archaic_ratio", "nature_ratio", "dark_ratio", "divine_ratio",
            "sensory_ratio", "wordsworth_ratio", "tennyson_ratio", "keats_ratio",
            "type_token_ratio", "avg_word_length", "punct_density",
            "exclaim_ratio", "question_ratio",
        ]
        structure_keys = [
            "avg_line_length", "std_line_length", "repetition_score",
            "enjambment_ratio", "cap_ratio", "line_count", "flesch_score",
        ]
        pos_keys = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "prop_ratio"]

        def pick(keys):
            return [{"name": k, "value": all_features.get(k, 0.0)} for k in keys if k in all_features]

        # TF-IDF top terms (word-level)
        tfidf_terms = []
        try:
            tfidf_vec = state.nlp_tfidf.transform([formatted])
            feature_names = state.nlp_tfidf.get_feature_names_out()
            nonzero = tfidf_vec.nonzero()
            term_scores = [
                (feature_names[col], round(float(tfidf_vec[0, col]), 4))
                for col in nonzero[1]
            ]
            term_scores.sort(key=lambda x: x[1], reverse=True)
            tfidf_terms = [{"term": t, "score": s} for t, s in term_scores[:15]]
        except Exception:
            pass

        payload["nlp"] = {
            "predicted_poet": nlp_result["predicted_poet"],
            "confidence": nlp_result["confidence"],
            "top_poets": nlp_result["top_poets"],
            "feature_highlights": nlp_result["feature_highlights"],
            "categories": {
                "prosody": pick(prosody_keys),
                "rhyme": pick(rhyme_keys),
                "vocabulary": pick(vocab_keys),
                "structure": pick(structure_keys),
                "pos": pick(pos_keys),
            },
            "tfidf_top_terms": tfidf_terms,
        }

    return payload


@app.get("/corpus-embeddings")
def corpus_embeddings() -> dict[str, Any]:
    state = get_state()
    return {
        "points": state.corpus_points,
        "poets": [
            {
                "poet": poet,
                "color": metadata["color"],
                "era": metadata["era"],
            }
            for poet, metadata in POET_METADATA.items()
        ],
    }


@app.get("/dataset-stats")
def dataset_stats() -> dict[str, Any]:
    state = get_state()
    return {
        "stats": state.dataset_stats,
        "poet_metadata": POET_METADATA,
    }
