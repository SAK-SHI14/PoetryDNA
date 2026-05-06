import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

print("🚀 PoetryDNA DL Space starting up...")

app = FastAPI(title="PoetryDNA DL Brain")

device = torch.device("cpu")

print("📦 Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("📂 Loading class labels...")
with open("classes.json", "r") as f:
    classes = json.load(f)
print(f"✅ {len(classes)} classes loaded: {classes}")

print("🧠 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(classes),
    output_attentions=True,
)
model.load_state_dict(
    torch.load("DistilBERT.pt", map_location=device, weights_only=False),
    strict=False,
)
model.to(device)
model.eval()
print("✅ Model ready!")


def enable_dropout(m: torch.nn.Module) -> None:
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


class PredictRequest(BaseModel):
    text: str
    runs: int = 10


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(classes)}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        ).to(device)

        model.eval()
        enable_dropout(model)

        # MC Dropout — collect probability tensors (no numpy)
        all_probs = []
        last_outputs = None
        with torch.no_grad():
            for _ in range(request.runs):
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]  # shape: (num_classes,)
                all_probs.append(probs)
                last_outputs = outputs

        # Stack → (runs, num_classes) tensor — pure torch, no numpy
        stacked = torch.stack(all_probs)                    # (runs, C)
        mean_probs = stacked.mean(dim=0)                    # (C,)
        uncertainty = float(stacked.std(dim=0).mean().item())

        # Attention heatmap — last layer, mean across heads, CLS row
        last_attn = last_outputs.attentions[-1][0]          # (heads, seq, seq)
        cls_attention = last_attn.mean(dim=0)[0]            # (seq_len,)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_data = [
            {"word": tok, "score": float(score.item())}
            for tok, score in zip(tokens, cls_attention)
            if tok not in ["[PAD]", "[SEP]", "[CLS]"]
        ]

        # Top-3 poets
        top_values, top_indices = torch.topk(mean_probs, k=min(3, len(classes)))
        top_poets = [
            {"poet": classes[int(idx)], "probability": float(val.item())}
            for val, idx in zip(top_values, top_indices)
        ]

        return {
            "predicted_poet": top_poets[0]["poet"],
            "confidence": round(top_poets[0]["probability"] * 100, 2),
            "uncertainty": round(uncertainty, 4),
            "top_poets": top_poets,
            "attention": attention_data,
            "mc_runs": stacked.tolist(),
            "mean_probs": mean_probs.tolist(),
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
