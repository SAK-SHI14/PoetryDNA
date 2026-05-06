import torch
import json

# Load your label encoder
ENCODER_PATH = "DL-Mehtab/results/label_encoder.pt"
le = torch.load(ENCODER_PATH, weights_only=False)

# Save classes as JSON
with open("DL-Mehtab/results/classes.json", "w") as f:
    json.dump(list(le.classes_), f)

print(f"Exported {len(le.classes_)} classes to results/classes.json ✅")
