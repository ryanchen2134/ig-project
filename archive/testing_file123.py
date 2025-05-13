# diagnostics_runner.py
"""
Robust loader & diagnostic script for the contrastive image‑song model.
Detects projection dimensions from checkpoint weights, then confirms
1) song projection input‑dim matches CSV vectors
2) image encoder is not collapsed (cosine(img1,img2) < 0.95)

Example:
    python diagnostics_runner.py \
        --ckpt checkpoints/best_model.pth \
        --csv  data_cleaned.csv \
        --img1 examples/a.jpg --img2 examples/b.jpg \
        --device cuda
"""
import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

# ---------------- helper fns -------------------------------------------------

def parse_vec(s: str) -> torch.Tensor:
    """Convert CSV string "[0.1, 0.2, ...]" → 1‑D float tensor."""
    return torch.tensor(eval(s), dtype=torch.float32)


def load_img(path: str | Path, device: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def find_weight_key(state: dict, keywords: tuple[str, ...]) -> str:
    """Return first key ending with ".weight" whose name contains **all** keywords.
    Raises with helpful key dump if not found."""
    kws = tuple(k.lower() for k in keywords)
    for k in state:
        if k.endswith(".weight") and all(kw in k.lower() for kw in kws):
            return k
    # nothing matched
    print("\nAvailable weight keys:\n" + "\n".join(state.keys()))
    raise KeyError(
        f"No weight tensor found whose name contains {keywords} and ends with '.weight'."
    )

# ---------------- CLI -------------------------------------------------------

parser = argparse.ArgumentParser(description="Contrastive model diagnostics")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--img1", required=True)
parser.add_argument("--img2", required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

device = args.device
ckpt_path = Path(args.ckpt)
print("=== Loading checkpoint ===")
ckpt_raw = torch.load(ckpt_path, map_location=device)

# determine which key actually stores weights
if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw:
    state_dict = ckpt_raw["state_dict"]
elif isinstance(ckpt_raw, dict) and "model_state_dict" in ckpt_raw:
    state_dict = ckpt_raw["model_state_dict"]
else:
    state_dict = ckpt_raw  # assume plain state‑dict

# ---------------- infer dims -----------------------------------------------

song_w_key = find_weight_key(state_dict, ("song", "proj"))
emb_dim, song_dim = state_dict[song_w_key].shape
print(f"Detected song projection: {song_w_key}  → shape ({emb_dim}, {song_dim})")

img_w_key = None
for cand in ("image", "img", "vision"):
    try:
        img_w_key = find_weight_key(state_dict, (cand, "proj"))
        break
    except KeyError:
        continue
if img_w_key:
    print(f"Detected image projection: {img_w_key} → out_dim {state_dict[img_w_key].shape[0]}")

# ---------------- rebuild model --------------------------------------------

from models.decoder.contrastive import ContrastiveImageSongModel  # adjust if path differs

model = ContrastiveImageSongModel(
    backbone_type="resnet18",  # change if you used a different backbone
    embedding_dim=emb_dim,
    song_embedding_dim=song_dim
).to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---------------- song proj test -------------------------------------------
print("=== Song projection sanity‑check ===")

csv_df = pd.read_csv(args.csv)
if csv_df.empty:
    raise RuntimeError("CSV is empty")
vec = parse_vec(csv_df.iloc[0].audio_embedding).to(device)
try:
    out = model.song_projection(vec)
    print("Song sample projected OK →", out.shape)
except Exception as e:
    raise RuntimeError("Projection failed; dim mismatch?") from e

# ---------------- image variance test --------------------------------------
print("=== Image variance test ===")
img1_emb = model.image_encoder(load_img(args.img1, device))
img2_emb = model.image_encoder(load_img(args.img2, device))
cos = torch.nn.functional.cosine_similarity(img1_emb, img2_emb).item()
print(f"cos(img1, img2) = {cos:.4f}")
if cos > 0.95:
    print("⚠️  Image branch may be collapsed (very high similarity)")
else:
    print("✅  Image branch shows healthy variance")

print("Diagnostics finished.")
