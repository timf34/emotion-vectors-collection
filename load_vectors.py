"""Load emotion / syndrome vectors from the Hugging Face dataset timf34/emotion-vectors.

    from load_vectors import load_vectors
    vecs = load_vectors("qwen3_32b")                 # denoised emotion vectors
    v = vecs["depressed"][40]                        # torch.Tensor [hidden]

    python load_vectors.py qwen3_32b depressed       # prints available layers and norms
"""

from __future__ import annotations

import json
import sys

import torch
from huggingface_hub import hf_hub_download

REPO = "timf34/emotion-vectors"


def load_vectors(model: str, kind: str = "dn", label_set: str = "emotions") -> dict[str, dict[int, torch.Tensor]]:
    """model: folder name (e.g. "gemma3_27b"); kind: "dn" (denoised, analysis layers) or "raw" (every layer);
    label_set: "emotions" (42 words) or "syndromes" (6 clusters). Returns {label: {layer: vector}}."""
    path = hf_hub_download(REPO, f"{model}/vectors_{label_set}_{kind}.pt", repo_type="dataset")
    return torch.load(path, map_location="cpu")


def load_meta(model: str) -> dict:
    with open(hf_hub_download(REPO, f"{model}/meta.json", repo_type="dataset")) as f:
        return json.load(f)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma3_27b"
    label = sys.argv[2] if len(sys.argv) > 2 else "depressed"
    vecs = load_vectors(model)
    print(f"{model}: {len(vecs)} labels; {label} at layers {sorted(vecs[label])}")
    for layer, v in sorted(vecs[label].items()):
        print(f"  layer {layer:>3}  dim {v.shape[0]}  norm {v.norm():.1f}")
