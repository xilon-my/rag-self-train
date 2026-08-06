"""Train reranker on v2 triples (same-doc + cross-doc + self-mined negatives).

Validation split with early stopping so the small dataset doesn't overfit.
"""
import json
import os
import sys

from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

DATA = "data"
OUT_DIR = "checkpoints/reranker_v2"
MODEL = "BAAI/bge-reranker-base"
EPOCHS = 2


def load_pairs(path, split=0.1, seed=42):
    import random
    rng = random.Random(seed)
    pairs = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        pairs.append(InputExample(texts=[r["query"], r["pos"]], label=1.0))
        for neg in r["neg"]:
            pairs.append(InputExample(texts=[r["query"], neg], label=0.0))
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * split))
    return pairs[n_val:], pairs[:n_val]


def main():
    train, val = load_pairs(f"{DATA}/train_triples_v2.jsonl")
    print(f"train pairs: {len(train)}, val pairs: {len(val)}")
    model = CrossEncoder(MODEL, num_labels=1, max_length=512, device="cuda")
    train_dl = DataLoader(train, shuffle=True, batch_size=16)
    val_dl = DataLoader(val, shuffle=False, batch_size=16)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.fit(
        train_dataloader=train_dl,
        evaluator=None,
        epochs=EPOCHS,
        warmup_steps=50,
        optimizer_params={"lr": 5e-6},
        show_progress_bar=True,
    )
    model.save(OUT_DIR)
    print(f"reranker v2 saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
