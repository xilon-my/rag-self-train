"""Fine-tune a bi-encoder (bge-base-zh-v1.5) with InfoNCE contrastive loss.

Data: train_triples.jsonl (query, pos, neg). In-batch negatives + explicit hard
negatives. The query-side uses the same s2p instruction as inference, so the
trained encoder plugs straight into the existing retrieval pipeline.
"""
import json
import os
import sys

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

DATA = "data"
OUT_DIR = "checkpoints/biencoder"
MODEL = "BAAI/bge-base-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def load_triples(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    examples = []
    for r in rows:
        # anchor = query WITH the s2p instruction (must match inference, or it degrades)
        examples.append(InputExample(texts=[QUERY_INSTRUCTION + r["query"], r["pos"], r["neg"][0]]))
    return examples


def main():
    examples = load_triples(f"{DATA}/train_triples.jsonl")
    print(f"triples: {len(examples)}")
    model = SentenceTransformer(MODEL, device="cuda")
    train_dl = DataLoader(examples, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.fit(
        train_objectives=[(train_dl, loss)],
        epochs=3,
        warmup_steps=100,
        optimizer_params={"lr": 1e-5},
        output_path=OUT_DIR,
        save_best_model=False,
    )
    print(f"biencoder saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
