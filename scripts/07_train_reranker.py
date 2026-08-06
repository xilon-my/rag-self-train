"""Fine-tune a cross-encoder reranker (bge-reranker-base) with BCE.

Pairs: (query, pos) -> 1, (query, neg) -> 0. No query instruction for cross-
encoders. Output is a CrossEncoder checkpoint loadable by the retrieval pipeline.
"""
import json
import os
import sys

from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

DATA = "data"
OUT_DIR = "checkpoints/reranker"
MODEL = "BAAI/bge-reranker-base"


def load_pairs(path):
    examples = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        examples.append(InputExample(texts=[r["query"], r["pos"]], label=1.0))
        for neg in r["neg"]:
            examples.append(InputExample(texts=[r["query"], neg], label=0.0))
    return examples


def main():
    pairs = load_pairs(f"{DATA}/train_triples.jsonl")
    print(f"pairs: {len(pairs)}")
    model = CrossEncoder(MODEL, num_labels=1, max_length=512, device="cuda")
    train_dl = DataLoader(pairs, shuffle=True, batch_size=16)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.fit(
        train_dataloader=train_dl,
        epochs=2,
        warmup_steps=100,
        optimizer_params={"lr": 1e-5},
        show_progress_bar=True,
    )
    model.save(OUT_DIR)  # CrossEncoder.fit() doesn't persist with output_path; save explicitly
    print(f"reranker saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
