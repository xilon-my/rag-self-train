"""Distill the trained 278M reranker's margin signal into the 110M bi-encoder.

Teacher = checkpoints/reranker_v2 (cross-encoder). For each training triple it
scores (q, pos) vs (q, neg) -> margin. Student bi-encoder learns to reproduce
the margin with MSE on (sim(q,pos) - sim(q,neg)). Row 6 in the ablation.

Start from the already-fine-tuned bi-encoder (checkpoints/biencoder) so we keep
the contrastive gains and only add the reranker's ranking signal.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA = "data"
OUT_DIR = "checkpoints/biencoder_distilled"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
LR = 1e-5
EPOCHS = 3
BATCH = 16


def main():
    triples = [json.loads(l) for l in open(f"{DATA}/train_triples_v2.jsonl", encoding="utf-8")]
    print(f"triples: {len(triples)}")

    # teacher margins
    teacher = CrossEncoder("checkpoints/reranker_v2", device="cuda")
    samples = []  # (q, pos, neg, teacher_margin)
    for t in triples:
        q = t["query"]
        s_pos = float(teacher.predict([[q, t["pos"]]])[0])
        for neg in t["neg"]:
            s_neg = float(teacher.predict([[q, neg]])[0])
            samples.append((q, t["pos"], neg, s_pos - s_neg))
    print(f"margin samples: {len(samples)}")

    # student
    model = SentenceTransformer("checkpoints/biencoder", device="cuda")
    tokenizer = model.tokenizer
    dev = "cuda"

    def tok(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total = 0.0; n = 0
        for i in range(0, len(samples), BATCH):
            batch = samples[i:i + BATCH]
            zq = model(tok([QUERY_INSTRUCTION + s[0] for s in batch]))["sentence_embedding"]
            zp = model(tok([s[1] for s in batch]))["sentence_embedding"]
            zn = model(tok([s[2] for s in batch]))["sentence_embedding"]
            zq = F.normalize(zq, dim=-1); zp = F.normalize(zp, dim=-1); zn = F.normalize(zn, dim=-1)
            pred = (zq * zp).sum(-1) - (zq * zn).sum(-1)
            margins = torch.tensor([s[3] for s in batch], device=dev, dtype=torch.float)
            loss = F.mse_loss(pred, margins)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(batch); n += len(batch)
        print(f"epoch {epoch+1}: margin-MSE = {total/n:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(OUT_DIR)
    print(f"distilled bi-encoder saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
