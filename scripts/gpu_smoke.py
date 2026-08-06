"""D1 GPU smoke test: verify torch 2.7 on RTX 5090 (sm_120) can load a bge
model and run a forward + backward pass (the minimum needed for fine-tuning).

Usage:  HF_ENDPOINT=https://hf-mirror.com python scripts/gpu_smoke.py
"""
import os
import torch
from sentence_transformers import SentenceTransformer

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | device={dev}")
    if dev == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} | sm: {torch.cuda.get_device_capability(0)}")

    model_name = os.environ.get("SMOKE_MODEL", "BAAI/bge-base-zh-v1.5")
    print(f"loading {model_name} ...")
    m = SentenceTransformer(model_name, device=dev)
    m.train()

    texts = [
        "试用期可以随时辞职吗",
        "劳动者在试用期内提前三日通知用人单位,可以解除劳动合同。",
        "发明专利的保护期限是多久",
    ]
    tok = m.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
    out = m(tok)  # train mode → autograd graph kept
    emb = out["sentence_embedding"]
    print(f"encode OK: emb shape={tuple(emb.shape)}")

    # dummy contrastive forward+backward (the minimal training step)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    logits = emb @ emb.T / 0.05
    target = torch.arange(emb.shape[0]).to(dev)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    gn = torch.norm(torch.stack([p.grad.norm() for p in m.parameters() if p.grad is not None]))
    print(f"forward+backward OK | loss={loss.item():.4f} | grad_norm={gn.item():.4f}")
    print("GPU_SMOKE_PASS")

if __name__ == "__main__":
    main()
