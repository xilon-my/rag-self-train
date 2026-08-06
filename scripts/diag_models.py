import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

texts = [
    "为这个句子生成表示以用于检索相关文章：试用期可以随时辞职吗",
    "劳动者在试用期内提前三日通知用人单位,可以解除劳动合同。",
]
base = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cuda")
tr = SentenceTransformer("checkpoints/biencoder", device="cuda")
vb = base.encode(texts, normalize_embeddings=True)
vt = tr.encode(texts, normalize_embeddings=True)
print("bi-encoder delta:", float(np.linalg.norm(vb - vt)))
del base, tr
import gc; gc.collect()

rb = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
rt = CrossEncoder("checkpoints/reranker", device="cuda")
for q, p in [(texts[0], texts[1]), (texts[1], texts[1])]:
    print("pair", q[:15], "| base:", float(rb.predict([[q, p]])[0]), "| trained:", float(rt.predict([[q, p]])[0]))
