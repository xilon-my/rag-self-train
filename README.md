# RAGSelfTrain — 自训练检索模型的多模态文档 RAG

> Same pipeline, same code, same 120 queries — the only variable is whose weights are in the models.

中文文档:中文金融 PDF 多模态文档 RAG,自研检索链路(BM25 + 稠密 + RRF + cross-encoder 重排),
InfoNCE 微调 bi-encoder、BCE 微调 reranker,权重发布 HuggingFace,ragas + RAGChecker 双框架评测。

*(脚手架:文档、脚本与实验结构将在 D1-D5 落地后补齐。)*

## 状态

- [ ] D1 语料 + GPU 冒烟 + 解析冒烟
- [ ] D2 全量解析 + 训练数据 + golden set
- [ ] D3 检索第一性原理 + off-the-shelf 基线
- [ ] D4 训练两模型 + 发布 HF
- [ ] D5 双框架评测 + Demo + README
