# RAGSelfTrain — 自训练检索模型的多模态金融文档 RAG

> 从第一性原理实现 BM25 + 稠密 + RRF + cross-encoder 重排的完整检索链路(无 LangChain / 向量数据库),在 FinRAGBench-V(EMNLP 2025, Apache-2.0)上用 **InfoNCE 微调 110M bi-encoder、BCE 微调 278M reranker**,并证明**自训模型在排序质量上超过 off-the-shelf 栈**——全部用真实、人工标注的评测与配对置信区间测量。

## 核心结果(99 条真实 golden 查询,同代码同查询,仅权重不同)

| 行 | 配置 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|
| ① | BM25(char-bigram) | 0.899 | 0.740 | 0.688 |
| ② | 冻结 dense + RRF | 0.879 | 0.722 | 0.673 |
| ③ | ② + 冻结 278M reranker(off-the-shelf 全栈) | 0.909 | 0.754 | 0.702 |
| ④ | **自训 110M dense + RRF(无 reranker)** | 0.909 | 0.753 | 0.702 |
| ⑤ | **自训 110M dense + 自训 278M reranker** | 0.909 | **0.797** | **0.761** |

**诚实措辞(由配对 bootstrap + Wilcoxon 决定,见 `results/stats_summary.md`)**:

- **头条(无需 p 值)**:29 秒微调的 110M bi-encoder,单独就**匹配**了需要冻结 278M reranker 的 off-the-shelf 栈(nDCG 0.753 vs 0.754,p=0.99 统计上不可区分)——微调把排序能力内化进了编码器,推理时省掉了整条 rerank 链。
- **次级**:训练两个模型,在 **MRR 上显著超过** off-the-shelf 全栈(+0.059,p=0.026);nDCG 上是方向性提升(+0.044,p=0.05,CI 含零,报告为"接近显著")。
- **召回已饱和**(0.909,99/99 可检索,9 条即使在最优行也够不着)——所以对比的是**排序质量**,不是召回。

## 简历故事:旧 → 新

**旧项目**(框架胶水 + 现成模型):"基于 RAGAnything + LightRAG + MinerU 构建,接入 vllm embedding、BGE-Reranker 精排… context precision 0.542→0.993"——所有模型都是"用"的,没有一个是"造"的。

**这个项目**:从 transformer 地基自写检索链路(无检索框架);**InfoNCE 微调 bi-encoder、BCE 微调 reranker**,权重可下载;与 off-the-shelf 栈同语料、同 99 条查询、同代码(仅权重不同)对照,自训栈在排序质量上超过 off-the-shelf 全栈。

## 关键工程判断(负样本质量 >> 损失函数)

- **v1 reranker 翻车**(nDCG 0.848):负样本只用了跨文档 BM25,**排除了同文档其他页**。但推理时 RRF 的 top-50 全是同文档页——reranker 从没见过这种最难的情况,不知道怎么拒绝。
- **v2 修复后夺冠**(nDCG 0.797):负样本升级为**同文档页(最难)+ 跨文档 BM25 + 自挖难负样本**(自训模型自己最困惑的页),每查询平均 6.7 个。
- **教训**:负样本的选择比损失函数更能决定上限——这句话这次是亲手验证的。

**也试过(诚实记录,未纳入最终结果)**:cross→bi 蒸馏(Margin-MSE,把 278M reranker 的分数差蒸进 110M bi-encoder)——没帮助(0.879 vs 对比微调 0.909):margin-MSE 在 618 条数据上把 embedding 带离了对比微调学到的检索信号。结论:此数据规模下**对比微调 > 蒸馏**(脚本 `scripts/10_distill_biencoder.py`)。

## 诚实的边界(可见即防身)

- **召回天花板 0.909**:9/99 的答案页即使最优行也够不着(OCR 后仍有部分图页检索不到)。
- **纯图表页依赖 MinerU OCR**:130 个查询引用的图片页由 MinerU 补了文字(pymupdf 对数字原生页抽文本,对纯图页抽不到)。
- **语料是子集**:8,008 页(193 文档),不是基准全量 60,780 页。
- **小数据微调**:整个训练 = 618 条三元组、一张 RTX 5090、约 2 分钟。

## 复现

1. 下载 `data/` + `checkpoints/`:[GitHub Release `v0.1-weights`](https://github.com/xilon-my/rag-self-train/releases/tag/v0.1-weights)(`rag_artifacts.tar.gz`),解压到 repo 根目录。
2. 语料 PDF 来自 FinRAGBench-V 的 `pdfs_for_QA/pdf_ch.tar.gz`(见 `scripts/00_fetch_corpus.sh`)。
3. 检索管线:`src/retrieve/`(bm25/dense/fusion/rerank/query_encoder)。
4. 评测:`scripts/08_full_ablation.py`(生成结果表)→ `scripts/09_stats.py`(配对 bootstrap + Wilcoxon)。
5. 训练:`scripts/05b_build_train_data_v2.py` → `06_train_biencoder.py` → `07b_train_reranker_v2.py`。

## Demo

```bash
# CLI
python scripts/run_demo_cli.py "根据苏州科达2020年度报告,计算留抵进项税在期末和期初的变化率" --render
# 网页(Gradio)
python scripts/run_gradio.py
```

查询 → 检索 top-5 页(得分)→ 渲染页面原图(图表/表格)→ 展示。模型加载一次,CPU/GPU 均可。

## 技术栈

Python · torch 2.7 (RTX 5090) · sentence-transformers · rank_bm25 · jieba · PyMuPDF · MinerU(仅图片页 OCR)· FinRAGBench-V
