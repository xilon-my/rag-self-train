"""Gradio demo: query -> retrieved pages (text + rendered page image).
Run: python scripts/run_gradio.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gradio as gr  # noqa: E402
from demo.engine import DemoEngine  # noqa: E402

eng = DemoEngine()

EXAMPLE_QUERIES = [
    "根据苏州科达2020年度报告,计算其他流动资产中'留抵进项税'在期末和期初余额中的变化率",
    "分析2018年家庭建立期女性用户选择金融产品的主要动因中,哪个因素占比最高?",
    "根据文件中的图表内容,2022年12月美国的GDP平减指数是多少?",
]


def answer(query):
    hits = eng.retrieve(query)
    md = [f"### 查询: {query}\n"]
    tmp = tempfile.mkdtemp()
    for h in hits:
        img = eng.render_page(h["doc"], h["page"], os.path.join(tmp, f"p{h['page']}.png"))
        md.append(f"**{h['doc']} — 第 {h['page']} 页** (score {h['score']:.3f})")
        md.append(f"\n{h['text'][:200]}...\n")
        if img:
            md.append(f"![page {h['page']}]({img})")
        md.append("\n---\n")
    return "\n".join(md)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="金融问题", lines=2),
    outputs=gr.Markdown(label="检索结果(带页面原图)"),
    title="RAGSelfTrain — 自训练检索模型的多模态金融文档 RAG",
    description="自训 110M bi-encoder + 278M reranker,在 FinRAGBench-V 上检索",
    examples=[[q] for q in EXAMPLE_QUERIES],
)

if __name__ == "__main__":
    demo.launch()
