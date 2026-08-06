"""Quick empirical check: which kwargs make the RT-DETR processor accept a
batch of differently-sized images (docling's layout model bug)."""
from transformers import AutoProcessor
from PIL import Image

p = AutoProcessor.from_pretrained("docling/layout-rtdetr")
print("processor size:", p.size)
imgs = [Image.new("RGB", (800, 1200), "white"), Image.new("RGB", (600, 900), "black")]
for kw in [dict(padding=True), dict(size={"longest_edge": 1024}), dict(size=1024)]:
    try:
        out = p(images=imgs, return_tensors="pt", **kw)
        print("OK", kw, "->", tuple(out["pixel_values"].shape))
    except Exception as e:
        print("FAIL", kw, "->", str(e)[:100])
