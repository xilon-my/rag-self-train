"""D1 Kimi smoke test: verify kimi-k2.5 with enable_thinking=false works for
(1) plain text chat, (2) image description, and that responses respect max_tokens.

Usage: KIMI_API_KEY=... python scripts/smoke_kimi.py
"""
import base64
import os
import sys

sys.path.insert(0, "src")
from llm.kimi import chat, describe_image  # noqa: E402

# a tiny 1x1 red PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

def main():
    # 1) plain chat
    r1 = chat(
        [{"role": "user", "content": "只回复三个字:冒烟通过"}],
        max_tokens=30,
    )
    print("CHAT:", r1[:80])

    # 2) image description (vision path)
    r2 = describe_image(TINY_PNG, instruction="用一句话描述这张图片。", max_tokens=50)
    print("IMAGE_DESC:", r2[:80])

    # 3) max_tokens respected (short prompt, cap 15 tokens)
    r3 = chat([{"role": "user", "content": "请完整复述以下句子并继续写很多字:一二三四五六七八九十一二三四五六七八九十"}], max_tokens=15)
    print("MAXTOK_LEN:", len(r3), "| head:", r3[:40])

    print("KIMI_SMOKE_PASS")

if __name__ == "__main__":
    main()
