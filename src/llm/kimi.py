"""Shared Kimi (Moonshot) API client for all five roles:
image_description | synthetic_pairs | golden_queries | judge | demo_generation.

OpenAI-compatible endpoint. kimi-k2.5 is the cheapest viable text+vision tier.
CRITICAL: enable_thinking must be False on every call — thinking is ON by
default and reasoning tokens consume the max_tokens quota (budget & truncation bomb).
"""
import base64
import os
import time

import httpx

BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
TEXT_MODEL = os.environ.get("KIMI_MODEL", "moonshot-v1-8k")
VISION_MODEL = os.environ.get("KIMI_VISION_MODEL", "moonshot-v1-8k-vision-preview")
ENABLE_THINKING = False  # verified no-op on kimi-k2.x; harmless on v1


class KimiError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("KIMI_API_KEY", "")
    if not key:
        raise KimiError("KIMI_API_KEY not set")
    return key


def _post(payload: dict, timeout: float = 120.0, retries: int = 4) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {_key()}"},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network error / 5xx / 429 → backoff & retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise KimiError(f"kimi call failed after {retries} attempts: {last_err}")


def chat(
    messages,
    max_tokens: int = 400,
    temperature: float | None = None,  # None → API default (0.6); never pass arbitrary temp
    model: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Return the assistant's text. Raises on empty answer."""
    payload = {
        "model": model or TEXT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "enable_thinking": ENABLE_THINKING,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    data = _post(payload, timeout=timeout)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise KimiError(f"unexpected kimi response shape: {str(data)[:200]}")
    if not content:
        raise KimiError("kimi returned empty content")
    return content.strip()


def describe_image(image_bytes: bytes, instruction: str = "", mime: str = "image/png", max_tokens: int = 400) -> str:
    """Textify an image/chart for ingestion (role: image_description). Uses VISION_MODEL."""
    b64 = base64.b64encode(image_bytes).decode()
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        },
        {"type": "text", "text": instruction or "请用中文详细描述这张图的内容:图的类型、标题、坐标轴、关键数值与结论。"},
    ]
    return chat([{"role": "user", "content": user_content}], max_tokens=max_tokens, model=VISION_MODEL)
