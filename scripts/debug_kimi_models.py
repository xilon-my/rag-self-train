import os, json, httpx

key = os.environ["KIMI_API_KEY"]
MODELS = [
    "kimi-k2.5",
    "kimi-k2.6",
    "moonshot-v1-8k",
    "moonshot-v1-8k-vision-preview",
]
PROMPT = "请用三句话回答:什么是 RAG,并举一个中文例子。"
for model in MODELS:
    for max_tokens in [120, 600]:
        try:
            r = httpx.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": max_tokens,
                    "enable_thinking": False,
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=90,
            )
            j = r.json()
            msg = j.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            reason = msg.get("reasoning_content", "")
            finish = j.get("choices", [{}])[0].get("finish_reason")
            usage = j.get("usage", {})
            rt = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            print(
                f"[{model} mt={max_tokens}] status={r.status_code} finish={finish} "
                f"content_len={len(content)} reason_len={len(reason)} reasoning_tokens={rt} "
                f"comp_tok={usage.get('completion_tokens')}"
            )
            if content:
                print(f"   CONTENT: {content[:90]!r}")
        except Exception as e:
            print(f"[{model} mt={max_tokens}] EXC {type(e).__name__}: {str(e)[:150]}")
