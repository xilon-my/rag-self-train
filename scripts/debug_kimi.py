import os, json, httpx

key = os.environ["KIMI_API_KEY"]
for thinking in [False, True]:
    try:
        r = httpx.post(
            "https://api.moonshot.cn/v1/chat/completions",
            json={
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "只回复三个字:冒烟通过"}],
                "max_tokens": 50,
                "enable_thinking": thinking,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        print(f"--- enable_thinking={thinking} status={r.status_code} ---")
        print(json.dumps(r.json(), ensure_ascii=False)[:700])
    except Exception as e:
        print(f"--- enable_thinking={thinking} EXC {type(e).__name__}: {e} ---")
