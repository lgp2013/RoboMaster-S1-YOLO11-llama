"""Test the local llama.cpp OpenAI-compatible chat completions API."""

from __future__ import annotations

import json

import requests

from config import LLM_BASE_URL, LLM_MODEL


def main() -> int:
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是机器人控制助手，只输出JSON。"},
            {"role": "user", "content": "检测到 person 在画面左侧，距离较远，请给出动作建议。"},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }

    print(f"POST {url}")
    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[FAIL] llama.cpp API request failed: {exc}")
        return 1

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        print(f"[FAIL] llama.cpp API returned non-JSON response: {exc}")
        print(response.text)
        return 1

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        print(f"[FAIL] Unexpected llama.cpp response shape: {exc}")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 1

    print("[OK] Model response:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
