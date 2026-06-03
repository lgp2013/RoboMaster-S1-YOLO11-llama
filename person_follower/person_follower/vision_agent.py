"""Vision-language model client for Qwen3-VL compatible OpenAI APIs."""

import base64
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import requests


@dataclass
class VisionAgentResult:
    description: str
    latency_ms: float
    tokens: Dict[str, int]
    error: str = ""


class VisionAgent:
    """调用 OpenAI 兼容 VLM 接口，把当前画面转换成自然语言描述。"""

    def __init__(self, enabled: bool, base_url: str, model: str, timeout_sec: float = 8.0) -> None:
        self.enabled = bool(enabled)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = float(timeout_sec)

    def describe(self, frame_bgr, prompt: Optional[str] = None) -> VisionAgentResult:
        if not self.enabled:
            return VisionAgentResult("VLM disabled", 0.0, {})

        ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return VisionAgentResult("", 0.0, {}, "frame encode failed")

        image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        image_url = "data:image/jpeg;base64,%s" % image_b64
        user_prompt = prompt or (
            "请用一句中文描述机器人前方场景，重点说明人数、手势、障碍物、"
            "水杯、椅子、桌子、墙、目标方位和潜在危险。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        started = time.time()
        try:
            response = requests.post(
                "%s/chat/completions" % self.base_url,
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {}) or {}
            tokens = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            return VisionAgentResult(content.strip(), (time.time() - started) * 1000.0, tokens)
        except Exception as exc:
            return VisionAgentResult("", (time.time() - started) * 1000.0, {}, str(exc))
