"""LLM planning agent for safe RoboMaster action plans."""

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List

import requests


ACTION_WHITELIST = {
    "STOP",
    "FORWARD",
    "BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FOLLOW_PERSON",
    "SEARCH_TARGET",
    "SPEAK",
    "PATROL",
}


SYSTEM_PROMPT = """
你是 RoboMaster S1 的行为规划智能体。
你只能输出 JSON，不要输出 Markdown，不要解释。
你不能直接控制底盘速度，只能输出动作计划，最终动作会经过 Safety Filter。
可选 action 只能是：
STOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, FOLLOW_PERSON, SEARCH_TARGET, SPEAK, PATROL
安全规则：
1. 看到危险、跌倒、墙、桌子、柜子、遮挡、目标丢失时优先 STOP。
2. 目标跟随用 FOLLOW_PERSON，不要连续建议 FORWARD。
3. 不确定时 STOP。
4. 不要射击，不要攻击，不要撞击。
输出示例：
{"action":"FOLLOW_PERSON","reason":"person waving","speak":""}
"""


@dataclass
class PlannerAgentResult:
    plan: Dict[str, str]
    latency_ms: float
    tokens: Dict[str, int]
    raw_text: str = ""
    error: str = ""


class PlannerAgent:
    """调用 OpenAI 兼容 Llama Server，根据场景输出动作计划。"""

    def __init__(self, enabled: bool, base_url: str, model: str, timeout_sec: float = 8.0) -> None:
        self.enabled = bool(enabled)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = float(timeout_sec)

    def plan(self, scene: Dict[str, object], robot_state: Dict[str, object], mode: str, user_request: str = "") -> PlannerAgentResult:
        if not self.enabled:
            return PlannerAgentResult({"action": "STOP", "reason": "LLM disabled", "speak": ""}, 0.0, {})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "scene": scene,
                            "robot_state": robot_state,
                            "current_mode": mode,
                            "user_request": user_request,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 160,
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
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {}) or {}
            plan = extract_plan_json(text)
            tokens = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            return PlannerAgentResult(plan, (time.time() - started) * 1000.0, tokens, raw_text=text)
        except Exception as exc:
            return PlannerAgentResult(
                {"action": "STOP", "reason": "LLM request failed: %s" % exc, "speak": ""},
                (time.time() - started) * 1000.0,
                {},
                error=str(exc),
            )


def extract_plan_json(text: str) -> Dict[str, str]:
    """解析模型 JSON；如果混入其它文字，提取第一个 {...}。"""
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*?\}", text, re.S)
        if not match:
            return {"action": "STOP", "reason": "LLM output parse failed", "speak": ""}
        try:
            data = json.loads(match.group(0))
        except Exception:
            return {"action": "STOP", "reason": "LLM output parse failed", "speak": ""}

    action = str(data.get("action", "STOP")).upper()
    if action not in ACTION_WHITELIST:
        action = "STOP"
    return {
        "action": action,
        "reason": str(data.get("reason", "no reason")),
        "speak": str(data.get("speak", "")),
    }
