"""Convert Agent action plans into safe ROS-level intents."""

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from geometry_msgs.msg import Twist

from .gesture_controller import ControlMode
from .utils import make_twist


class RobotExecutor:
    """把 Agent 输出转换为模式切换或低速动作。

    LLM 不直接控制速度。这里仅接受白名单 action，并把它们映射为固定、
    保守、可中断的 ROS 命令或模式切换。
    """

    def __init__(self, max_history: int, forward_speed: float, turn_speed: float, action_duration_sec: float = 0.8) -> None:
        self.logs: Deque[Dict[str, object]] = deque(maxlen=int(max_history))
        self.forward_speed = abs(float(forward_speed))
        self.turn_speed = abs(float(turn_speed))
        self.action_duration_sec = float(action_duration_sec)
        self.active_until = 0.0
        self.active_started_at = 0.0
        self.active_cmd: Optional[Twist] = None
        self.active_action = "STOP"
        self.last_plan = {"action": "STOP", "reason": "not planned", "speak": ""}
        self.last_latency_ms = 0.0
        self.last_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def update_plan(self, plan: Dict[str, str], latency_ms: float = 0.0, tokens: Optional[Dict[str, int]] = None) -> None:
        action = str(plan.get("action", "STOP")).upper()
        reason = str(plan.get("reason", ""))
        self.last_plan = {"action": action, "reason": reason, "speak": str(plan.get("speak", ""))}
        self.last_latency_ms = float(latency_ms)
        self.last_tokens = tokens or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.logs.appendleft(
            {
                "timestamp": time.time(),
                "action": action,
                "reason": reason,
                "latency_ms": round(self.last_latency_ms, 1),
                "tokens": dict(self.last_tokens),
            }
        )

        if action == "FORWARD":
            self._start_motion(action, make_twist(self.forward_speed, 0.0))
        elif action == "BACKWARD":
            self._start_motion(action, make_twist(-self.forward_speed, 0.0))
        elif action == "TURN_LEFT":
            self._start_motion(action, make_twist(0.0, self.turn_speed))
        elif action == "TURN_RIGHT":
            self._start_motion(action, make_twist(0.0, -self.turn_speed))
        else:
            self.active_cmd = None
            self.active_until = 0.0
            self.active_started_at = 0.0
            self.active_action = action

    def resolve(self, current_mode: str) -> Tuple[str, Optional[Twist], str]:
        """返回建议模式、一次性动作命令和原因。"""
        action = self.last_plan.get("action", "STOP")
        reason = self.last_plan.get("reason", "")
        if action == "STOP":
            return current_mode, None, reason
        if action == "FOLLOW_PERSON":
            return ControlMode.FOLLOW, None, reason
        if action == "SEARCH_TARGET":
            return ControlMode.AGENT_MODE, make_twist(0.0, self.turn_speed * 0.5), reason
        if action == "PATROL":
            return ControlMode.PATROL_READY, None, reason
        if action == "SPEAK":
            return current_mode, None, reason
        if self.active_cmd is not None and time.time() <= self.active_until:
            return ControlMode.AGENT_MODE, self.active_cmd, reason
        self.active_started_at = 0.0
        return current_mode, None, "agent action finished"

    def _start_motion(self, action: str, cmd: Twist) -> None:
        self.active_action = action
        self.active_cmd = cmd
        self.active_started_at = time.time()
        self.active_until = self.active_started_at + self.action_duration_sec
