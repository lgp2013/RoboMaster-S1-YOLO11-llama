"""Gesture debounce, mode switching and ROS topic publishing."""

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from geometry_msgs.msg import Twist

from .gesture_detector import GestureResult
from .utils import json_string_msg, make_twist


class ControlMode:
    IDLE = "IDLE"
    FOLLOW = "FOLLOW"
    GESTURE_CONTROL = "GESTURE_CONTROL"
    PATROL_READY = "PATROL_READY"
    AGENT_MODE = "AGENT_MODE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class GestureDebouncer:
    """同一手势连续出现多帧后才触发，避免单帧误识别。"""

    def __init__(self, stable_frame_count: int, cooldown_seconds: float) -> None:
        self.stable_frame_count = int(stable_frame_count)
        self.cooldown_seconds = float(cooldown_seconds)
        self.candidate_gesture = "none"
        self.candidate_count = 0
        self.stable_gesture = "none"
        self.last_trigger_time = 0.0

    def update(self, gesture: Optional[GestureResult]) -> Tuple[Optional[str], Dict[str, object]]:
        name = gesture.gesture if gesture is not None else "none"
        confidence = gesture.confidence if gesture is not None else 0.0

        if name == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = name
            self.candidate_count = 1

        now = time.time()
        stable = False
        triggered = None
        if (
            name not in ("none", "unknown")
            and self.candidate_count >= self.stable_frame_count
            and now - self.last_trigger_time >= self.cooldown_seconds
        ):
            self.stable_gesture = name
            self.last_trigger_time = now
            stable = True
            triggered = name

        state = {
            "candidate": self.candidate_gesture,
            "candidate_count": self.candidate_count,
            "stable_gesture": self.stable_gesture,
            "stable": stable,
            "confidence": round(float(confidence), 3),
            "cooldown_remaining": round(max(0.0, self.cooldown_seconds - (now - self.last_trigger_time)), 2),
        }
        return triggered, state


class GestureController:
    """根据稳定手势切换控制模式，并发布 gesture state/command。"""

    def __init__(
        self,
        state_pub,
        command_pub,
        cmd_pub,
        gesture_actions: Dict[str, str],
        default_mode: str = ControlMode.IDLE,
        follow_mode_enabled: bool = True,
        emergency_stop_enabled: bool = True,
        turn_angular_speed: float = 0.6,
        action_duration_sec: float = 1.0,
    ) -> None:
        self.state_pub = state_pub
        self.command_pub = command_pub
        self.cmd_pub = cmd_pub
        self.gesture_actions = dict(gesture_actions)
        self.mode = default_mode if default_mode else ControlMode.IDLE
        self.previous_mode = ControlMode.IDLE
        self.follow_mode_enabled = bool(follow_mode_enabled)
        self.emergency_stop_enabled = bool(emergency_stop_enabled)
        self.turn_angular_speed = abs(float(turn_angular_speed))
        self.action_duration_sec = float(action_duration_sec)
        self.action_until = 0.0
        self.action_started_at = 0.0
        self.action_cmd = Twist()
        self.action_name = "NONE"
        self.logs: Deque[Dict[str, object]] = deque(maxlen=10)

    def publish_state(self, debounce_state: Dict[str, object], best: Optional[GestureResult]) -> None:
        payload = {
            "gesture": best.gesture if best is not None else "none",
            "stable": bool(debounce_state.get("stable", False)),
            "candidate": debounce_state.get("candidate", "none"),
            "stable_gesture": debounce_state.get("stable_gesture", "none"),
            "confidence": debounce_state.get("confidence", 0.0),
            "mode": self.mode,
        }
        self.state_pub.publish(json_string_msg(payload))

    def handle_stable_gesture(self, gesture_name: str) -> None:
        action = self.gesture_actions.get(gesture_name, "NONE")
        self._apply_action(action, source="gesture", gesture=gesture_name)

    def handle_web_command(self, command: str) -> Dict[str, object]:
        mapping = {
            "STOP": "STOP",
            "START_FOLLOW": "START_FOLLOW",
            "PAUSE": "PAUSE",
            "CLEAR_LOGS": "CLEAR_LOGS",
        }
        action = mapping.get(command, "NONE")
        if action == "CLEAR_LOGS":
            self.logs.clear()
            return {"ok": True, "message": "logs cleared"}
        self._apply_action(action, source="web", gesture="button")
        return {"ok": True, "mode": self.mode, "action": action}

    def force_stop(self, source: str = "safety", gesture: str = "open_palm") -> None:
        """最高优先级停止，供 open_palm 或异常流程立即调用。"""
        self._apply_action("STOP", source=source, gesture=gesture)

    def get_override_cmd(self) -> Tuple[Optional[Twist], str]:
        """GESTURE_CONTROL 模式下返回一次性动作速度。"""
        if self.mode != ControlMode.GESTURE_CONTROL:
            return None, self.action_name
        if time.time() <= self.action_until:
            return self.action_cmd, self.action_name

        self._publish_zero()
        self.mode = self.previous_mode if self.previous_mode in (ControlMode.FOLLOW, ControlMode.IDLE) else ControlMode.IDLE
        self.action_name = "DONE"
        self.action_started_at = 0.0
        return None, self.action_name

    def _apply_action(self, action: str, source: str, gesture: str) -> None:
        timestamp = time.time()
        if action == "STOP":
            self.previous_mode = self.mode
            self.mode = ControlMode.EMERGENCY_STOP if self.emergency_stop_enabled else ControlMode.IDLE
            self.action_name = "STOP"
            self._publish_zero()
        elif action == "START_FOLLOW":
            if self.follow_mode_enabled:
                self.previous_mode = self.mode
                self.mode = ControlMode.FOLLOW
                self.action_name = "START_FOLLOW"
        elif action == "PAUSE":
            self.previous_mode = self.mode
            self.mode = ControlMode.IDLE
            self.action_name = "PAUSE"
            self._publish_zero()
        elif action == "TURN_LEFT":
            self._start_turn(action, angular_z=abs(self.turn_angular_speed))
        elif action == "TURN_RIGHT":
            self._start_turn(action, angular_z=-abs(self.turn_angular_speed))
        elif action == "PATROL_READY":
            self.previous_mode = self.mode
            self.mode = ControlMode.PATROL_READY
            self.action_name = "PATROL_READY"
            self._publish_zero()
        elif action == "RESUME":
            self.mode = self.previous_mode if self.previous_mode else ControlMode.IDLE
            self.action_name = "RESUME"
        else:
            return

        payload = {
            "command": action,
            "source": source,
            "gesture": gesture,
            "mode": self.mode,
            "timestamp": timestamp,
        }
        self.command_pub.publish(json_string_msg(payload))
        self.logs.appendleft(payload)

    def _start_turn(self, action: str, angular_z: float) -> None:
        self.previous_mode = self.mode if self.mode != ControlMode.GESTURE_CONTROL else self.previous_mode
        self.mode = ControlMode.GESTURE_CONTROL
        self.action_name = action
        self.action_started_at = time.time()
        self.action_until = time.time() + self.action_duration_sec
        self.action_cmd = make_twist(0.0, angular_z)

    def _publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())
