"""OpenCV dashboard renderer for the RoboMaster S1 AI control demo."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from config import (
    DASHBOARD_COLOR_BG,
    DASHBOARD_COLOR_BLUE,
    DASHBOARD_COLOR_BORDER,
    DASHBOARD_COLOR_GREEN,
    DASHBOARD_COLOR_MUTED,
    DASHBOARD_COLOR_PANEL,
    DASHBOARD_COLOR_PANEL_DARK,
    DASHBOARD_COLOR_RED,
    DASHBOARD_COLOR_TEXT,
    DASHBOARD_COLOR_YELLOW,
    DASHBOARD_HEIGHT,
    DASHBOARD_WIDTH,
    ENABLE_DASHBOARD,
    LOG_PANEL_HEIGHT,
    MAX_LOG_LINES,
    STATUS_PANEL_HEIGHT,
    STATUS_PANEL_WIDTH,
    VIDEO_PANEL_HEIGHT,
    VIDEO_PANEL_WIDTH,
)


COLOR_BG = DASHBOARD_COLOR_BG
COLOR_PANEL = DASHBOARD_COLOR_PANEL
COLOR_PANEL_DARK = DASHBOARD_COLOR_PANEL_DARK
COLOR_TEXT = DASHBOARD_COLOR_TEXT
COLOR_MUTED = DASHBOARD_COLOR_MUTED
COLOR_GREEN = DASHBOARD_COLOR_GREEN
COLOR_YELLOW = DASHBOARD_COLOR_YELLOW
COLOR_RED = DASHBOARD_COLOR_RED
COLOR_BLUE = DASHBOARD_COLOR_BLUE
COLOR_BORDER = DASHBOARD_COLOR_BORDER

HEADER_X, HEADER_Y = 0, 0
HEADER_W, HEADER_H = DASHBOARD_WIDTH, 50
VIDEO_X, VIDEO_Y = 20, 70
STATUS_X, STATUS_Y = 860, 70
LOG_X, LOG_Y = 20, 570
LOG_W = DASHBOARD_WIDTH - 40


def truncate_text(text: object, max_len: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


class DashboardRenderer:
    """Render a clean debugging dashboard. It does not control the robot."""

    def __init__(self) -> None:
        self.logs: List[str] = []

    def add_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        self.logs = self.logs[-MAX_LOG_LINES:]

    def render(
        self,
        frame,
        detection_info: Dict[str, object],
        llm_decision: Dict[str, object],
        safety_info: Dict[str, object],
        robot_status: Dict[str, object],
    ):
        if not ENABLE_DASHBOARD:
            return frame

        canvas = np.full((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), COLOR_BG, dtype=np.uint8)
        self._draw_header(canvas, robot_status)
        self._draw_video_panel(canvas, frame, detection_info)
        self._draw_status_panel(canvas, detection_info, llm_decision, safety_info, robot_status)
        self._draw_log_panel(canvas)
        return canvas

    def _draw_header(self, canvas, robot_status: Dict[str, object]) -> None:
        cv2.rectangle(canvas, (HEADER_X, HEADER_Y), (HEADER_X + HEADER_W, HEADER_Y + HEADER_H), COLOR_PANEL, -1)
        cv2.line(canvas, (0, HEADER_H - 1), (DASHBOARD_WIDTH, HEADER_H - 1), COLOR_BORDER, 1)
        title = "RoboMaster S1 AI Control Dashboard"
        mode = str(robot_status.get("control_mode", "AI Assist"))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._put_text(canvas, title, 20, 33, scale=0.8, thickness=2)
        self._put_text(canvas, f"Mode: {mode}", 570, 32, color=COLOR_MUTED)
        self._put_text(canvas, now, 760, 32, color=COLOR_MUTED)
        self._put_text(canvas, "q quit | s stop | m auto | r recenter", 990, 32, scale=0.5, color=COLOR_YELLOW)

    def _draw_video_panel(self, canvas, frame, detection_info: Dict[str, object]) -> None:
        self._draw_panel(canvas, VIDEO_X, VIDEO_Y, VIDEO_PANEL_WIDTH, VIDEO_PANEL_HEIGHT, "Camera + YOLO")
        inner_x, inner_y = VIDEO_X + 10, VIDEO_Y + 34
        inner_w, inner_h = VIDEO_PANEL_WIDTH - 20, VIDEO_PANEL_HEIGHT - 44
        if frame is None:
            cv2.rectangle(canvas, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), COLOR_PANEL_DARK, -1)
            self._put_text(canvas, "No camera frame", inner_x + 20, inner_y + 40, color=COLOR_YELLOW)
            return

        video = self._fit_image(frame, inner_w, inner_h)
        y_offset = inner_y + (inner_h - video.shape[0]) // 2
        x_offset = inner_x + (inner_w - video.shape[1]) // 2
        canvas[y_offset : y_offset + video.shape[0], x_offset : x_offset + video.shape[1]] = video

        cx = x_offset + video.shape[1] // 2
        cy = y_offset + video.shape[0] // 2
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), COLOR_BLUE, 1)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), COLOR_BLUE, 1)

        detection = "person" if detection_info.get("has_person") else "none"
        conf = float(detection_info.get("confidence", 0.0) or 0.0)
        fps = float(detection_info.get("fps", 0.0) or 0.0)
        overlay_lines = [f"FPS: {fps:.1f}", f"Detection: {detection}"]
        if detection_info.get("has_person"):
            overlay_lines.append(f"Conf: {conf:.2f}")
        for index, line in enumerate(overlay_lines):
            self._put_text(canvas, line, x_offset + 12, y_offset + 26 + index * 24, scale=0.58, thickness=2)

    def _draw_status_panel(
        self,
        canvas,
        detection_info: Dict[str, object],
        llm_decision: Dict[str, object],
        safety_info: Dict[str, object],
        robot_status: Dict[str, object],
    ) -> None:
        self._draw_panel(canvas, STATUS_X, STATUS_Y, STATUS_PANEL_WIDTH, STATUS_PANEL_HEIGHT, "Status")
        card_x = STATUS_X + 12
        card_w = STATUS_PANEL_WIDTH - 24
        card_h = 102
        y = STATUS_Y + 38
        self._draw_card(
            canvas,
            "Robot Status",
            [
                ("Connection", robot_status.get("connection", "disconnected")),
                ("Conn Type", robot_status.get("conn_type", "unknown")),
                ("Camera", robot_status.get("camera", "stopped")),
                ("Control Mode", robot_status.get("control_mode", "AI Assist")),
                ("Emergency Stop", robot_status.get("emergency_stop", False)),
            ],
            card_x,
            y,
            card_w,
            card_h,
        )
        y += card_h + 10
        self._draw_card(
            canvas,
            "Detection Status",
            [
                (
                    "Target",
                    f"{detection_info.get('target', 'none')} / {detection_info.get('detected_gesture', 'none')}",
                ),
                ("Confidence", f"{float(detection_info.get('confidence', 0.0) or 0.0):.2f}"),
                ("Position", detection_info.get("horizontal_position", "unknown")),
                ("Vertical", detection_info.get("vertical_position", "unknown")),
                ("Distance", detection_info.get("distance", "unknown")),
                ("Area Ratio", f"{float(detection_info.get('area_ratio', 0.0) or 0.0):.3f}"),
            ],
            card_x,
            y,
            card_w,
            card_h,
        )
        y += card_h + 10
        usage = llm_decision.get("usage") or {}
        usage_text = "N/A"
        if usage:
            usage_text = (
                f"{usage.get('prompt_tokens', 0)} / "
                f"{usage.get('completion_tokens', 0)} / "
                f"{usage.get('total_tokens', 0)}"
            )
        self._draw_card(
            canvas,
            "LLM Decision",
            [
                ("Raw Action", llm_decision.get("action", "stop")),
                ("Reason", truncate_text(llm_decision.get("reason", ""), 38)),
                ("Last LLM Time", llm_decision.get("last_llm_time", "N/A")),
                ("LLM Error", truncate_text(llm_decision.get("llm_error", "none"), 34)),
                ("Usage p/c/t", usage_text),
            ],
            card_x,
            y,
            card_w,
            card_h,
        )
        y += card_h + 10
        cooldown = "active" if safety_info.get("forward_cooldown_active") else "inactive"
        auto_move = "Enabled" if safety_info.get("auto_move_enabled") else "Disabled"
        self._draw_card(
            canvas,
            "Safety Guard",
            [
                ("Safe Action", safety_info.get("safe_action", "stop")),
                ("Safety Reason", truncate_text(safety_info.get("safety_reason", ""), 34)),
                ("Auto Move", auto_move),
                ("Forward Cooldown", cooldown),
                ("Allow Turn", safety_info.get("allow_turn_in_place", True)),
            ],
            card_x,
            y,
            card_w,
            card_h,
        )

    def _draw_log_panel(self, canvas) -> None:
        self._draw_panel(canvas, LOG_X, LOG_Y, LOG_W, LOG_PANEL_HEIGHT, "Event Log")
        y = LOG_Y + 38
        if not self.logs:
            self._put_text(canvas, "No events yet", LOG_X + 14, y, color=COLOR_MUTED)
            return
        for line in self.logs[-MAX_LOG_LINES:]:
            color = COLOR_TEXT
            lower = line.lower()
            if "fail" in lower or "error" in lower or "stop" in lower or "intercept" in lower:
                color = COLOR_YELLOW
            self._put_text(canvas, truncate_text(line, 150), LOG_X + 14, y, scale=0.5, color=color)
            y += 18

    def _draw_panel_title(self, canvas, title: str, x: int, y: int) -> None:
        self._put_text(canvas, title, x + 12, y + 24, scale=0.62, thickness=2)

    def _put_text(
        self,
        canvas,
        text: object,
        x: int,
        y: int,
        scale: float = 0.6,
        thickness: int = 1,
        color=COLOR_TEXT,
    ) -> None:
        cv2.putText(
            canvas,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_panel(self, canvas, x: int, y: int, w: int, h: int, title: str) -> None:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        cv2.line(canvas, (x, y + 30), (x + w, y + 30), COLOR_BORDER, 1)
        self._draw_panel_title(canvas, title, x, y)

    def _draw_card(self, canvas, title: str, rows, x: int, y: int, w: int, h: int) -> None:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_DARK, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        self._put_text(canvas, title, x + 10, y + 20, scale=0.52, thickness=2, color=COLOR_BLUE)
        row_y = y + 40
        for label, value in rows:
            value_text = truncate_text(value, 30)
            value_color = self._status_color(str(value_text))
            self._put_text(canvas, f"{label}:", x + 10, row_y, scale=0.43, color=COLOR_MUTED)
            self._put_text(canvas, value_text, x + 155, row_y, scale=0.43, color=value_color)
            row_y += 15
            if row_y > y + h - 8:
                break

    def _status_color(self, value: str):
        lower = value.lower()
        if "disconnected" in lower or "stopped" in lower:
            return COLOR_RED
        if "stop" in lower or "error" in lower or "failed" in lower or "disabled" in lower:
            return COLOR_RED if "error" in lower or "failed" in lower else COLOR_YELLOW
        if "connected" in lower or "running" in lower or "enabled" in lower or "approved" in lower:
            return COLOR_GREEN
        return COLOR_TEXT

    def _fit_image(self, frame, max_w: int, max_h: int):
        h, w = frame.shape[:2]
        scale = min(max_w / float(w), max_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    renderer = DashboardRenderer()
    renderer.add_log("Dashboard smoke demo started")
    demo_frame = np.full((480, 640, 3), (35, 45, 55), dtype=np.uint8)
    cv2.rectangle(demo_frame, (230, 90), (410, 390), (0, 255, 0), 2)
    cv2.circle(demo_frame, (320, 240), 5, (0, 255, 255), -1)
    cv2.putText(demo_frame, "person 0.88", (230, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    image = renderer.render(
        frame=demo_frame,
        detection_info={
            "has_person": True,
            "target": "person",
            "confidence": 0.88,
            "horizontal_position": "center",
            "vertical_position": "middle",
            "distance": "medium",
            "area_ratio": 0.12,
            "detected_gesture": "open_palm",
            "fps": 24.0,
        },
        llm_decision={
            "action": "forward",
            "reason": "person is far but this is only a suggestion",
            "last_llm_time": "N/A",
            "llm_error": "none",
            "usage": None,
        },
        safety_info={
            "safe_action": "stop",
            "safety_reason": "AUTO_MOVE_ENABLED is False",
            "auto_move_enabled": False,
            "forward_cooldown_active": False,
            "allow_turn_in_place": True,
        },
        robot_status={
            "connection": "disconnected",
            "conn_type": "sta",
            "camera": "stopped",
            "control_mode": "Smoke Demo",
            "emergency_stop": False,
        },
    )
    cv2.imshow("RoboMaster S1 AI Control Dashboard", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
