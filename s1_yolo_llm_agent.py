"""RoboMaster S1 + YOLO11 + local llama.cpp intelligent control demo.

The LLM can only choose from a fixed action whitelist. All speeds are defined
in config.py and are intentionally conservative.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional, Tuple

import cv2
import requests
from robomaster import robot
from ultralytics import YOLO

from config import (
    ALLOW_TURN_IN_PLACE,
    AUTO_MOVE_ENABLED,
    EMERGENCY_STOP,
    FORWARD_COOLDOWN,
    LLM_BASE_URL,
    LLM_INTERVAL_SECONDS,
    LLM_MODEL,
    MAX_FORWARD_DURATION,
    MAX_PERSON_AREA_RATIO_FOR_FORWARD,
    MIN_PERSON_AREA_RATIO_FOR_FORWARD,
    PERSON_CLASS_ID,
    ROBOT_CONN_TYPE,
    SAFE_BACKWARD_SPEED,
    SAFE_FORWARD_SPEED,
    SAFE_GIMBAL_PITCH_SPEED,
    SAFE_GIMBAL_YAW_SPEED,
    SAFE_TURN_SPEED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
)

SYSTEM_PROMPT = """
你是 RoboMaster S1 机器人控制助手。
你只能输出 JSON，不要输出 Markdown，不要解释。
你只能根据 YOLO 检测结果给机器人一个动作建议，最终动作会由 Safety Guard 安全层判断。

可选 action 只能是：

* stop
* gimbal_left
* gimbal_right
* gimbal_up
* gimbal_down
* turn_left
* turn_right
* forward
* backward

规则：

1. 如果没有检测到目标，action=stop。
2. 如果检测到 person，并且人在画面左侧，优先 gimbal_left。
3. 如果检测到 person，并且人在画面右侧，优先 gimbal_right。
4. 如果检测到 person，并且人在画面上方，优先 gimbal_up。
5. 如果检测到 person，并且人在画面下方，优先 gimbal_down。
6. 如果 person 在画面中间但框很小，说明距离较远，可以 forward。
7. 如果 person 框很大，说明距离较近，应该 stop。
8. 优先 stop 或云台调整，不要连续 forward。
9. 不要尝试撞击、冲撞或追逐目标。
10. 摄像头和 YOLO 不能可靠判断墙面距离，如果不确定应该 stop。
11. 只能返回 JSON，例如：
   {"action":"gimbal_left","reason":"person is on the left side"}
"""

ALLOWED_ACTIONS = {
    "stop",
    "gimbal_left",
    "gimbal_right",
    "gimbal_up",
    "gimbal_down",
    "turn_left",
    "turn_right",
    "forward",
    "backward",
}

Box = Tuple[float, float, float, float, float]


def select_best_person(results) -> Optional[Box]:
    """Return (x1, y1, x2, y2, confidence) for the best person detection."""
    if not results or results[0].boxes is None:
        return None

    best = None
    best_conf = -1.0
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        if class_id != PERSON_CLASS_ID or confidence < best_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        best = (x1, y1, x2, y2, confidence)
        best_conf = confidence
    return best


def classify_scene(person_box: Optional[Box], frame_width: int, frame_height: int) -> Dict[str, object]:
    if person_box is None:
        return {"has_person": False}

    x1, y1, x2, y2, confidence = person_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)

    if center_x < frame_width * 0.4:
        horizontal = "left"
    elif center_x > frame_width * 0.6:
        horizontal = "right"
    else:
        horizontal = "center"

    if center_y < frame_height * 0.4:
        vertical = "upper"
    elif center_y > frame_height * 0.6:
        vertical = "lower"
    else:
        vertical = "middle"

    if area_ratio < 0.08:
        distance = "far"
    elif area_ratio > 0.35:
        distance = "near"
    else:
        distance = "medium"

    return {
        "has_person": True,
        "confidence": confidence,
        "horizontal": horizontal,
        "vertical": vertical,
        "distance": distance,
        "area_ratio": area_ratio,
    }


class SafetyGuard:
    """Filter LLM action suggestions before they reach the robot.

    The LLM only suggests actions. This guard owns the final decision for
    chassis movement so accidental repeated forward commands cannot keep the
    robot driving into a wall.
    """

    def __init__(self) -> None:
        self.last_forward_time = 0.0
        self.forward_start_time = 0.0
        self.is_forwarding = False

    def stop_forward_tracking(self) -> None:
        self.is_forwarding = False
        self.forward_start_time = 0.0

    def filter_action(self, raw_action: str, scene_info: Dict[str, object]) -> Tuple[str, str]:
        """Return (safe_action, reason) for a raw LLM action suggestion."""
        if EMERGENCY_STOP:
            self.stop_forward_tracking()
            return "stop", "EMERGENCY_STOP=True"

        if raw_action not in ALLOWED_ACTIONS:
            self.stop_forward_tracking()
            return "stop", f"Invalid action '{raw_action}' not in whitelist"

        current_time = time.time()

        if raw_action == "stop":
            self.stop_forward_tracking()
            return "stop", "LLM suggested stop"

        if not AUTO_MOVE_ENABLED and raw_action in ("forward", "backward"):
            self.stop_forward_tracking()
            return "stop", "AUTO_MOVE_ENABLED is False, forward/backward disabled"

        if not scene_info.get("has_person", False):
            self.stop_forward_tracking()
            return "stop", "No person detected, stopping for safety"

        if raw_action == "forward":
            area_ratio = float(scene_info.get("area_ratio", 0.0))

            if area_ratio < MIN_PERSON_AREA_RATIO_FOR_FORWARD:
                self.stop_forward_tracking()
                return "stop", (
                    f"Person area too small for safe forward "
                    f"({area_ratio:.3f} < {MIN_PERSON_AREA_RATIO_FOR_FORWARD})"
                )

            if area_ratio > MAX_PERSON_AREA_RATIO_FOR_FORWARD:
                self.stop_forward_tracking()
                return "stop", (
                    f"Person too close for forward "
                    f"({area_ratio:.3f} > {MAX_PERSON_AREA_RATIO_FOR_FORWARD})"
                )

            if current_time - self.last_forward_time < FORWARD_COOLDOWN:
                self.stop_forward_tracking()
                remaining = FORWARD_COOLDOWN - (current_time - self.last_forward_time)
                return "stop", f"Forward cooldown active ({remaining:.1f}s remaining)"

            if not self.is_forwarding:
                self.is_forwarding = True
                self.forward_start_time = current_time
                self.last_forward_time = current_time

            if current_time - self.forward_start_time > MAX_FORWARD_DURATION:
                self.stop_forward_tracking()
                return "stop", f"Maximum forward duration ({MAX_FORWARD_DURATION}s) exceeded"

            return "forward", "Forward approved by Safety Guard"

        if raw_action == "backward":
            self.stop_forward_tracking()
            return "stop", "Backward movement disabled for safety"

        if raw_action in ("turn_left", "turn_right") and not ALLOW_TURN_IN_PLACE:
            self.stop_forward_tracking()
            return "stop", "In-place turning disabled"

        if raw_action != "forward":
            self.stop_forward_tracking()

        return raw_action, "Action approved by Safety Guard"


def build_scene_text(scene: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    if not scene.get("has_person"):
        scene_text = """YOLO检测结果：

* 没有检测到 person
请输出机器人动作 JSON。"""
        scene_info = {
            "has_person": False,
            "area_ratio": 0.0,
            "horizontal_position": "none",
            "vertical_position": "none",
            "distance": "none",
        }
        return scene_text, scene_info

    scene_text = f"""YOLO检测结果：

* 目标类别: person
* 置信度: {float(scene["confidence"]):.2f}
* 水平位置: {scene["horizontal"]}
* 垂直位置: {scene["vertical"]}
* 距离估计: {scene["distance"]}
* 目标框面积占比: {float(scene["area_ratio"]):.3f}
请输出机器人动作 JSON。"""

    scene_info = {
        "has_person": True,
        "area_ratio": scene["area_ratio"],
        "horizontal_position": scene["horizontal"],
        "vertical_position": scene["vertical"],
        "distance": scene["distance"],
    }
    return scene_text, scene_info


def extract_json(text: str) -> Dict[str, str]:
    """Parse a JSON object, or extract the first {...} block from noisy text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return {"action": "stop", "reason": "LLM output parse failed"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"action": "stop", "reason": "LLM output parse failed"}

    if not isinstance(data, dict):
        return {"action": "stop", "reason": "LLM output parse failed"}

    action = str(data.get("action", "stop"))
    reason = str(data.get("reason", "no reason"))
    if action not in ALLOWED_ACTIONS:
        return {"action": "stop", "reason": f"unknown action from LLM: {action}"}
    return {"action": action, "reason": reason}


def ask_llm(scene_text: str) -> Dict[str, str]:
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scene_text},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }

    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        return {"action": "stop", "reason": f"LLM request failed: {exc}"}

    return extract_json(text)


def execute_action(ep_robot: robot.Robot, action: str) -> None:
    """Execute a whitelisted low-speed action."""
    chassis = ep_robot.chassis
    gimbal = ep_robot.gimbal

    if action == "stop":
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    elif action == "gimbal_left":
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=0, yaw_speed=-SAFE_GIMBAL_YAW_SPEED)
    elif action == "gimbal_right":
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=0, yaw_speed=SAFE_GIMBAL_YAW_SPEED)
    elif action == "gimbal_up":
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=SAFE_GIMBAL_PITCH_SPEED, yaw_speed=0)
    elif action == "gimbal_down":
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=-SAFE_GIMBAL_PITCH_SPEED, yaw_speed=0)
    elif action == "turn_left":
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        chassis.drive_speed(x=0, y=0, z=-SAFE_TURN_SPEED)
    elif action == "turn_right":
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        chassis.drive_speed(x=0, y=0, z=SAFE_TURN_SPEED)
    elif action == "forward":
        chassis.drive_speed(x=SAFE_FORWARD_SPEED, y=0, z=0)
        time.sleep(MAX_FORWARD_DURATION)
        chassis.drive_speed(x=0, y=0, z=0)
    elif action == "backward":
        chassis.drive_speed(x=SAFE_BACKWARD_SPEED, y=0, z=0)
        time.sleep(MAX_FORWARD_DURATION)
        chassis.drive_speed(x=0, y=0, z=0)
    else:
        chassis.drive_speed(x=0, y=0, z=0)
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)


def draw_overlay(
    frame,
    person_box: Optional[Box],
    scene: Dict[str, object],
    decision: Dict[str, str],
) -> None:
    if person_box is not None:
        x1, y1, x2, y2, confidence = person_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    raw_action = decision.get("raw_action", decision.get("action", "stop"))
    safe_action = decision.get("safe_action", decision.get("action", "stop"))
    action_text = f"raw: {raw_action}  safe: {safe_action}"
    reason_text = f"safety: {decision.get('safety_reason', decision.get('reason', ''))[:90]}"
    cv2.putText(frame, action_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, reason_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    if scene.get("has_person"):
        scene_text = (
            f"{scene['horizontal']} / {scene['vertical']} / "
            f"{scene['distance']} / area={float(scene['area_ratio']):.3f}"
        )
    else:
        scene_text = "no person"
    cv2.putText(frame, scene_text, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)


def main() -> int:
    ep_robot = robot.Robot()
    camera = None
    last_llm_time = 0.0
    safety_guard = SafetyGuard()
    decision = {
        "raw_action": "stop",
        "safe_action": "stop",
        "reason": "starting",
        "safety_reason": "starting",
    }

    try:
        print(f"Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)

        print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
        ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
        camera = ep_robot.camera

        print("Starting video stream and recentering gimbal...")
        camera.start_video_stream(display=False)
        ep_robot.gimbal.recenter().wait_for_completed()
        execute_action(ep_robot, "stop")

        print("Press q to quit.")
        while True:
            frame = camera.read_cv2_image(strategy="newest", timeout=5)
            if frame is None:
                print("[WARN] No camera frame received.")
                execute_action(ep_robot, "stop")
                continue

            height, width = frame.shape[:2]
            results = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )
            person_box = select_best_person(results)
            scene = classify_scene(person_box, width, height)

            now = time.time()
            if now - last_llm_time >= LLM_INTERVAL_SECONDS:
                scene_text, scene_info = build_scene_text(scene)
                current_decision = ask_llm(scene_text)
                raw_action = current_decision.get("action", "stop")
                safe_action, safety_reason = safety_guard.filter_action(raw_action, scene_info)
                execute_action(ep_robot, safe_action)
                decision = {
                    "raw_action": raw_action,
                    "safe_action": safe_action,
                    "reason": current_decision.get("reason", ""),
                    "safety_reason": safety_reason,
                }
                last_llm_time = now

            draw_overlay(frame, person_box, scene, decision)
            cv2.imshow("S1 YOLO11 LLM Agent", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 0
    except Exception as exc:
        print(f"[FAIL] LLM agent failed: {exc}")
        return 1
    finally:
        try:
            execute_action(ep_robot, "stop")
        except Exception as exc:
            print(f"[WARN] robot stop failed: {exc}")
        if camera is not None:
            try:
                camera.stop_video_stream()
            except Exception as exc:
                print(f"[WARN] camera.stop_video_stream failed: {exc}")
        try:
            ep_robot.close()
        except Exception as exc:
            print(f"[WARN] ep_robot.close failed: {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
