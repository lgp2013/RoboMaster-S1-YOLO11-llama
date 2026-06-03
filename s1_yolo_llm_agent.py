"""RoboMaster S1 + YOLO11 + local llama.cpp intelligent control demo.

The LLM can only choose from a fixed action whitelist. All speeds are defined
in config.py and are intentionally conservative.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import requests
from robomaster import robot
from ultralytics import YOLO

from config import (
    ALLOW_TURN_IN_PLACE,
    AUTO_MOVE_ENABLED,
    EMERGENCY_STOP,
    ENABLE_DASHBOARD,
    ENABLE_GESTURE_DETECTION,
    GESTURE_ACTION_ENABLED,
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
from dashboard import DashboardRenderer
from gesture_action_mapper import map_gesture_to_action
from gesture_detector import GestureDetector, GestureResult

SYSTEM_PROMPT = """
你是 RoboMaster S1 机器人控制助手。
你只能输出 JSON，不要输出 Markdown，不要解释。
你只能根据 YOLO 检测结果给机器人一个动作建议，最终动作会由 Safety Guard 安全层判断。
你会收到 detected_gesture 字段。手势也是建议来源，最终动作仍由 Safety Guard 判断。

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
11. 如果 detected_gesture 是 open_palm 或 fist，优先建议 stop。
12. 如果 detected_gesture 是 point_left/right，优先建议 gimbal_left/right。
13. 只能返回 JSON，例如：
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
        self.auto_move_enabled = AUTO_MOVE_ENABLED

    def stop_forward_tracking(self) -> None:
        self.is_forwarding = False
        self.forward_start_time = 0.0

    def is_forward_cooldown_active(self) -> bool:
        return time.time() - self.last_forward_time < FORWARD_COOLDOWN

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

        if not self.auto_move_enabled and raw_action in ("forward", "backward"):
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


def build_scene_text(scene: Dict[str, object], detected_gesture: str = "none") -> Tuple[str, Dict[str, object]]:
    if not scene.get("has_person"):
        scene_text = """YOLO检测结果：

* 没有检测到 person
* detected_gesture: {gesture}
请输出机器人动作 JSON。"""
        scene_text = scene_text.format(gesture=detected_gesture)
        scene_info = {
            "has_person": False,
            "area_ratio": 0.0,
            "horizontal_position": "none",
            "vertical_position": "none",
            "distance": "none",
            "detected_gesture": detected_gesture,
        }
        return scene_text, scene_info

    scene_text = f"""YOLO检测结果：

* 目标类别: person
* 置信度: {float(scene["confidence"]):.2f}
* 水平位置: {scene["horizontal"]}
* 垂直位置: {scene["vertical"]}
* 距离估计: {scene["distance"]}
* 目标框面积占比: {float(scene["area_ratio"]):.3f}
* detected_gesture: {detected_gesture}
请输出机器人动作 JSON。"""

    scene_info = {
        "has_person": True,
        "area_ratio": scene["area_ratio"],
        "horizontal_position": scene["horizontal"],
        "vertical_position": scene["vertical"],
        "distance": scene["distance"],
        "detected_gesture": detected_gesture,
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


def ask_llm(scene_text: str) -> Dict[str, object]:
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
        return {
            "action": "stop",
            "reason": f"LLM request failed: {exc}",
            "last_llm_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_error": str(exc),
            "usage": None,
        }

    decision = extract_json(text)
    decision["last_llm_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decision["llm_error"] = "none"
    decision["usage"] = data.get("usage")
    return decision


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


def draw_detection_overlay(frame, person_box: Optional[Box], gesture_result: GestureResult, gesture_detector=None) -> None:
    """Draw only essential YOLO target visuals on the video frame."""
    if person_box is not None:
        x1, y1, x2, y2, confidence = person_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"person {confidence:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    if gesture_detector is not None:
        gesture_detector.draw(frame, gesture_result)


def build_detection_info(scene: Dict[str, object], fps: float, gesture_result: GestureResult) -> Dict[str, object]:
    has_person = bool(scene.get("has_person"))
    return {
        "has_person": has_person,
        "target": "person" if has_person else "none",
        "confidence": float(scene.get("confidence", 0.0) or 0.0),
        "horizontal_position": scene.get("horizontal", "unknown") if has_person else "unknown",
        "vertical_position": scene.get("vertical", "unknown") if has_person else "unknown",
        "distance": scene.get("distance", "unknown") if has_person else "unknown",
        "area_ratio": float(scene.get("area_ratio", 0.0) or 0.0),
        "detected_gesture": gesture_result.gesture,
        "gesture_confidence": gesture_result.confidence,
        "handedness": gesture_result.handedness,
        "fps": fps,
    }


def build_safety_info(safety_guard: SafetyGuard, safe_action: str, safety_reason: str) -> Dict[str, object]:
    return {
        "safe_action": safe_action,
        "safety_reason": safety_reason,
        "auto_move_enabled": safety_guard.auto_move_enabled,
        "forward_cooldown_active": safety_guard.is_forward_cooldown_active(),
        "allow_turn_in_place": ALLOW_TURN_IN_PLACE,
    }


def build_robot_status(connection: str, camera_state: str, control_mode: str) -> Dict[str, object]:
    return {
        "connection": connection,
        "conn_type": ROBOT_CONN_TYPE,
        "camera": camera_state,
        "control_mode": control_mode,
        "emergency_stop": EMERGENCY_STOP,
    }


def main() -> int:
    ep_robot = robot.Robot()
    camera = None
    last_llm_time = 0.0
    safety_guard = SafetyGuard()
    dashboard = DashboardRenderer()
    gesture_detector = None
    dashboard.add_log("Program started")
    llm_decision: Dict[str, object] = {
        "action": "stop",
        "reason": "starting",
        "last_llm_time": "N/A",
        "llm_error": "none",
        "usage": None,
    }
    safe_action = "stop"
    safety_reason = "starting"
    previous_safe_action = "stop"
    frame_count = 0
    last_fps_time = time.time()
    current_fps = 0.0
    connection_state = "disconnected"
    camera_state = "stopped"
    control_mode = "AI Assist"

    try:
        print(f"Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)
        dashboard.add_log("YOLO model loaded")
        if ENABLE_GESTURE_DETECTION:
            gesture_detector = GestureDetector()
            dashboard.add_log("MediaPipe Hands loaded")

        print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
        ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
        connection_state = "connected"
        dashboard.add_log("S1 connected")
        camera = ep_robot.camera

        print("Starting video stream and recentering gimbal...")
        camera.start_video_stream(display=False)
        camera_state = "running"
        dashboard.add_log("Camera started")
        ep_robot.gimbal.recenter().wait_for_completed()
        execute_action(ep_robot, "stop")

        window_name = "RoboMaster S1 AI Control Dashboard"
        print("Press q to quit. Press s for safe stop. Press m to toggle auto move. Press r to recenter gimbal.")
        while True:
            frame = camera.read_cv2_image(strategy="newest", timeout=5)
            if frame is None:
                print("[WARN] No camera frame received.")
                dashboard.add_log("No camera frame received")
                execute_action(ep_robot, "stop")
                continue

            frame_count += 1
            now_for_fps = time.time()
            elapsed_for_fps = now_for_fps - last_fps_time
            if elapsed_for_fps >= 1.0:
                current_fps = frame_count / elapsed_for_fps
                frame_count = 0
                last_fps_time = now_for_fps

            height, width = frame.shape[:2]
            results = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )
            person_box = select_best_person(results)
            scene = classify_scene(person_box, width, height)
            if gesture_detector is not None:
                gesture_result = gesture_detector.detect(frame)
            else:
                gesture_result = GestureResult()
            detection_info = build_detection_info(scene, current_fps, gesture_result)

            now = time.time()
            if now - last_llm_time >= LLM_INTERVAL_SECONDS:
                scene_text, scene_info = build_scene_text(scene, gesture_result.gesture)
                current_decision = ask_llm(scene_text)
                gesture_action, gesture_reason = map_gesture_to_action(gesture_result.gesture)
                if GESTURE_ACTION_ENABLED and gesture_action is not None:
                    raw_action = gesture_action
                    current_decision = {
                        **current_decision,
                        "action": gesture_action,
                        "reason": f"{gesture_reason}; llm_reason={current_decision.get('reason', '')}",
                    }
                    dashboard.add_log(f"gesture {gesture_result.gesture} -> {gesture_action}")
                else:
                    raw_action = current_decision.get("action", "stop")
                safe_action, safety_reason = safety_guard.filter_action(str(raw_action), scene_info)
                execute_action(ep_robot, safe_action)
                llm_decision = current_decision
                if current_decision.get("llm_error") == "none":
                    dashboard.add_log(f"LLM ok raw_action={raw_action}")
                else:
                    dashboard.add_log(f"LLM failed: {current_decision.get('llm_error')}")
                if raw_action == "forward" and safe_action == "stop":
                    dashboard.add_log(f"SafetyGuard intercepted forward: {safety_reason}")
                if safe_action != previous_safe_action:
                    dashboard.add_log(f"action changed: {previous_safe_action} -> {safe_action}")
                    previous_safe_action = safe_action
                last_llm_time = now

            annotated = frame.copy()
            draw_detection_overlay(annotated, person_box, gesture_result, gesture_detector)
            safety_info = build_safety_info(safety_guard, safe_action, safety_reason)
            robot_status = build_robot_status(connection_state, camera_state, control_mode)
            dashboard_img = dashboard.render(
                frame=annotated,
                detection_info=detection_info,
                llm_decision=llm_decision,
                safety_info=safety_info,
                robot_status=robot_status,
            )
            cv2.imshow(window_name if ENABLE_DASHBOARD else "S1 YOLO11 LLM Agent", dashboard_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                execute_action(ep_robot, "stop")
                safety_guard.stop_forward_tracking()
                safe_action = "stop"
                safety_reason = "Emergency safe stop triggered by user"
                llm_decision = {
                    **llm_decision,
                    "action": llm_decision.get("action", "stop"),
                    "reason": llm_decision.get("reason", ""),
                }
                control_mode = "Safe Stop"
                dashboard.add_log("Emergency safe stop triggered by user")
            elif key == ord("m"):
                safety_guard.auto_move_enabled = not safety_guard.auto_move_enabled
                dashboard.add_log(f"AUTO_MOVE toggled to {safety_guard.auto_move_enabled}")
                control_mode = "AI Assist" if safety_guard.auto_move_enabled else "AI Assist Safe"
            elif key == ord("r"):
                execute_action(ep_robot, "stop")
                ep_robot.gimbal.recenter().wait_for_completed(timeout=3)
                dashboard.add_log("Gimbal recentered by user")
            elif key == ord("q"):
                dashboard.add_log("User pressed q, exiting")
                break

        return 0
    except KeyboardInterrupt:
        dashboard.add_log("Interrupted by user")
        print("Interrupted by user.")
        return 0
    except Exception as exc:
        dashboard.add_log(f"Program exception: {exc}")
        print(f"[FAIL] LLM agent failed: {exc}")
        return 1
    finally:
        try:
            execute_action(ep_robot, "stop")
            dashboard.add_log("finally stop executed")
        except Exception as exc:
            print(f"[WARN] robot stop failed: {exc}")
        if camera is not None:
            try:
                camera.stop_video_stream()
            except Exception as exc:
                print(f"[WARN] camera.stop_video_stream failed: {exc}")
        if gesture_detector is not None:
            try:
                gesture_detector.close()
            except Exception as exc:
                print(f"[WARN] gesture detector close failed: {exc}")
        try:
            ep_robot.close()
        except Exception as exc:
            print(f"[WARN] ep_robot.close failed: {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
