"""YOLO11 person detection with safe RoboMaster S1 gimbal following.

This demo does not use the LLM. It only tracks the highest-confidence person
and gently steers the gimbal toward the target center.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
from robomaster import robot
from ultralytics import YOLO

from config import (
    PERSON_CLASS_ID,
    ROBOT_CONN_TYPE,
    SAFE_GIMBAL_PITCH_SPEED,
    SAFE_GIMBAL_YAW_SPEED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
)

Box = Tuple[float, float, float, float, float]

X_DEAD_ZONE = 40
Y_DEAD_ZONE = 35
CONTROL_INTERVAL_SECONDS = 0.15


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


def update_gimbal(gimbal, person_box: Optional[Box], frame_width: int, frame_height: int) -> str:
    """Drive the gimbal toward the person center and return a short status."""
    if person_box is None:
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        return "no person: gimbal stop"

    x1, y1, x2, y2, confidence = person_box
    target_x = (x1 + x2) / 2
    target_y = (y1 + y2) / 2
    offset_x = target_x - frame_width / 2
    offset_y = target_y - frame_height / 2

    yaw_speed = 0
    pitch_speed = 0

    if abs(offset_x) >= X_DEAD_ZONE:
        yaw_speed = SAFE_GIMBAL_YAW_SPEED if offset_x > 0 else -SAFE_GIMBAL_YAW_SPEED
    if abs(offset_y) >= Y_DEAD_ZONE:
        pitch_speed = -SAFE_GIMBAL_PITCH_SPEED if offset_y > 0 else SAFE_GIMBAL_PITCH_SPEED

    gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
    return f"person conf={confidence:.2f}, yaw={yaw_speed}, pitch={pitch_speed}"


def draw_status(frame, person_box: Optional[Box], status: str) -> None:
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
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)


def main() -> int:
    ep_robot = robot.Robot()
    camera = None
    gimbal = None
    last_control_time = 0.0

    try:
        print(f"Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)

        print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
        ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
        camera = ep_robot.camera
        gimbal = ep_robot.gimbal

        print("Starting video stream and recentering gimbal...")
        camera.start_video_stream(display=False)
        gimbal.recenter().wait_for_completed()

        print("Press q to quit.")
        status = "starting"
        while True:
            frame = camera.read_cv2_image(strategy="newest", timeout=5)
            if frame is None:
                print("[WARN] No camera frame received.")
                continue

            height, width = frame.shape[:2]
            results = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )
            person_box = select_best_person(results)

            now = time.time()
            if now - last_control_time >= CONTROL_INTERVAL_SECONDS:
                status = update_gimbal(gimbal, person_box, width, height)
                last_control_time = now

            draw_status(frame, person_box, status)
            cv2.imshow("S1 YOLO11 Gimbal Follow", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 0
    except Exception as exc:
        print(f"[FAIL] Gimbal follow failed: {exc}")
        return 1
    finally:
        if gimbal is not None:
            try:
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception as exc:
                print(f"[WARN] gimbal stop failed: {exc}")
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
