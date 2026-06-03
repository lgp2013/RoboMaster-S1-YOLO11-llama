"""Web debug panel for RoboMaster S1 camera, YOLO detection, and target lock.

This script uses only the Python standard library for HTTP serving.

Run from this directory:
    ..\\.venv\\Scripts\\python.exe .\\s1_web_debug.py

Then open:
    http://127.0.0.1:5000

The page can preview the camera stream, print YOLO detections, manually control
the gimbal, lock one detected person by snapshot, and follow that person at low
speed. It never controls the blaster.
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from robomaster import robot
from ultralytics import YOLO

from config import (
    LOCKED_PERSON_DIR,
    PERSON_CLASS_ID,
    ROBOT_CONN_TYPE,
    SAFE_BACKWARD_SPEED,
    SAFE_FORWARD_SPEED,
    SAFE_GIMBAL_PITCH_SPEED,
    SAFE_GIMBAL_YAW_SPEED,
    TARGET_CENTER_DEADZONE_X,
    TARGET_CENTER_DEADZONE_Y,
    TARGET_FAR_AREA_RATIO,
    TARGET_FOLLOW_CONTROL_INTERVAL_SECONDS,
    TARGET_NEAR_AREA_RATIO,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
)


HOST = "127.0.0.1"
PORT = 5000

state_lock = threading.RLock()
ep_robot: Optional[robot.Robot] = None
camera = None
gimbal = None
chassis = None
yolo_model: Optional[YOLO] = None

latest_raw_frame = None
latest_annotated_frame = None
latest_detections: List[Dict[str, object]] = []
latest_target_detection: Optional[Dict[str, object]] = None

target_state: Dict[str, object] = {
    "locked": False,
    "auto_follow": False,
    "locked_at": "",
    "snapshot_path": "",
    "crop_path": "",
    "last_match_score": 0.0,
    "last_seen_age_seconds": None,
}
target_histogram = None
last_target_seen_time = 0.0
last_control_time = 0.0

status = {
    "robot_connected": False,
    "video_running": False,
    "yolo_loaded": False,
    "last_error": "",
    "last_action": "idle",
}

stop_event = threading.Event()
capture_thread: Optional[threading.Thread] = None


def set_error(message: str) -> None:
    with state_lock:
        status["last_error"] = message
    if message:
        print(f"[WARN] {message}")


def load_yolo() -> None:
    global yolo_model
    if yolo_model is not None:
        return
    print(f"Loading YOLO model: {YOLO_MODEL}")
    yolo_model = YOLO(YOLO_MODEL)
    with state_lock:
        status["yolo_loaded"] = True


def connect_robot() -> None:
    """Connect the robot, start video stream, and recenter the gimbal."""
    global ep_robot, camera, gimbal, chassis

    if ep_robot is not None:
        return

    print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
    camera = ep_robot.camera
    gimbal = ep_robot.gimbal
    chassis = ep_robot.chassis

    print("Starting video stream...")
    camera.start_video_stream(display=False)

    try:
        gimbal.recenter().wait_for_completed(timeout=3)
    except Exception as exc:
        set_error(f"gimbal recenter failed: {exc}")

    with state_lock:
        status["robot_connected"] = True
        status["video_running"] = True


def safe_gimbal_stop() -> None:
    if gimbal is None:
        return
    try:
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    except Exception as exc:
        set_error(f"gimbal stop failed: {exc}")


def safe_chassis_stop() -> None:
    if chassis is None:
        return
    try:
        chassis.drive_speed(x=0, y=0, z=0)
    except Exception as exc:
        set_error(f"chassis stop failed: {exc}")


def safe_robot_stop() -> None:
    safe_chassis_stop()
    safe_gimbal_stop()


def cleanup() -> None:
    """Stop robot resources safely."""
    global ep_robot, camera, gimbal, chassis, capture_thread

    stop_event.set()
    safe_robot_stop()

    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=1.5)
    capture_thread = None

    if camera is not None:
        try:
            camera.stop_video_stream()
        except Exception as exc:
            print(f"[WARN] camera.stop_video_stream failed: {exc}")

    if ep_robot is not None:
        try:
            ep_robot.close()
        except Exception as exc:
            print(f"[WARN] ep_robot.close failed: {exc}")

    with state_lock:
        status["robot_connected"] = False
        status["video_running"] = False
        target_state["auto_follow"] = False

    ep_robot = None
    camera = None
    gimbal = None
    chassis = None


def clamp_box(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def crop_detection(frame, detection: Dict[str, object]):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(detection["bbox"], width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def build_color_histogram(image) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def histogram_score(image) -> float:
    global target_histogram
    if target_histogram is None:
        return 0.0
    candidate_hist = build_color_histogram(image)
    if candidate_hist is None:
        return 0.0
    score = cv2.compareHist(target_histogram, candidate_hist, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, float(score)))


def parse_detections(results, frame_width: int, frame_height: int) -> List[Dict[str, object]]:
    detections: List[Dict[str, object]] = []
    if not results or results[0].boxes is None:
        return detections

    names = results[0].names
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)

        if isinstance(names, dict):
            class_name = names.get(cls_id, str(cls_id))
        else:
            class_name = str(cls_id)

        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": [round(center_x, 1), round(center_y, 1)],
                "area_ratio": round(area_ratio, 4),
                "target_score": 0.0,
                "is_locked_target": False,
            }
        )

    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return detections


def find_best_person(detections: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    people = [item for item in detections if int(item["class_id"]) == PERSON_CLASS_ID]
    if not people:
        return None
    return max(people, key=lambda item: float(item["confidence"]))


def find_locked_target(frame, detections: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    people = [item for item in detections if int(item["class_id"]) == PERSON_CLASS_ID]
    if not people:
        return None

    if not bool(target_state["locked"]):
        return find_best_person(detections)

    best_item = None
    best_score = -1.0
    for item in people:
        crop = crop_detection(frame, item)
        appearance_score = histogram_score(crop)
        confidence = float(item["confidence"])
        score = appearance_score * 0.7 + confidence * 0.3
        item["target_score"] = round(score, 3)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None:
        return None
    best_item["is_locked_target"] = True
    with state_lock:
        target_state["last_match_score"] = round(best_score, 3)
    return best_item


def draw_target_overlay(frame, target: Optional[Dict[str, object]]) -> None:
    if target is None:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(target["bbox"], width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 3)
    label = f"LOCKED person score={target.get('target_score', 0)}"
    cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120), 2)


def execute_follow_control(target: Optional[Dict[str, object]], width: int, height: int) -> None:
    """Follow locked target with low-speed gimbal and chassis commands."""
    global last_control_time

    now = time.time()
    if now - last_control_time < TARGET_FOLLOW_CONTROL_INTERVAL_SECONDS:
        return
    last_control_time = now

    if not bool(target_state["auto_follow"]):
        return

    if target is None:
        safe_robot_stop()
        with state_lock:
            status["last_action"] = "target_lost_stop"
        return

    center_x, center_y = [float(v) for v in target["center"]]
    area_ratio = float(target["area_ratio"])
    offset_x = center_x - width / 2
    offset_y = center_y - height / 2

    yaw_speed = 0
    pitch_speed = 0
    if offset_x < -TARGET_CENTER_DEADZONE_X:
        yaw_speed = -SAFE_GIMBAL_YAW_SPEED
    elif offset_x > TARGET_CENTER_DEADZONE_X:
        yaw_speed = SAFE_GIMBAL_YAW_SPEED

    if offset_y < -TARGET_CENTER_DEADZONE_Y:
        pitch_speed = SAFE_GIMBAL_PITCH_SPEED
    elif offset_y > TARGET_CENTER_DEADZONE_Y:
        pitch_speed = -SAFE_GIMBAL_PITCH_SPEED

    forward_speed = 0
    if abs(offset_x) < width * 0.2:
        if area_ratio < TARGET_FAR_AREA_RATIO:
            forward_speed = SAFE_FORWARD_SPEED
        elif area_ratio > TARGET_NEAR_AREA_RATIO:
            forward_speed = SAFE_BACKWARD_SPEED

    try:
        if gimbal is not None:
            gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
        if chassis is not None:
            chassis.drive_speed(x=forward_speed, y=0, z=0)
        with state_lock:
            status["last_action"] = (
                f"follow yaw={yaw_speed} pitch={pitch_speed} x={forward_speed:.2f} "
                f"area={area_ratio:.3f}"
            )
    except Exception as exc:
        safe_robot_stop()
        set_error(f"follow control failed: {exc}")


def capture_loop() -> None:
    """Read camera frames, run YOLO, and store the newest annotated frame."""
    global latest_raw_frame, latest_annotated_frame, latest_detections
    global latest_target_detection, last_target_seen_time

    while not stop_event.is_set():
        try:
            if camera is None or yolo_model is None:
                time.sleep(0.1)
                continue

            frame = camera.read_cv2_image(strategy="newest", timeout=5)
            if frame is None:
                set_error("camera frame timeout")
                continue

            results = yolo_model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )
            annotated = results[0].plot()
            height, width = frame.shape[:2]
            detections = parse_detections(results, width, height)
            target = find_locked_target(frame, detections)
            draw_target_overlay(annotated, target)

            if target is not None:
                last_target_seen_time = time.time()

            execute_follow_control(target, width, height)

            with state_lock:
                latest_raw_frame = frame.copy()
                latest_annotated_frame = annotated
                latest_detections = detections[:20]
                latest_target_detection = dict(target) if target is not None else None
                if last_target_seen_time:
                    target_state["last_seen_age_seconds"] = round(time.time() - last_target_seen_time, 2)
                else:
                    target_state["last_seen_age_seconds"] = None
                status["last_error"] = ""

        except Exception as exc:
            safe_robot_stop()
            set_error(f"capture loop failed: {exc}")
            time.sleep(0.5)


def ensure_started() -> None:
    global capture_thread
    load_yolo()
    connect_robot()
    if capture_thread is None or not capture_thread.is_alive():
        stop_event.clear()
        capture_thread = threading.Thread(target=capture_loop, name="s1-capture", daemon=True)
        capture_thread.start()


def encode_latest_frame() -> Optional[bytes]:
    with state_lock:
        frame = latest_annotated_frame
    if frame is None:
        return None

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buffer.tobytes()


def get_status_payload() -> Dict[str, object]:
    with state_lock:
        return {
            "status": dict(status),
            "detections": list(latest_detections),
            "target": dict(target_state),
            "target_detection": dict(latest_target_detection) if latest_target_detection else None,
        }


def lock_current_person(enable_follow: bool = True) -> Dict[str, object]:
    """Lock the current best person by saving a full snapshot and person crop."""
    global target_histogram

    with state_lock:
        frame = None if latest_raw_frame is None else latest_raw_frame.copy()
        detections = list(latest_detections)

    if frame is None:
        raise RuntimeError("no camera frame available")

    person = find_best_person(detections)
    if person is None:
        raise RuntimeError("no person detected to lock")

    crop = crop_detection(frame, person)
    if crop is None or crop.size == 0:
        raise RuntimeError("person crop is empty")

    target_histogram = build_color_histogram(crop)
    if target_histogram is None:
        raise RuntimeError("failed to build target histogram")

    output_dir = Path(__file__).with_name(LOCKED_PERSON_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = output_dir / f"locked_person_{timestamp}_frame.jpg"
    crop_path = output_dir / f"locked_person_{timestamp}_crop.jpg"
    cv2.imwrite(str(snapshot_path), frame)
    cv2.imwrite(str(crop_path), crop)

    with state_lock:
        target_state.update(
            {
                "locked": True,
                "auto_follow": enable_follow,
                "locked_at": timestamp,
                "snapshot_path": str(snapshot_path),
                "crop_path": str(crop_path),
                "last_match_score": 1.0,
                "last_seen_age_seconds": 0,
            }
        )
        status["last_action"] = "target_locked_follow_on" if enable_follow else "target_locked"

    return get_status_payload()


def unlock_target() -> None:
    global target_histogram, latest_target_detection
    safe_robot_stop()
    target_histogram = None
    with state_lock:
        latest_target_detection = None
        target_state.update(
            {
                "locked": False,
                "auto_follow": False,
                "locked_at": "",
                "snapshot_path": "",
                "crop_path": "",
                "last_match_score": 0.0,
                "last_seen_age_seconds": None,
            }
        )
        status["last_action"] = "target_unlocked_stop"


class WebDebugHandler(BaseHTTPRequestHandler):
    server_version = "S1WebDebug/2.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def send_json(self, payload: Dict[str, object], code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.serve_index()
        elif path == "/api/status":
            self.send_json(get_status_payload())
        elif path == "/video_feed":
            self.serve_video_feed()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/start":
            self.handle_start()
        elif path == "/api/stop":
            cleanup()
            self.send_json({"ok": True, **get_status_payload()})
        elif path == "/api/gimbal":
            self.handle_gimbal()
        elif path == "/api/lock_person":
            self.handle_lock_person()
        elif path == "/api/unlock_person":
            unlock_target()
            self.send_json({"ok": True, **get_status_payload()})
        elif path == "/api/auto_follow":
            self.handle_auto_follow()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def serve_index(self) -> None:
        index_path = Path(__file__).with_name("templates") / "index.html"
        body = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_video_feed(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        while True:
            jpg = encode_latest_frame()
            if jpg is None:
                time.sleep(0.1)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                break

    def handle_start(self) -> None:
        try:
            ensure_started()
            self.send_json({"ok": True, **get_status_payload()})
        except Exception as exc:
            safe_robot_stop()
            set_error(str(exc))
            self.send_json({"ok": False, "error": str(exc), **get_status_payload()}, code=500)

    def handle_gimbal(self) -> None:
        if gimbal is None:
            self.send_json({"ok": False, "error": "robot is not started"}, code=400)
            return

        try:
            payload = self.read_json()
            direction = str(payload.get("direction", ""))

            if direction == "left":
                gimbal.drive_speed(pitch_speed=0, yaw_speed=-SAFE_GIMBAL_YAW_SPEED)
            elif direction == "right":
                gimbal.drive_speed(pitch_speed=0, yaw_speed=SAFE_GIMBAL_YAW_SPEED)
            elif direction == "up":
                gimbal.drive_speed(pitch_speed=SAFE_GIMBAL_PITCH_SPEED, yaw_speed=0)
            elif direction == "down":
                gimbal.drive_speed(pitch_speed=-SAFE_GIMBAL_PITCH_SPEED, yaw_speed=0)
            elif direction == "stop":
                safe_robot_stop()
            elif direction == "recenter":
                safe_robot_stop()
                gimbal.recenter().wait_for_completed(timeout=3)
            else:
                safe_robot_stop()
                self.send_json({"ok": False, "error": f"unknown direction: {direction}"}, code=400)
                return

            with state_lock:
                status["last_action"] = f"gimbal_{direction}"
            self.send_json({"ok": True, "direction": direction, **get_status_payload()})
        except Exception as exc:
            safe_robot_stop()
            set_error(str(exc))
            self.send_json({"ok": False, "error": str(exc), **get_status_payload()}, code=500)

    def handle_lock_person(self) -> None:
        try:
            payload = self.read_json()
            enable_follow = bool(payload.get("enable_follow", True))
            data = lock_current_person(enable_follow=enable_follow)
            self.send_json({"ok": True, **data})
        except Exception as exc:
            safe_robot_stop()
            set_error(str(exc))
            self.send_json({"ok": False, "error": str(exc), **get_status_payload()}, code=400)

    def handle_auto_follow(self) -> None:
        try:
            payload = self.read_json()
            enabled = bool(payload.get("enabled", False))
            with state_lock:
                if enabled and not bool(target_state["locked"]):
                    raise RuntimeError("lock a person before enabling auto follow")
                target_state["auto_follow"] = enabled
                status["last_action"] = "auto_follow_on" if enabled else "auto_follow_off_stop"
            if not enabled:
                safe_robot_stop()
            self.send_json({"ok": True, **get_status_payload()})
        except Exception as exc:
            safe_robot_stop()
            set_error(str(exc))
            self.send_json({"ok": False, "error": str(exc), **get_status_payload()}, code=400)


atexit.register(cleanup)


if __name__ == "__main__":
    print("Starting RoboMaster S1 Web Debug Panel")
    print(f"Open http://{HOST}:{PORT}")
    httpd = ThreadingHTTPServer((HOST, PORT), WebDebugHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        httpd.server_close()
        cleanup()
