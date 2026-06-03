"""Web debug panel for RoboMaster S1 camera, YOLO detection, and gimbal control.

This version uses only the Python standard library for HTTP serving, so no
extra web framework is required.

Run from this directory:
    ..\\.venv\\Scripts\\python.exe .\\s1_web_debug.py

Then open:
    http://127.0.0.1:5000

This script only controls the gimbal. It does not drive the chassis and does
not use the blaster.
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import cv2
from robomaster import robot
from ultralytics import YOLO

from config import (
    ROBOT_CONN_TYPE,
    SAFE_GIMBAL_PITCH_SPEED,
    SAFE_GIMBAL_YAW_SPEED,
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
yolo_model: Optional[YOLO] = None

latest_annotated_frame = None
latest_detections: List[Dict[str, object]] = []
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
    global ep_robot, camera, gimbal

    if ep_robot is not None:
        return

    print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
    camera = ep_robot.camera
    gimbal = ep_robot.gimbal

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


def cleanup() -> None:
    """Stop robot resources safely."""
    global ep_robot, camera, gimbal

    stop_event.set()
    safe_gimbal_stop()

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

    ep_robot = None
    camera = None
    gimbal = None


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
            }
        )

    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return detections


def capture_loop() -> None:
    """Read camera frames, run YOLO, and store the newest annotated frame."""
    global latest_annotated_frame, latest_detections

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

            with state_lock:
                latest_annotated_frame = annotated
                latest_detections = detections[:20]
                status["last_error"] = ""

        except Exception as exc:
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
        return {"status": dict(status), "detections": list(latest_detections)}


class WebDebugHandler(BaseHTTPRequestHandler):
    server_version = "S1WebDebug/1.0"

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
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            elif direction == "recenter":
                gimbal.recenter().wait_for_completed(timeout=3)
            else:
                safe_gimbal_stop()
                self.send_json({"ok": False, "error": f"unknown direction: {direction}"}, code=400)
                return

            with state_lock:
                status["last_action"] = f"gimbal_{direction}"
            self.send_json({"ok": True, "direction": direction, **get_status_payload()})
        except Exception as exc:
            safe_gimbal_stop()
            set_error(str(exc))
            self.send_json({"ok": False, "error": str(exc), **get_status_payload()}, code=500)


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
