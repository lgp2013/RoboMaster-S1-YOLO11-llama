"""Flask Dashboard 服务：战术机器人指挥舱 UI。

这个模块只负责 Web 展示和 API 转发。所有 ROS2 控制仍由主节点负责，
Flask 线程异常不会影响机器人控制线程。
"""

import os
import threading
import time
from typing import Callable, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request


DASHBOARD_VERSION = "tactical-camera-title-fix-20260604"


class DashboardServer:
    """线程安全的 Flask Dashboard 服务。"""

    def __init__(
        self,
        host: str,
        port: int,
        jpeg_quality: int = 80,
        command_callback: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
        settings_callback: Optional[Callable[[Optional[Dict[str, object]]], Dict[str, object]]] = None,
    ) -> None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.command_callback = command_callback
        self.settings_callback = settings_callback
        self.lock = threading.Lock()
        self.latest_frame = None
        self.status: Dict[str, object] = {}

        self.app = Flask(
            __name__,
            template_folder=os.path.join(package_dir, "templates"),
            static_folder=os.path.join(package_dir, "static"),
        )
        self.version_info = {
            "version": DASHBOARD_VERSION,
            "template_dir": os.path.join(package_dir, "templates"),
            "static_dir": os.path.join(package_dir, "static"),
            "theme": "tactical-robotics-command-center",
        }
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.after_request
        def no_cache(response):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            response.headers["X-Dashboard-Version"] = DASHBOARD_VERSION
            return response

        @self.app.route("/")
        def index():
            return render_template("index.html", dashboard_version=DASHBOARD_VERSION)

        @self.app.route("/api/dashboard/version")
        def api_dashboard_version():
            return jsonify(dict(self.version_info))

        @self.app.route("/api/status")
        def api_status():
            try:
                with self.lock:
                    return jsonify(dict(self.status))
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc)}), 500

        @self.app.route("/api/logs")
        def api_logs():
            try:
                with self.lock:
                    logs = list(self.status.get("event_logs", []))
                return jsonify({"logs": logs})
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc), "logs": []}), 500

        @self.app.route("/api/settings", methods=["GET", "POST"])
        def api_settings():
            if self.settings_callback is None:
                return jsonify({"ok": False, "message": "no settings callback"}), 503
            try:
                if request.method == "GET":
                    return jsonify(self.settings_callback(None))
                payload = request.get_json(silent=True) or {}
                result = self.settings_callback(payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc)}), 500

        @self.app.route("/api/control", methods=["POST"])
        def api_control():
            if self.command_callback is None:
                return jsonify({"ok": False, "message": "no command callback"}), 503
            try:
                payload = request.get_json(silent=True) or {}
                command = str(payload.get("command", "")).strip().upper()
                if not command:
                    return jsonify({"ok": False, "message": "missing command"}), 400
                result = self.command_callback(command, payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc)}), 500

        @self.app.route("/api/control/<command>", methods=["POST"])
        def api_control_compat(command):
            if self.command_callback is None:
                return jsonify({"ok": False, "message": "no command callback"}), 503
            try:
                payload = request.get_json(silent=True) or {}
                result = self.command_callback(str(command).strip().upper(), payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc)}), 500

        @self.app.route("/video_feed")
        def video_feed():
            return Response(self._frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def update(self, frame, status: Dict[str, object]) -> None:
        """更新 Dashboard 数据。传入 frame 会被复制，避免跨线程修改。"""
        with self.lock:
            self.latest_frame = frame.copy() if frame is not None else None
            self.status = dict(status)

    def _merge_callback_status(self, result: Dict[str, object]) -> None:
        """把控制回调返回的即时状态合并进 Dashboard 缓存。"""
        if not isinstance(result, dict):
            return
        with self.lock:
            if "event_logs" in result:
                self.status["event_logs"] = result["event_logs"]
            if "mode" in result:
                self.status["mode"] = result["mode"]

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name="person-follower-dashboard", daemon=True)
        thread.start()

    def _run(self) -> None:
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True, use_reloader=False)

    def _frame_generator(self):
        while True:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.03)
