"""Flask Dashboard 服务，负责 Tactical UI 和详情 API。"""

import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file


DASHBOARD_VERSION = "tactical-detail-modal-20260604-1705"


class DashboardServer:
    """线程安全的 Flask Dashboard 服务。"""

    def __init__(
        self,
        host: str,
        port: int,
        jpeg_quality: int = 80,
        command_callback: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
        settings_callback: Optional[Callable[[Optional[Dict[str, object]]], Dict[str, object]]] = None,
        media_callback: Optional[Callable[[], Dict[str, object]]] = None,
    ) -> None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.command_callback = command_callback
        self.settings_callback = settings_callback
        self.media_callback = media_callback
        self.lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.preview_mode = "raw"
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

    def _ok(self, data: Dict[str, object]) -> Dict[str, object]:
        """统一成功响应。"""
        return {"success": True, "data": data, "timestamp": round(time.time(), 3)}

    def _error(self, message: str) -> Dict[str, object]:
        """统一错误响应。"""
        return {"success": False, "error": message, "timestamp": round(time.time(), 3)}

    def _status_copy(self) -> Dict[str, object]:
        with self.lock:
            return dict(self.status)

    def _detail_data(self, name: str) -> Dict[str, object]:
        status = self._status_copy()
        return dict(status.get("details", {}).get(name, {}))

    def _media_data(self) -> Dict[str, object]:
        if self.media_callback is not None:
            return dict(self.media_callback())
        return self._detail_data("media")

    def _resolve_media_path(self, relative_path: str) -> Path:
        data = self._media_data()
        record_root = data.get("record_root")
        if not record_root:
            raise FileNotFoundError("record root unavailable")
        root = Path(str(record_root)).resolve()
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise FileNotFoundError("path escapes record root")
        if not candidate.exists():
            raise FileNotFoundError("media file not found")
        return candidate

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

        @self.app.route("/media")
        def media_page():
            return render_template("media.html", dashboard_version=DASHBOARD_VERSION)

        @self.app.route("/api/dashboard/version")
        def api_dashboard_version():
            return jsonify(self._ok(dict(self.version_info)))

        @self.app.route("/api/dashboard/preview_mode", methods=["GET", "POST"])
        def api_dashboard_preview_mode():
            try:
                if request.method == "POST":
                    payload = request.get_json(silent=True) or {}
                    mode = str(payload.get("mode", "raw")).strip().lower()
                    if mode not in ("raw", "annotated"):
                        return jsonify(self._error("invalid preview mode")), 400
                    with self.lock:
                        self.preview_mode = mode
                with self.lock:
                    mode = self.preview_mode
                return jsonify(self._ok({"mode": mode}))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/status")
        def api_status():
            try:
                return jsonify(self._status_copy())
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/settings", methods=["GET", "POST"])
        def api_settings():
            if self.settings_callback is None:
                return jsonify(self._error("no settings callback")), 503
            try:
                if request.method == "GET":
                    return jsonify(self.settings_callback(None))
                payload = request.get_json(silent=True) or {}
                result = self.settings_callback(payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/control", methods=["POST"])
        def api_control():
            if self.command_callback is None:
                return jsonify(self._error("no command callback")), 503
            try:
                payload = request.get_json(silent=True) or {}
                command = str(payload.get("command", "")).strip().upper()
                if not command:
                    return jsonify(self._error("missing command")), 400
                result = self.command_callback(command, payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/control/<command>", methods=["POST"])
        def api_control_compat(command):
            if self.command_callback is None:
                return jsonify(self._error("no command callback")), 503
            try:
                payload = request.get_json(silent=True) or {}
                result = self.command_callback(str(command).strip().upper(), payload)
                self._merge_callback_status(result)
                return jsonify(result)
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/logs")
        def api_logs():
            try:
                logs = self._detail_data("logs").get("items", [])
                return jsonify({"logs": logs})
            except Exception as exc:
                return jsonify({"logs": [], "success": False, "error": str(exc)}), 500

        @self.app.route("/api/detail/telemetry")
        def api_detail_telemetry():
            try:
                return jsonify(self._ok(self._detail_data("telemetry")))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/detail/agent")
        def api_detail_agent():
            try:
                return jsonify(self._ok(self._detail_data("agent")))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/detail/logs")
        def api_detail_logs():
            try:
                return jsonify(self._ok(self._detail_data("logs")))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/detail/models")
        def api_detail_models():
            try:
                return jsonify(self._ok(self._detail_data("models")))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/detail/sub_agents")
        def api_detail_sub_agents():
            try:
                return jsonify(self._ok(self._detail_data("sub_agents")))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/detail/media")
        def api_detail_media():
            try:
                return jsonify(self._ok(self._media_data()))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/media/file")
        def api_media_file():
            try:
                relative_path = str(request.args.get("path", "")).strip()
                if not relative_path:
                    return jsonify(self._error("missing path")), 400
                return send_file(self._resolve_media_path(relative_path))
            except FileNotFoundError as exc:
                return jsonify(self._error(str(exc))), 404
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/api/logs/clear", methods=["POST"])
        def api_logs_clear():
            if self.command_callback is None:
                return jsonify(self._error("no command callback")), 503
            try:
                result = self.command_callback("CLEAR_LOGS", {})
                self._merge_callback_status(result)
                return jsonify(self._ok({"cleared": True}))
            except Exception as exc:
                return jsonify(self._error(str(exc))), 500

        @self.app.route("/video_feed")
        def video_feed():
            return Response(self._frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def update(self, raw_frame, annotated_frame, status: Dict[str, object]) -> None:
        """更新 Dashboard 数据。传入 frame 会被复制，避免跨线程修改。"""
        with self.lock:
            self.latest_raw_frame = raw_frame.copy() if raw_frame is not None else None
            self.latest_annotated_frame = annotated_frame.copy() if annotated_frame is not None else None
            self.status = dict(status)

    def _merge_callback_status(self, result: Dict[str, object]) -> None:
        """把控制回调返回的即时状态合并进 Dashboard 缓存。"""
        if not isinstance(result, dict):
            return
        status = result.get("status")
        if isinstance(status, dict):
            with self.lock:
                self.status = dict(status)
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
                source = self.latest_raw_frame if self.preview_mode == "raw" else self.latest_annotated_frame
                frame = None if source is None else source.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.03)
