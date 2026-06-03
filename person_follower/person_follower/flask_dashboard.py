"""Flask dashboard for live person follower debugging."""

import threading
import time
from typing import Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>RoboMaster S1 Person Follower</title>
  <style>
    body { margin: 0; background: #101820; color: #e8eef4; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { padding: 14px 20px; background: #182638; border-bottom: 1px solid #2f455e; }
    h1 { margin: 0; font-size: 22px; }
    main { display: grid; grid-template-columns: minmax(640px, 2fr) 420px; gap: 16px; padding: 16px; }
    section { background: #162331; border: 1px solid #2f455e; border-radius: 8px; overflow: hidden; }
    h2 { margin: 0; padding: 12px 14px; font-size: 16px; background: #1d3044; border-bottom: 1px solid #2f455e; }
    .body { padding: 14px; }
    img { width: 100%; height: auto; display: block; background: #05080c; }
    pre { white-space: pre-wrap; word-break: break-word; line-height: 1.45; }
    .ok { color: #78e08f; }
    .warn { color: #ffd166; }
  </style>
</head>
<body>
  <header><h1>RoboMaster S1 Person Follower Dashboard</h1></header>
  <main>
    <section>
      <h2>实时视频 / YOLO11</h2>
      <img src="/video_feed">
    </section>
    <section>
      <h2>机器人状态</h2>
      <div class="body"><pre id="status">loading...</pre></div>
    </section>
  </main>
  <script>
    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      document.getElementById('status').textContent = JSON.stringify(data, null, 2);
    }
    setInterval(refreshStatus, 500);
    refreshStatus();
  </script>
</body>
</html>
"""


class DashboardServer:
    """线程安全 Dashboard 状态与 MJPEG 视频服务。"""

    def __init__(self, host: str, port: int, jpeg_quality: int = 80) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.app = Flask(__name__)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.status: Dict[str, object] = {}
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.route("/")
        def index():
            return render_template_string(HTML)

        @self.app.route("/api/status")
        def api_status():
            with self.lock:
                return jsonify(dict(self.status))

        @self.app.route("/video_feed")
        def video_feed():
            return Response(self._frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def update(self, frame, status: Dict[str, object]) -> None:
        with self.lock:
            self.latest_frame = frame.copy() if frame is not None else None
            self.status = dict(status)

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
