"""Flask dashboard for YOLO, gestures and Vision-Language-Agent debugging."""

import threading
import time
from typing import Callable, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboMaster S1 Vision-Language-Agent</title>
  <style>
    :root {
      --bg: #0d1620;
      --panel: #142232;
      --panel2: #1b2d41;
      --line: #31506b;
      --text: #eaf2f8;
      --muted: #9fb8cc;
      --green: #6ee7a8;
      --yellow: #ffd166;
      --red: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { height: 58px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; background: #102033; border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 20px; }
    .hint { color: var(--muted); font-size: 13px; }
    main { display: grid; grid-template-columns: minmax(620px, 1fr) 430px; grid-template-rows: 1fr 285px; gap: 14px; padding: 14px; height: calc(100vh - 58px); }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; min-height: 0; }
    h2 { margin: 0; padding: 10px 12px; font-size: 15px; background: var(--panel2); border-bottom: 1px solid var(--line); }
    .video { grid-row: 1 / span 2; display: flex; flex-direction: column; }
    .video img { width: 100%; height: 100%; min-height: 0; object-fit: contain; background: #05090d; display: block; }
    .body { padding: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .card { background: #0f1c29; border: 1px solid #28455f; border-radius: 6px; padding: 10px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .value { font-size: 17px; font-weight: 700; word-break: break-word; }
    .ok { color: var(--green); }
    .warn { color: var(--yellow); }
    .danger { color: var(--red); }
    .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    button { background: #21394f; color: var(--text); border: 1px solid #4f7699; border-radius: 6px; padding: 10px 8px; font-size: 14px; cursor: pointer; }
    button:hover { background: #2a4660; }
    button.stop { background: #4a2028; border-color: #91414d; }
    input { width: 100%; background: #0b1722; color: var(--text); border: 1px solid #365c7a; border-radius: 6px; padding: 10px; margin-top: 10px; }
    ul { margin: 0; padding-left: 18px; line-height: 1.45; color: var(--muted); }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: var(--muted); font-size: 12px; max-height: 170px; overflow: auto; }
  </style>
</head>
<body>
  <header>
    <h1>RoboMaster S1 视觉语言智能体</h1>
    <div class="hint">模式: <span id="mode">--</span> · <span id="clock">--</span> · Ctrl+C 退出 ROS 节点</div>
  </header>
  <main>
    <section class="video">
      <h2>实时视频 / YOLO / MediaPipe Hands</h2>
      <img src="/video_feed" alt="video">
    </section>
    <section>
      <h2>机器人状态</h2>
      <div class="body">
        <div class="grid">
          <div class="card"><div class="label">FPS</div><div class="value" id="fps">--</div></div>
          <div class="card"><div class="label">当前模式</div><div class="value ok" id="modeCard">--</div></div>
          <div class="card"><div class="label">候选手势</div><div class="value" id="gesture">--</div></div>
          <div class="card"><div class="label">稳定手势</div><div class="value" id="stable">--</div></div>
          <div class="card"><div class="label">目标人数</div><div class="value" id="people">--</div></div>
          <div class="card"><div class="label">安全速度</div><div class="value" id="speed">--</div></div>
        </div>
        <div class="buttons">
          <button class="stop" onclick="sendCommand('STOP')">手动停止</button>
          <button onclick="sendCommand('START_FOLLOW')">启动跟随</button>
          <button onclick="sendCommand('PAUSE')">暂停跟随</button>
          <button onclick="sendCommand('AGENT_MODE')">Agent模式</button>
        </div>
        <input id="query" placeholder="输入任务，例如：你看到了什么？/ 帮我找水杯 / 跟着我">
        <div class="buttons">
          <button onclick="sendAgentQuery()">发送给Agent</button>
          <button onclick="sendCommand('CLEAR_LOGS')">清空手势日志</button>
        </div>
      </div>
    </section>
    <section>
      <h2>Agent 面板</h2>
      <div class="body">
        <div class="grid">
          <div class="card"><div class="label">响应时间</div><div class="value" id="latency">--</div></div>
          <div class="card"><div class="label">Token</div><div class="value" id="tokens">--</div></div>
        </div>
        <p class="label">场景描述</p>
        <pre id="description">--</pre>
        <p class="label">Agent计划</p>
        <pre id="plan">--</pre>
        <p class="label">最近Agent日志</p>
        <ul id="agentLogs"></ul>
        <p class="label">完整状态</p>
        <pre id="raw"></pre>
      </div>
    </section>
  </main>
  <script>
    async function sendCommand(command) {
      await fetch('/api/control/' + command, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
      await refreshStatus();
    }
    async function sendAgentQuery() {
      const text = document.getElementById('query').value;
      await fetch('/api/control/AGENT_QUERY', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      await refreshStatus();
    }
    function fmtSpeed(cmd) {
      if (!cmd) return '--';
      return 'x=' + cmd.linear_x + ', z=' + cmd.angular_z;
    }
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const mode = data.mode || '--';
        const agent = data.agent || {};
        document.getElementById('mode').textContent = mode;
        document.getElementById('modeCard').textContent = mode;
        document.getElementById('fps').textContent = data.fps ?? '--';
        document.getElementById('gesture').textContent = data.gesture?.candidate || data.gesture?.current || '--';
        document.getElementById('stable').textContent = data.gesture?.stable_gesture || '--';
        document.getElementById('people').textContent = data.people_count ?? '--';
        document.getElementById('speed').textContent = fmtSpeed(data.safe_cmd);
        document.getElementById('description').textContent = agent.vision_description || data.scene?.description || '--';
        document.getElementById('plan').textContent = JSON.stringify(agent.last_plan || {}, null, 2);
        document.getElementById('latency').textContent = (agent.latency_ms ?? 0) + ' ms';
        document.getElementById('tokens').textContent = agent.tokens?.total_tokens ?? 0;
        const logs = agent.logs || [];
        document.getElementById('agentLogs').innerHTML = logs.map(item => '<li>' + item.action + ' · ' + item.reason + '</li>').join('');
        document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('raw').textContent = 'Dashboard refresh failed: ' + err;
      }
    }
    setInterval(() => {
      document.getElementById('clock').textContent = new Date().toLocaleTimeString();
      refreshStatus();
    }, 500);
    refreshStatus();
  </script>
</body>
</html>
"""


class DashboardServer:
    """线程安全 Dashboard 服务。

    Flask 运行在 daemon 线程中，异常不会影响 ROS2 控制循环。
    """

    def __init__(
        self,
        host: str,
        port: int,
        jpeg_quality: int = 80,
        command_callback: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.command_callback = command_callback
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

        @self.app.route("/api/control/<command>", methods=["POST"])
        def api_control(command):
            if self.command_callback is None:
                return jsonify({"ok": False, "message": "no command callback"}), 503
            try:
                payload = request.get_json(silent=True) or {}
                return jsonify(self.command_callback(command, payload))
            except Exception as exc:
                return jsonify({"ok": False, "message": str(exc)}), 500

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
