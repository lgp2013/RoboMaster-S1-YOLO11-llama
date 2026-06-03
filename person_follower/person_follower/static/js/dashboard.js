const state = {
  lastStatus: {},
};

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? "--";
}

function setLamp(id, status) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("online", "standby");
  if (status === "online" || status === true) el.classList.add("online");
  else if (status === "standby") el.classList.add("standby");
}

function speedText(cmd) {
  if (!cmd) return "x=0 z=0";
  return `x=${Number(cmd.linear_x || 0).toFixed(2)} z=${Number(cmd.angular_z || 0).toFixed(2)}`;
}

async function sendControl(command, extra = {}) {
  try {
    const payload = { command, ...extra };
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) console.warn("control failed", data);
    await refreshStatus();
  } catch (err) {
    console.error("control request failed", err);
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    state.lastStatus = data;
    renderStatus(data);
  } catch (err) {
    setText("cameraState", "offline");
    setLamp("cameraLamp", false);
  }
}

async function refreshLogs() {
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();
    renderLogs(data.logs || []);
  } catch (err) {
    renderLogs([{ message: `log feed offline: ${err}` }]);
  }
}

function renderStatus(data) {
  const agent = data.agent || {};
  const gesture = data.gesture || {};
  const safe = data.safe_cmd || { linear_x: data.linear_x || 0, angular_z: data.angular_z || 0 };
  const target = data.target ? "person" : (data.target_name || "none");

  setText("rosState", data.connected ? "linked" : "lost");
  setText("llmState", data.llm_status || "standby");
  setText("cameraState", data.camera_status || "unknown");
  setText("connectedBadge", data.connected ? "CONNECTED" : "DISCONNECTED");
  setText("videoBadge", data.camera_status === "online" ? "ONLINE" : "STANDBY");

  setLamp("rosLamp", data.connected ? "online" : false);
  setLamp("llmLamp", data.llm_status === "online" ? "online" : "standby");
  setLamp("cameraLamp", data.camera_status === "online" ? "online" : false);

  setText("mode", data.mode || "--");
  setText("hudMode", data.mode || "--");
  setText("fps", data.fps ?? "--");
  setText("hudFps", data.fps ?? "--");
  setText("battery", data.battery == null ? "N/A" : `${data.battery}%`);
  setText("linearX", Number(data.linear_x || safe.linear_x || 0).toFixed(2));
  setText("angularZ", Number(data.angular_z || safe.angular_z || 0).toFixed(2));
  setText("target", target);
  setText("hudTarget", target);
  setText("gesture", data.gesture_name || gesture.candidate || gesture.current || "none");
  setText("hudGesture", data.gesture_name || gesture.candidate || gesture.current || "none");
  setText("hudVector", speedText(safe));

  setText("agentState", agent.job_running ? "THINKING" : (data.llm_status || "STANDBY"));
  setText("sceneText", agent.vision_description || data.scene?.description || "No scene report yet.");
  setText("agentPlan", JSON.stringify(agent.last_plan || {}, null, 2));
  setText("agentLatency", `${agent.latency_ms || 0} ms`);
  setText("agentTokens", agent.tokens?.total_tokens ?? 0);
}

function renderLogs(logs) {
  const el = document.getElementById("eventLog");
  if (!el) return;
  el.innerHTML = logs.slice(0, 20).map((item) => {
    const msg = item.message || item.reason || item.command || item.action || JSON.stringify(item);
    const time = item.time || item.timestamp || "";
    return `<li><strong>${String(time).slice(0, 19)}</strong> ${msg}</li>`;
  }).join("");
}

function boot() {
  document.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => {
      const command = button.dataset.command;
      if (command === "AGENT_QUERY") {
        const text = document.getElementById("agentQuery")?.value || "";
        sendControl(command, { text });
      } else {
        sendControl(command);
      }
    });
  });

  setInterval(() => {
    setText("clock", new Date().toLocaleTimeString());
    refreshStatus();
  }, 500);
  setInterval(refreshLogs, 1000);
  refreshStatus();
  refreshLogs();
}

document.addEventListener("DOMContentLoaded", boot);
