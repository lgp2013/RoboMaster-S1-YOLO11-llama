const i18n = {
  en: {
    langButton: "中文",
    liveFeed: "LIVE OPTICAL FEED",
    modeShort: "MODE",
    gestureShort: "GESTURE",
    targetShort: "TARGET",
    vectorShort: "VECTOR",
    robotTelemetry: "ROBOT TELEMETRY",
    battery: "BATTERY",
    agentIntel: "AGENT INTEL",
    sceneUnderstanding: "Scene Understanding",
    actionPlan: "Action Plan",
    latency: "Latency",
    tokens: "Tokens",
    send: "SEND",
    manualOverride: "MANUAL OVERRIDE",
    lowSpeed: "LOW SPEED",
    emergencyStop: "EMERGENCY STOP",
    startFollow: "START FOLLOW",
    pauseFollow: "PAUSE FOLLOW",
    forward: "FORWARD",
    turnLeft: "TURN LEFT",
    stop: "STOP",
    turnRight: "TURN RIGHT",
    backward: "BACKWARD",
    clearLogs: "CLEAR LOGS",
    eventLog: "EVENT LOG",
    last20: "LAST 20",
    agentPlaceholder: "Task: What do you see? / Find the cup / Follow me",
    linked: "linked",
    lost: "lost",
    online: "online",
    offline: "offline",
    standby: "standby",
    unknown: "unknown",
    connected: "CONNECTED",
    disconnected: "DISCONNECTED",
    thinking: "THINKING",
    noScene: "No scene report yet.",
    none: "none",
    person: "person",
  },
  zh: {
    langButton: "EN",
    liveFeed: "实时光学画面",
    modeShort: "模式",
    gestureShort: "手势",
    targetShort: "目标",
    vectorShort: "速度向量",
    robotTelemetry: "机器人遥测",
    battery: "电池",
    agentIntel: "智能体情报",
    sceneUnderstanding: "场景理解",
    actionPlan: "行动计划",
    latency: "响应",
    tokens: "Token",
    send: "发送",
    manualOverride: "手动接管",
    lowSpeed: "低速安全",
    emergencyStop: "紧急停止",
    startFollow: "启动跟随",
    pauseFollow: "暂停跟随",
    forward: "前进",
    turnLeft: "左转",
    stop: "停止",
    turnRight: "右转",
    backward: "后退",
    clearLogs: "清空日志",
    eventLog: "事件日志",
    last20: "最近 20 条",
    agentPlaceholder: "任务：你看到了什么？ / 帮我找水杯 / 跟着我",
    linked: "已连接",
    lost: "断开",
    online: "在线",
    offline: "离线",
    standby: "待命",
    unknown: "未知",
    connected: "已连接",
    disconnected: "未连接",
    thinking: "推理中",
    noScene: "暂无场景报告。",
    none: "无",
    person: "人",
  },
};

const state = {
  lastStatus: {},
  lang: localStorage.getItem("dashboardLanguage") || "en",
};

function t(key) {
  return i18n[state.lang]?.[key] || i18n.en[key] || key;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? "--";
}

function localizeValue(value) {
  if (value === true) return t("online");
  if (value === false) return t("offline");
  const key = String(value || "").toLowerCase();
  return i18n[state.lang]?.[key] || value;
}

function applyLanguage() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en-US";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  setText("langToggle", t("langButton"));
  renderStatus(state.lastStatus);
}

function toggleLanguage() {
  state.lang = state.lang === "zh" ? "en" : "zh";
  localStorage.setItem("dashboardLanguage", state.lang);
  applyLanguage();
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
    setText("cameraState", t("offline"));
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

function renderStatus(data = {}) {
  const agent = data.agent || {};
  const gesture = data.gesture || {};
  const safe = data.safe_cmd || { linear_x: data.linear_x || 0, angular_z: data.angular_z || 0 };
  const target = data.target ? t("person") : localizeValue(data.target_name || "none");
  const cameraStatus = data.camera_status || "unknown";
  const llmStatus = data.llm_status || "standby";

  setText("rosState", data.connected ? t("linked") : t("lost"));
  setText("llmState", localizeValue(llmStatus));
  setText("cameraState", localizeValue(cameraStatus));
  setText("connectedBadge", data.connected ? t("connected") : t("disconnected"));
  setText("videoBadge", cameraStatus === "online" ? t("online").toUpperCase() : t("standby").toUpperCase());

  setLamp("rosLamp", data.connected ? "online" : false);
  setLamp("llmLamp", llmStatus === "online" ? "online" : "standby");
  setLamp("cameraLamp", cameraStatus === "online" ? "online" : false);

  setText("mode", data.mode || "--");
  setText("hudMode", data.mode || "--");
  setText("fps", data.fps ?? "--");
  setText("hudFps", data.fps ?? "--");
  setText("battery", data.battery == null ? "N/A" : `${data.battery}%`);
  setText("linearX", Number(data.linear_x || safe.linear_x || 0).toFixed(2));
  setText("angularZ", Number(data.angular_z || safe.angular_z || 0).toFixed(2));
  setText("target", target);
  setText("hudTarget", target);
  setText("gesture", localizeValue(data.gesture_name || gesture.candidate || gesture.current || "none"));
  setText("hudGesture", localizeValue(data.gesture_name || gesture.candidate || gesture.current || "none"));
  setText("hudVector", speedText(safe));

  setText("agentState", agent.job_running ? t("thinking") : localizeValue(llmStatus).toUpperCase());
  setText("sceneText", agent.vision_description || data.scene?.description || t("noScene"));
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
  document.getElementById("langToggle")?.addEventListener("click", toggleLanguage);

  applyLanguage();
  setInterval(() => {
    setText("clock", new Date().toLocaleTimeString());
    refreshStatus();
  }, 500);
  setInterval(refreshLogs, 1000);
  refreshStatus();
  refreshLogs();
}

document.addEventListener("DOMContentLoaded", boot);
