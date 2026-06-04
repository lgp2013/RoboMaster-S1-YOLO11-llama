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
    aiLoad: "AI LOAD",
    modelState: "MODEL",
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
    lockLogs: "LOCK",
    unlockLogs: "LIVE",
    copyLogs: "COPY",
    copied: "Copied",
    copyFailed: "Copy failed",
    settings: "SETTINGS",
    settingsTitle: "SYSTEM SETTINGS",
    close: "CLOSE",
    robotIp: "Robot IP",
    llmEnabled: "LLM Enabled",
    llmBaseUrl: "LLM Base URL",
    llmModel: "LLM Model",
    vlmEnabled: "VLM Enabled",
    vlmBaseUrl: "VLM Base URL",
    vlmModel: "VLM Model",
    agentEnabled: "Agent Enabled",
    manualForwardSpeed: "Manual Forward Speed",
    manualTurnSpeed: "Manual Turn Speed",
    manualDuration: "Manual Duration",
    coreData: "Core Data",
    reload: "RELOAD",
    saveSettings: "SAVE SETTINGS",
    settingsSaved: "Settings applied at runtime.",
    settingsFailed: "Settings update failed",
    vehicleDetails: "VEHICLE DETAILS",
    connection: "Connection",
    miniMap: "MINI MAP",
    agentLog: "Agent Reasoning Log",
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
    aiLoad: "AI 算力",
    modelState: "模型",
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
    lockLogs: "锁定",
    unlockLogs: "实时",
    copyLogs: "复制",
    copied: "已复制",
    copyFailed: "复制失败",
    settings: "设置",
    settingsTitle: "系统设置",
    close: "关闭",
    robotIp: "机器人 IP",
    llmEnabled: "启用 LLM",
    llmBaseUrl: "LLM 地址",
    llmModel: "LLM 模型",
    vlmEnabled: "启用 VLM",
    vlmBaseUrl: "VLM 地址",
    vlmModel: "VLM 模型",
    agentEnabled: "启用 Agent",
    manualForwardSpeed: "手动前进速度",
    manualTurnSpeed: "手动转向速度",
    manualDuration: "手动动作时长",
    coreData: "核心数据",
    reload: "重新读取",
    saveSettings: "保存设置",
    settingsSaved: "设置已在运行时生效。",
    settingsFailed: "设置更新失败",
    vehicleDetails: "车辆详情",
    connection: "连接状态",
    miniMap: "迷你雷达",
    agentLog: "Agent 推理日志",
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
  logsLocked: false,
  latestLogs: [],
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
  setText("logLockToggle", state.logsLocked ? t("unlockLogs") : t("lockLogs"));
  renderStatus(state.lastStatus);
  renderLogs(state.latestLogs, true);
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
    setText("agentState", command);
    const payload = { command, ...extra };
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) console.warn("control failed", data);
    await refreshStatus();
    await refreshLogs(true);
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

async function refreshLogs(force = false) {
  if (state.logsLocked && !force) return;
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();
    renderLogs(data.logs || [], force);
  } catch (err) {
    renderLogs([{ message: `log feed offline: ${err}` }], force);
  }
}

function renderStatus(data = {}) {
  const agent = data.agent || {};
  const gesture = data.gesture || {};
  const safe = data.safe_cmd || { linear_x: data.linear_x || 0, angular_z: data.angular_z || 0 };
  const target = data.target ? t("person") : localizeValue(data.target_name || "none");
  const cameraStatus = data.camera_status || "unknown";
  const yoloStatus = data.yolo_status || "unknown";
  const handStatus = data.hand_status || "unknown";
  const vlmStatus = data.vlm_status || "standby";
  const llmStatus = data.llm_status || "standby";
  const agentStatus = data.agent_status || agent.status || "standby";
  const navStatus = data.nav_status || "standby";
  const compute = data.ai_compute || {};

  setText("rosState", data.connected ? t("linked") : t("lost"));
  setText("cameraState", localizeValue(cameraStatus));
  setText("yoloState", localizeValue(yoloStatus));
  setText("handState", localizeValue(handStatus));
  setText("vlmState", localizeValue(vlmStatus));
  setText("llmState", localizeValue(llmStatus));
  setText("agentTopState", localizeValue(agentStatus));
  setText("navState", localizeValue(navStatus));
  setText("connectedBadge", data.connected ? t("connected") : t("disconnected"));
  setText("videoBadge", cameraStatus === "online" ? t("online").toUpperCase() : t("standby").toUpperCase());

  setLamp("rosLamp", data.connected ? "online" : false);
  setLamp("cameraLamp", cameraStatus === "online" ? "online" : false);
  setLamp("yoloLamp", yoloStatus === "online" ? "online" : false);
  setLamp("handLamp", handStatus === "online" ? "online" : "standby");
  setLamp("vlmLamp", vlmStatus === "online" ? "online" : "standby");
  setLamp("llmLamp", llmStatus === "online" ? "online" : "standby");
  setLamp("agentLamp", agentStatus === "online" || agentStatus === "thinking" ? "online" : "standby");
  setLamp("navLamp", navStatus === "online" ? "online" : "standby");

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
  setText("aiLoad", agent.job_running ? t("thinking") : `${compute.latency_ms ?? agent.latency_ms ?? 0} ms`);
  setText("modelState", compute.model || "--");
  setText("radarState", target);
  moveRadarDot(data.target);

  setText("agentState", agent.job_running ? t("thinking") : localizeValue(llmStatus).toUpperCase());
  setText("sceneText", agent.vision_description || data.scene?.description || t("noScene"));
  setText("agentPlan", JSON.stringify(agent.last_plan || {}, null, 2));
  setText("agentLatency", `${agent.latency_ms || 0} ms`);
  setText("agentTokens", agent.tokens?.total_tokens ?? 0);
  renderAgentLogs(agent.logs || []);

  if (document.getElementById("robotDetailsModal")?.classList.contains("open")) {
    renderRobotDetails(data);
  }
}

function renderLogs(logs, force = false) {
  const el = document.getElementById("eventLog");
  if (!el) return;
  if (state.logsLocked && !force) return;
  state.latestLogs = logs.slice(0, 20);
  el.innerHTML = state.latestLogs.map((item) => {
    const msg = item.message || item.reason || item.command || item.action || JSON.stringify(item);
    const time = item.time || item.timestamp || "";
    return `<li><strong>${String(time).slice(0, 19)}</strong> ${escapeHtml(String(msg))}</li>`;
  }).join("");
}

function renderAgentLogs(logs = []) {
  const el = document.getElementById("agentLog");
  if (!el) return;
  const recent = logs.slice(0, 20);
  if (!recent.length) {
    el.innerHTML = `<li>${escapeHtml(t("noScene"))}</li>`;
    return;
  }
  el.innerHTML = recent.map((item) => {
    const action = item.action || "STOP";
    const reason = item.reason || item.message || "";
    const latency = item.latency_ms == null ? "" : ` ${item.latency_ms}ms`;
    return `<li><strong>${escapeHtml(String(action))}</strong>${escapeHtml(latency)} ${escapeHtml(String(reason))}</li>`;
  }).join("");
}

function moveRadarDot(targetInfo) {
  const dot = document.getElementById("radarDot");
  if (!dot) return;
  if (!targetInfo || targetInfo.center_x == null) {
    dot.style.left = "50%";
    dot.style.top = "50%";
    dot.style.opacity = "0.35";
    return;
  }
  const center = Number(targetInfo.center_x || 0);
  const left = Math.max(20, Math.min(80, 50 + ((center - 320) / 320) * 30));
  const size = Number(targetInfo.bbox_height_ratio || 0.3);
  const top = Math.max(20, Math.min(78, 70 - size * 70));
  dot.style.left = `${left}%`;
  dot.style.top = `${top}%`;
  dot.style.opacity = "1";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatLogsForCopy() {
  return state.latestLogs.map((item) => {
    const msg = item.message || item.reason || item.command || item.action || JSON.stringify(item);
    const time = item.time || item.timestamp || "";
    return `${String(time).slice(0, 19)} ${msg}`;
  }).join("\n");
}

async function copyLogs() {
  const text = formatLogsForCopy();
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    showLogHint(t("copied"));
  } catch (err) {
    showLogHint(`${t("copyFailed")}: ${err}`);
  }
}

function showLogHint(text) {
  const hint = document.getElementById("logCopyHint");
  if (!hint) return;
  hint.textContent = text;
  hint.classList.add("show");
  setTimeout(() => hint.classList.remove("show"), 1600);
}

function openSettings() {
  document.getElementById("settingsModal")?.classList.add("open");
  loadSettings();
}

function closeSettings() {
  document.getElementById("settingsModal")?.classList.remove("open");
}

function openRobotDetails() {
  document.getElementById("robotDetailsModal")?.classList.add("open");
  renderRobotDetails(state.lastStatus || {});
}

function closeRobotDetails() {
  document.getElementById("robotDetailsModal")?.classList.remove("open");
}

function renderRobotDetails(data = {}) {
  const safe = data.safe_cmd || { linear_x: data.linear_x || 0, angular_z: data.angular_z || 0 };
  const rawDetails = {
    dashboard_version: data.dashboard_version,
    robot_ip: data.robot_ip,
    camera_topic: data.camera_topic,
    cmd_vel_topic: data.cmd_vel_topic,
    mode: data.mode,
    battery: data.battery,
    fps: data.fps,
    camera_status: data.camera_status,
    llm_status: data.llm_status,
    control_reason: data.control_reason,
    safety_reason: data.safety_reason,
    last_image_age_sec: data.last_image_age_sec,
    last_target_age_sec: data.last_target_age_sec,
  };

  setText("detailConnected", data.connected ? t("connected") : t("disconnected"));
  setText("detailRobotIp", data.robot_ip || "--");
  setText("detailBattery", data.battery == null ? "N/A" : `${data.battery}%`);
  setText("detailMode", data.mode || "--");
  setText("detailCameraTopic", data.camera_topic || "--");
  setText("detailCmdVelTopic", data.cmd_vel_topic || "--");
  setText("detailCameraStatus", localizeValue(data.camera_status || "unknown"));
  setText("detailLlmStatus", localizeValue(data.llm_status || "standby"));
  setText("detailFps", data.fps ?? "--");
  setText("detailLinearX", Number(data.linear_x || safe.linear_x || 0).toFixed(2));
  setText("detailAngularZ", Number(data.angular_z || safe.angular_z || 0).toFixed(2));
  setText("detailTarget", data.target ? t("person") : localizeValue(data.target_name || "none"));
  setText("robotDetailsRaw", JSON.stringify(rawDetails, null, 2));
}

async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    const data = await res.json();
    if (!data.ok) throw new Error(data.message || "settings unavailable");
    fillSettings(data.settings || {}, data.core || {});
    setText("settingsMessage", "");
  } catch (err) {
    setText("settingsMessage", `${t("settingsFailed")}: ${err}`);
  }
}

function fillSettings(settings, core) {
  setInputValue("settingRobotIp", settings.robot_ip);
  setInputChecked("settingLlmEnabled", settings.llm_enabled);
  setInputValue("settingLlmBaseUrl", settings.llm_base_url);
  setInputValue("settingLlmModel", settings.llm_model);
  setInputChecked("settingVlmEnabled", settings.vlm_enabled);
  setInputValue("settingVlmBaseUrl", settings.vlm_base_url);
  setInputValue("settingVlmModel", settings.vlm_model);
  setInputChecked("settingAgentEnabled", settings.agent_enabled);
  setInputValue("settingManualForwardSpeed", settings.manual_forward_speed);
  setInputValue("settingManualTurnSpeed", settings.manual_turn_speed);
  setInputValue("settingManualDuration", settings.manual_action_duration);
  setText("settingsCoreData", JSON.stringify(core, null, 2));
}

function setInputValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value ?? "";
}

function setInputChecked(id, value) {
  const el = document.getElementById(id);
  if (el) el.checked = Boolean(value);
}

function readSettingsForm() {
  return {
    robot_ip: document.getElementById("settingRobotIp")?.value || "",
    llm_enabled: document.getElementById("settingLlmEnabled")?.checked || false,
    llm_base_url: document.getElementById("settingLlmBaseUrl")?.value || "",
    llm_model: document.getElementById("settingLlmModel")?.value || "",
    vlm_enabled: document.getElementById("settingVlmEnabled")?.checked || false,
    vlm_base_url: document.getElementById("settingVlmBaseUrl")?.value || "",
    vlm_model: document.getElementById("settingVlmModel")?.value || "",
    agent_enabled: document.getElementById("settingAgentEnabled")?.checked || false,
    manual_forward_speed: Number(document.getElementById("settingManualForwardSpeed")?.value || 0),
    manual_turn_speed: Number(document.getElementById("settingManualTurnSpeed")?.value || 0),
    manual_action_duration: Number(document.getElementById("settingManualDuration")?.value || 0),
  };
}

async function saveSettings() {
  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings: readSettingsForm() }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.message || "settings rejected");
    fillSettings(data.settings || {}, data.core || {});
    setText("settingsMessage", t("settingsSaved"));
    await refreshStatus();
    await refreshLogs(true);
  } catch (err) {
    setText("settingsMessage", `${t("settingsFailed")}: ${err}`);
  }
}

function toggleLogLock(forceValue = null) {
  state.logsLocked = forceValue === null ? !state.logsLocked : Boolean(forceValue);
  const button = document.getElementById("logLockToggle");
  button?.classList.toggle("active", state.logsLocked);
  setText("logLockToggle", state.logsLocked ? t("unlockLogs") : t("lockLogs"));
  if (!state.logsLocked) refreshLogs(true);
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
  document.getElementById("logLockToggle")?.addEventListener("click", () => toggleLogLock());
  document.getElementById("copyLogs")?.addEventListener("click", copyLogs);
  document.getElementById("eventLog")?.addEventListener("mouseenter", () => toggleLogLock(true));
  document.getElementById("settingsToggle")?.addEventListener("click", openSettings);
  document.getElementById("settingsClose")?.addEventListener("click", closeSettings);
  document.getElementById("settingsReload")?.addEventListener("click", loadSettings);
  document.getElementById("settingsSave")?.addEventListener("click", saveSettings);
  document.getElementById("settingsModal")?.addEventListener("click", (event) => {
    if (event.target?.id === "settingsModal") closeSettings();
  });
  document.getElementById("rosDetailsToggle")?.addEventListener("click", openRobotDetails);
  document.getElementById("robotDetailsClose")?.addEventListener("click", closeRobotDetails);
  document.getElementById("robotDetailsModal")?.addEventListener("click", (event) => {
    if (event.target?.id === "robotDetailsModal") closeRobotDetails();
  });

  applyLanguage();
  setInterval(() => {
    setText("clock", new Date().toLocaleTimeString());
    refreshStatus();
  }, 500);
  setInterval(() => refreshLogs(false), 1000);
  refreshStatus();
  refreshLogs(true);
}

document.addEventListener("DOMContentLoaded", boot);
