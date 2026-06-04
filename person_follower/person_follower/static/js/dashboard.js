const I18N = {
  zh: {
    appTitle: "RoboMaster Tactical AI Console",
    liveFeed: "LIVE FEED",
    modeShort: "MODE",
    gestureShort: "GESTURE",
    targetShort: "TARGET",
    vectorShort: "VECTOR",
    robotTelemetry: "ROBOT TELEMETRY",
    battery: "BATTERY",
    agentIntel: "AGENT INTELLIGENCE",
    actionPlan: "ACTION PLAN",
    sceneUnderstanding: "SCENE",
    eventLog: "EVENT LOG",
    subAgents: "SUB-AGENT STATUS",
    detail: "DETAIL",
    settings: "SETTINGS",
    settingsTitle: "SYSTEM SETTINGS",
    close: "CLOSE",
    refresh: "REFRESH",
    copy: "COPY",
    clear: "CLEAR",
    saveSettings: "SAVE SETTINGS",
    manualOverride: "MANUAL OVERRIDE",
    lowSpeed: "LOW SPEED",
    emergencyStop: "EMERGENCY STOP",
    sleep: "SLEEP",
    wake: "WAKE",
    startFollow: "START FOLLOW",
    pauseFollow: "PAUSE FOLLOW",
    forward: "FORWARD",
    backward: "BACKWARD",
    turnLeft: "TURN LEFT",
    turnRight: "TURN RIGHT",
    stop: "STOP",
    strafeLeft: "STRAFE LEFT",
    strafeRight: "STRAFE RIGHT",
    gimbalUp: "GIMBAL UP",
    gimbalDown: "GIMBAL DOWN",
    gimbalLeft: "GIMBAL LEFT",
    gimbalRight: "GIMBAL RIGHT",
    gimbalCenter: "GIMBAL CENTER",
    agentMode: "AGENT MODE",
    gestureMode: "GESTURE MODE",
    idleMode: "IDLE",
    autoLock: "AUTO LOCK",
    snapshot: "SNAPSHOT",
    startRecord: "START RECORD",
    stopRecord: "STOP RECORD",
    clearLogs: "CLEAR LOGS",
    resetTarget: "RESET TARGET",
    reconnectRobot: "RECONNECT",
    clickToLock: "Click video to lock person",
    lockState: "LOCK",
    connection: "CONNECTION",
    status: "STATUS",
    currentAction: "ACTION",
    latency: "LATENCY",
    tokens: "TOKENS",
    controlSource: "SOURCE",
    modelShort: "MODEL",
    modelRuntime: "MODEL RUNTIME",
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
    manualDuration: "Manual Action Duration",
    coreData: "Core Data",
    safetyZone: "Safety Control",
    chassisZone: "Chassis Control",
    gimbalZone: "Gimbal Control",
    modeZone: "Mode & Utility",
    copied: "Copied",
    copyFailed: "Copy failed",
    online: "online",
    offline: "offline",
    standby: "standby",
    thinking: "thinking",
    connected: "connected",
    disconnected: "disconnected",
    none: "none",
  },
  en: {
    appTitle: "RoboMaster Tactical AI Console",
    liveFeed: "LIVE FEED",
    modeShort: "MODE",
    gestureShort: "GESTURE",
    targetShort: "TARGET",
    vectorShort: "VECTOR",
    robotTelemetry: "ROBOT TELEMETRY",
    battery: "BATTERY",
    agentIntel: "AGENT INTELLIGENCE",
    actionPlan: "ACTION PLAN",
    sceneUnderstanding: "SCENE",
    eventLog: "EVENT LOG",
    subAgents: "SUB-AGENT STATUS",
    detail: "DETAIL",
    settings: "SETTINGS",
    settingsTitle: "SYSTEM SETTINGS",
    close: "CLOSE",
    refresh: "REFRESH",
    copy: "COPY",
    clear: "CLEAR",
    saveSettings: "SAVE SETTINGS",
    manualOverride: "MANUAL OVERRIDE",
    lowSpeed: "LOW SPEED",
    emergencyStop: "EMERGENCY STOP",
    sleep: "SLEEP",
    wake: "WAKE",
    startFollow: "START FOLLOW",
    pauseFollow: "PAUSE FOLLOW",
    forward: "FORWARD",
    backward: "BACKWARD",
    turnLeft: "TURN LEFT",
    turnRight: "TURN RIGHT",
    stop: "STOP",
    strafeLeft: "STRAFE LEFT",
    strafeRight: "STRAFE RIGHT",
    gimbalUp: "GIMBAL UP",
    gimbalDown: "GIMBAL DOWN",
    gimbalLeft: "GIMBAL LEFT",
    gimbalRight: "GIMBAL RIGHT",
    gimbalCenter: "GIMBAL CENTER",
    agentMode: "AGENT MODE",
    gestureMode: "GESTURE MODE",
    idleMode: "IDLE",
    autoLock: "AUTO LOCK",
    snapshot: "SNAPSHOT",
    startRecord: "START RECORD",
    stopRecord: "STOP RECORD",
    clearLogs: "CLEAR LOGS",
    resetTarget: "RESET TARGET",
    reconnectRobot: "RECONNECT",
    clickToLock: "Click video to lock person",
    lockState: "LOCK",
    connection: "CONNECTION",
    status: "STATUS",
    currentAction: "ACTION",
    latency: "LATENCY",
    tokens: "TOKENS",
    controlSource: "SOURCE",
    modelShort: "MODEL",
    modelRuntime: "MODEL RUNTIME",
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
    manualDuration: "Manual Action Duration",
    coreData: "Core Data",
    safetyZone: "Safety Control",
    chassisZone: "Chassis Control",
    gimbalZone: "Gimbal Control",
    modeZone: "Mode & Utility",
    copied: "Copied",
    copyFailed: "Copy failed",
    online: "online",
    offline: "offline",
    standby: "standby",
    thinking: "thinking",
    connected: "connected",
    disconnected: "disconnected",
    none: "none",
  },
};

const DETAIL_META = {
  telemetry: { endpoint: "/api/detail/telemetry", title: { zh: "Robot Telemetry Detail", en: "Robot Telemetry Detail" } },
  agent: { endpoint: "/api/detail/agent", title: { zh: "Agent Intelligence Detail", en: "Agent Intelligence Detail" } },
  logs: { endpoint: "/api/detail/logs", title: { zh: "Event Log Detail", en: "Event Log Detail" } },
  models: { endpoint: "/api/detail/models", title: { zh: "Model Runtime Detail", en: "Model Runtime Detail" } },
  sub_agents: { endpoint: "/api/detail/sub_agents", title: { zh: "Sub-Agent Status Detail", en: "Sub-Agent Status Detail" } },
};

const state = {
  lang: localStorage.getItem("dashboardLanguage") || "zh",
  lastStatus: {},
  currentDetailType: "",
  currentDetailData: {},
};

function t(key) {
  return I18N[state.lang]?.[key] || I18N.en[key] || key;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value ?? "--";
}

function setTitle(id, value) {
  const node = document.getElementById(id);
  if (node) node.title = value ?? "";
}

function setValue(id, value) {
  const node = document.getElementById(id);
  if (node) node.value = value ?? "";
}

function setChecked(id, value) {
  const node = document.getElementById(id);
  if (node) node.checked = Boolean(value);
}

function localizeStatus(value) {
  const key = String(value ?? "").toLowerCase();
  return I18N[state.lang]?.[key] || value || "--";
}

function ellipsis(value, limit = 40) {
  const text = String(value ?? "");
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function setLamp(id, status) {
  const node = document.getElementById(id);
  if (!node) return;
  node.classList.remove("online", "standby");
  const value = String(status ?? "").toLowerCase();
  if (["online", "connected", "ready", "locked", "active", "candidate"].includes(value)) {
    node.classList.add("online");
  } else if (["standby", "idle", "thinking", "scanning", "lost"].includes(value)) {
    node.classList.add("standby");
  }
}

function commandVectorText(cmd = {}) {
  const x = Number(cmd.linear_x || 0).toFixed(2);
  const y = Number(cmd.linear_y || 0).toFixed(2);
  const z = Number(cmd.angular_z || 0).toFixed(2);
  return `x=${x} y=${y} z=${z}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  return response.json();
}

async function sendControl(command, extra = {}) {
  const payload = await fetchJson("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command, ...extra }),
  });
  if (payload.status) {
    state.lastStatus = payload.status;
    renderStatus(payload.status);
  } else {
    await refreshStatus();
  }
  if (state.currentDetailType) {
    await refreshDetailModal();
  }
}

async function refreshStatus() {
  const payload = await fetchJson("/api/status");
  state.lastStatus = payload;
  renderStatus(payload);
}

function renderStatus(data = {}) {
  const telemetry = data.telemetry_summary || {};
  const agentSummary = data.agent_summary || {};
  const modelRuntime = data.model_runtime || {};
  const subAgents = data.sub_agents || {};

  setText("rosState", localizeStatus(data.connected ? "connected" : "disconnected"));
  setText("cameraState", localizeStatus(data.camera_status || "offline"));
  setText("followState", data.follow_lock_state || "--");
  setText("agentTopState", localizeStatus(data.agent_status || "standby"));
  setText("recordState", data.recording?.active ? "ON" : "OFF");
  setLamp("rosLamp", data.connected ? "connected" : "offline");
  setLamp("cameraLamp", data.camera_status || "offline");
  setLamp("followLamp", data.follow_lock_state || "none");
  setLamp("agentLamp", data.agent_status || "standby");
  setLamp("recordLamp", data.recording?.active ? "online" : "standby");

  setText("videoBadge", String(data.camera_status || "standby").toUpperCase());
  setText("hudMode", data.mode || "--");
  setText("hudLockState", data.follow_lock_state || "--");
  setText("hudTarget", localizeStatus(data.target_name || "none"));
  setText("hudGesture", localizeStatus(data.gesture_name || "none"));
  setText("hudFps", data.fps ?? "--");
  setText("hudVector", commandVectorText(data.safe_cmd || {}));

  setText("telemetryConnection", telemetry.connection || "--");
  setText("telemetryBattery", telemetry.battery == null ? "N/A" : `${telemetry.battery}%`);
  setText("telemetryMode", telemetry.mode || "--");
  setText("telemetryLinearX", telemetry.linear_x ?? "--");
  setText("telemetryAngularZ", telemetry.angular_z ?? "--");
  setText("telemetrySource", ellipsis(telemetry.control_source || "--", 18));
  setTitle("telemetrySource", telemetry.control_source || "--");
  setText("telemetryTarget", telemetry.target_summary || "--");
  setText("telemetryModel", telemetry.model_name_short || "--");
  setTitle("telemetryModel", telemetry.model_name_short || "--");

  setText("agentStatusSummary", localizeStatus(agentSummary.status || "standby"));
  setText("agentSceneSummary", ellipsis(agentSummary.scene_summary || "--", 54));
  setTitle("agentSceneSummary", agentSummary.scene_summary || "--");
  setText("agentPlanSummary", ellipsis(agentSummary.action_summary || "--", 54));
  setTitle("agentPlanSummary", agentSummary.action_summary || "--");
  setText("agentActionSummary", agentSummary.current_action || "--");
  setText("agentLatencySummary", `${agentSummary.latency_ms ?? 0} ms`);
  setText("agentTokenSummary", agentSummary.token_summary?.total_tokens ?? 0);

  setText("followAgentSummary", subAgents.FOLLOW_AGENT?.status || "--");
  setText("gestureAgentSummary", subAgents.GESTURE_AGENT?.status || "--");
  setText("safetyAgentSummary", subAgents.SAFETY_AGENT?.status || "--");
  setText("vlmAgentSummary", subAgents.VLM_AGENT?.status || "--");
  setText("llmAgentSummary", subAgents.LLM_AGENT?.status || "--");
  setText("manualAgentSummary", subAgents.MANUAL_AGENT?.status || "--");
  setText("dashboardAgentSummary", subAgents.DASHBOARD_AGENT?.status || "--");

  setText("modelYoloSummary", modelRuntime.YOLO?.status || "--");
  setText("modelHandSummary", modelRuntime.HAND?.status || "--");
  setText("modelVlmSummary", modelRuntime.VLM?.status || "--");
  setText("modelLlmSummary", modelRuntime.LLM?.status || "--");
  setText("modelAgentSummary", modelRuntime.AGENT?.status || "--");

  renderEventSummary(data.event_summary || []);

  const followButton = document.getElementById("startFollowButton");
  if (followButton) followButton.disabled = !data.can_start_follow;
}

function renderEventSummary(items) {
  const node = document.getElementById("eventSummaryList");
  if (!node) return;
  if (!items.length) {
    node.innerHTML = `<li>${escapeHtml(t("none"))}</li>`;
    return;
  }
  node.innerHTML = items.slice(0, 5).map((item) => {
    const full = `${item.time || "--"} [${item.source || "SYSTEM"}] ${item.message || ""}`;
    return `<li title="${escapeHtml(full)}">${escapeHtml(ellipsis(full, 82))}</li>`;
  }).join("");
}

async function openDetailModal(type) {
  const meta = DETAIL_META[type];
  if (!meta) return;
  const payload = await fetchJson(meta.endpoint);
  state.currentDetailType = type;
  state.currentDetailData = payload.data || {};
  document.getElementById("detailModalTitle").textContent = meta.title[state.lang] || meta.title.en;
  document.getElementById("detailModal").classList.add("open");
  renderDetailModal(type, state.currentDetailData);
}

function closeDetailModal() {
  document.getElementById("detailModal").classList.remove("open");
}

async function refreshDetailModal() {
  if (!state.currentDetailType) return;
  await openDetailModal(state.currentDetailType);
}

async function copyDetailContent() {
  const text = JSON.stringify(state.currentDetailData || {}, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    showHint(t("copied"));
  } catch (error) {
    showHint(`${t("copyFailed")}: ${error}`);
  }
}

async function clearDetailLogs() {
  if (state.currentDetailType !== "logs") return;
  await fetchJson("/api/logs/clear", { method: "POST" });
  await refreshStatus();
  await refreshDetailModal();
}

function renderDetailModal(type, data) {
  const body = document.getElementById("detailBody");
  const filter = document.getElementById("detailFilter");
  const clearButton = document.getElementById("detailClear");

  filter.style.display = type === "logs" ? "inline-block" : "none";
  clearButton.style.display = type === "logs" ? "inline-flex" : "none";

  if (type === "logs") {
    renderLogDetail(body, filter, data);
    return;
  }

  filter.innerHTML = "";
  filter.onchange = null;

  if (type === "sub_agents") {
    body.innerHTML = renderSubAgentDetail(data);
    return;
  }

  body.innerHTML = renderKeyValueDetail(data);
}

function renderLogDetail(body, filter, data) {
  const filters = Array.isArray(data.filters) ? data.filters : ["ALL"];
  const currentValue = filter.value || "ALL";
  filter.innerHTML = filters.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  filter.value = filters.includes(currentValue) ? currentValue : "ALL";
  filter.onchange = () => renderLogDetail(body, filter, data);

  const selected = filter.value || "ALL";
  const items = (data.items || []).filter((item) => {
    if (selected === "ALL") return true;
    if (selected === "ERROR") return String(item.level || "").toUpperCase() === "ERROR";
    return String(item.source || "").toUpperCase() === selected;
  });

  body.innerHTML = `
    <ul class="detail-log-list">
      ${items.map((item) => `
        <li>
          <strong>${escapeHtml(item.time || "--")}</strong>
          [${escapeHtml(item.source || "SYSTEM")}]
          ${escapeHtml(item.message || "")}
        </li>
      `).join("") || `<li>${escapeHtml(t("none"))}</li>`}
    </ul>
  `;
}

function renderSubAgentDetail(data) {
  const items = data.items || {};
  const followMeta = data.follow_meta || {};
  const cards = Object.entries(items).map(([name, info]) => `
    <div class="detail-item">
      <span>${escapeHtml(name)}</span>
      <strong>${escapeHtml(String(info.status || "--"))}</strong>
      <p>${escapeHtml(String(info.message || info.last_event || "--"))}</p>
      <pre>${escapeHtml(JSON.stringify(info, null, 2))}</pre>
    </div>
  `);
  cards.push(`
    <div class="detail-item">
      <span>FOLLOW_META</span>
      <pre>${escapeHtml(JSON.stringify(followMeta, null, 2))}</pre>
    </div>
  `);
  return `<div class="detail-grid">${cards.join("")}</div>`;
}

function renderKeyValueDetail(data) {
  const cards = Object.entries(data || {}).map(([key, value]) => {
    const content = value && typeof value === "object"
      ? `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`
      : `<strong>${escapeHtml(String(value))}</strong>`;
    return `
      <div class="detail-item">
        <span>${escapeHtml(key)}</span>
        ${content}
      </div>
    `;
  });
  return `<div class="detail-grid">${cards.join("")}</div>`;
}

function applyLanguage() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en-US";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.title = t("appTitle");
  setText("langToggle", state.lang === "zh" ? "EN" : "ZH");
  if (state.currentDetailType) {
    const meta = DETAIL_META[state.currentDetailType];
    if (meta) setText("detailModalTitle", meta.title[state.lang] || meta.title.en);
  }
}

function toggleLanguage() {
  state.lang = state.lang === "zh" ? "en" : "zh";
  localStorage.setItem("dashboardLanguage", state.lang);
  applyLanguage();
  renderStatus(state.lastStatus);
  if (state.currentDetailType) {
    renderDetailModal(state.currentDetailType, state.currentDetailData);
  }
}

function showHint(text) {
  setText("settingsMessage", text);
}

function fillSettings(settings = {}) {
  setValue("settingRobotIp", settings.robot_ip);
  setChecked("settingLlmEnabled", settings.llm_enabled);
  setValue("settingLlmBaseUrl", settings.llm_base_url);
  setValue("settingLlmModel", settings.llm_model);
  setChecked("settingVlmEnabled", settings.vlm_enabled);
  setValue("settingVlmBaseUrl", settings.vlm_base_url);
  setValue("settingVlmModel", settings.vlm_model);
  setChecked("settingAgentEnabled", settings.agent_enabled);
  setValue("settingManualForwardSpeed", settings.manual_forward_speed);
  setValue("settingManualTurnSpeed", settings.manual_turn_speed);
  setValue("settingManualDuration", settings.manual_action_duration);
}

async function loadSettings() {
  const payload = await fetchJson("/api/settings");
  fillSettings(payload.settings || {});
  setText("settingsCoreData", JSON.stringify(payload.core || {}, null, 2));
}

function readSettings() {
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
  const payload = await fetchJson("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings: readSettings() }),
  });
  fillSettings(payload.settings || {});
  setText("settingsCoreData", JSON.stringify(payload.core || {}, null, 2));
  showHint(payload.success ? "OK" : payload.error || "failed");
}

function bindCommands() {
  document.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => sendControl(button.dataset.command));
  });

  document.getElementById("videoFeed")?.addEventListener("click", (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    sendControl("LOCK_TARGET", { x, y });
  });
}

function bindDetailTriggers() {
  document.querySelectorAll(".summary-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      openDetailModal(card.dataset.detailType);
    });
  });

  document.querySelectorAll(".detail-button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      openDetailModal(button.dataset.detailType);
    });
  });

  document.getElementById("rosDetailsToggle")?.addEventListener("click", () => openDetailModal("telemetry"));
}

function bindSettingsModal() {
  document.getElementById("settingsToggle")?.addEventListener("click", () => {
    document.getElementById("settingsModal").classList.add("open");
  });
  document.getElementById("settingsClose")?.addEventListener("click", () => {
    document.getElementById("settingsModal").classList.remove("open");
  });
  document.getElementById("settingsModal")?.addEventListener("click", (event) => {
    if (event.target?.id === "settingsModal") {
      document.getElementById("settingsModal").classList.remove("open");
    }
  });
  document.getElementById("settingsReload")?.addEventListener("click", loadSettings);
  document.getElementById("settingsSave")?.addEventListener("click", saveSettings);
}

function bindDetailModal() {
  document.getElementById("detailClose")?.addEventListener("click", closeDetailModal);
  document.getElementById("detailRefresh")?.addEventListener("click", refreshDetailModal);
  document.getElementById("detailCopy")?.addEventListener("click", copyDetailContent);
  document.getElementById("detailClear")?.addEventListener("click", clearDetailLogs);
  document.getElementById("detailModal")?.addEventListener("click", (event) => {
    if (event.target?.id === "detailModal") closeDetailModal();
  });
}

function startClock() {
  const updateClock = () => setText("clock", new Date().toLocaleTimeString());
  updateClock();
  window.setInterval(updateClock, 1000);
}

async function boot() {
  applyLanguage();
  bindCommands();
  bindDetailTriggers();
  bindSettingsModal();
  bindDetailModal();
  document.getElementById("langToggle")?.addEventListener("click", toggleLanguage);
  startClock();
  window.setInterval(refreshStatus, 1000);
  await refreshStatus();
  await loadSettings();
}

document.addEventListener("DOMContentLoaded", boot);
