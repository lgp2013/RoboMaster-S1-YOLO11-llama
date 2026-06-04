"""ROS2 Foxy node for RoboMaster S1 multi-agent following and dashboard control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import String

from .follower_control import ControlConfig, PersonFollowerController
from .gesture_controller import ControlMode, GestureController, GestureDebouncer
from .gesture_detector import GestureDetector, GestureResult, best_gesture
from .planner_agent import PlannerAgent
from .robot_executor import RobotExecutor
from .safety import SafetyGuard
from .scene_understanding import build_scene_summary
from .utils import clamp, json_string_msg, make_twist, now_sec, twist_to_dict
from .vision_agent import VisionAgent
from .web_dashboard import DASHBOARD_VERSION, DashboardServer
from .yolo_detector import PersonDetection, YoloPersonDetector


@dataclass
class LockedTarget:
    """记录当前锁定人物的几何信息。"""

    bbox: Tuple[float, float, float, float]
    selected_by: str
    locked_at: float
    label: str
    confidence: float


class InternalEventBus:
    """轻量内部事件总线，用于子智能体之间共享事件与日志。"""

    def __init__(self, maxlen: int = 80) -> None:
        self._events: Deque[Dict[str, object]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def publish(self, source: str, message: str, level: str = "info", event_type: str = "status") -> Dict[str, object]:
        """发布一条内部事件，同时返回事件对象供调用方复用。"""
        event = {
            "time": time.strftime("%H:%M:%S"),
            "timestamp": round(time.time(), 3),
            "source": source,
            "level": level,
            "type": event_type,
            "message": message,
        }
        with self._lock:
            self._events.appendleft(event)
        return event

    def snapshot(self, limit: int = 20) -> List[Dict[str, object]]:
        """读取最近事件，供 Dashboard 轮询。"""
        with self._lock:
            return list(list(self._events)[:limit])


class PersonFollowerNode(Node):
    """多子智能体控制节点：跟随、手势、VLM、LLM 和安全过滤共用一个 ROS2 主线程。"""

    def __init__(self) -> None:
        super().__init__("person_follower")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.event_bus = InternalEventBus()

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_gimbal_topic = str(self.get_parameter("cmd_gimbal_topic").value)
        self.robot_mode_topic = str(self.get_parameter("robot_mode_topic").value)
        self.robot_command_topic = str(self.get_parameter("robot_command_topic").value)
        self.led_command_topic = str(self.get_parameter("led_command_topic").value)
        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.lost_target_timeout_sec = float(self.get_parameter("lost_target_timeout_sec").value)
        self.target_distance_m = float(self.get_parameter("target_distance_m").value)
        self.gesture_enabled = bool(self.get_parameter("gesture.enabled").value)
        self.publish_debug_image = bool(self.get_parameter("gesture.publish_debug_image").value)
        self.agent_enabled = bool(self.get_parameter("agent.enabled").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.record_root = Path(self.get_parameter("record_root").value)
        self.snapshot_dir = self.record_root / "snapshots"
        self.video_dir = self.record_root / "videos"

        self.detector = YoloPersonDetector(
            model_path=str(self.get_parameter("yolo_model").value),
            confidence=float(self.get_parameter("confidence_threshold").value),
            imgsz=int(self.get_parameter("imgsz").value),
            person_class_id=int(self.get_parameter("person_class_id").value),
        )
        self.controller = PersonFollowerController(self._control_config())
        self.safety_guard = SafetyGuard(
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            command_timeout_sec=float(self.get_parameter("command_timeout_sec").value),
            image_timeout_sec=float(self.get_parameter("image_timeout_sec").value),
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.gimbal_pub = self.create_publisher(Twist, self.cmd_gimbal_topic, 10)
        self.robot_mode_pub = self.create_publisher(String, self.robot_mode_topic, 10)
        self.robot_command_pub = self.create_publisher(String, self.robot_command_topic, 10)
        self.led_command_pub = self.create_publisher(String, self.led_command_topic, 10)
        self.gesture_state_pub = self.create_publisher(String, "/gesture/state", 10)
        self.gesture_command_pub = self.create_publisher(String, "/gesture/command", 10)
        self.gesture_debug_pub = self.create_publisher(Image, "/gesture/debug_image", 5)

        self.gesture_detector = None
        if self.gesture_enabled:
            try:
                self.gesture_detector = GestureDetector(
                    max_num_hands=int(self.get_parameter("gesture.max_num_hands").value),
                    min_detection_confidence=float(self.get_parameter("gesture.min_detection_confidence").value),
                    min_tracking_confidence=float(self.get_parameter("gesture.min_tracking_confidence").value),
                )
            except Exception as exc:
                self.gesture_enabled = False
                self.gesture_detector = None
                self.get_logger().warning("MediaPipe gesture disabled: %s" % exc)
        self.gesture_debouncer = GestureDebouncer(
            stable_frame_count=int(self.get_parameter("gesture.stable_frame_count").value),
            cooldown_seconds=float(self.get_parameter("gesture.cooldown_seconds").value),
        )
        self.gesture_controller = GestureController(
            state_pub=self.gesture_state_pub,
            command_pub=self.gesture_command_pub,
            cmd_pub=self.cmd_pub,
            gesture_actions=self._gesture_actions(),
            default_mode=str(self.get_parameter("control_modes.default_mode").value),
            follow_mode_enabled=bool(self.get_parameter("control_modes.follow_mode_enabled").value),
            emergency_stop_enabled=bool(self.get_parameter("control_modes.emergency_stop_enabled").value),
            turn_angular_speed=float(self.get_parameter("gesture.turn_angular_speed").value),
            action_duration_sec=float(self.get_parameter("gesture.action_duration_sec").value),
        )
        self.vision_agent = VisionAgent(
            enabled=bool(self.get_parameter("vlm.enabled").value),
            base_url=str(self.get_parameter("vlm.base_url").value),
            model=str(self.get_parameter("vlm.model").value),
            timeout_sec=float(self.get_parameter("vlm.timeout_sec").value),
        )
        self.planner_agent = PlannerAgent(
            enabled=bool(self.get_parameter("llm.enabled").value),
            base_url=str(self.get_parameter("llm.base_url").value),
            model=str(self.get_parameter("llm.model").value),
            timeout_sec=float(self.get_parameter("llm.timeout_sec").value),
        )
        self.robot_executor = RobotExecutor(
            max_history=int(self.get_parameter("agent.max_history").value),
            forward_speed=float(self.get_parameter("agent.forward_speed").value),
            turn_speed=float(self.get_parameter("agent.turn_speed").value),
            action_duration_sec=float(self.get_parameter("agent.action_duration_sec").value),
        )

        self.latest_frame = None
        self.latest_annotated = None
        self.latest_people: List[PersonDetection] = []
        self.latest_target: Optional[PersonDetection] = None
        self.latest_gestures: List[GestureResult] = []
        self.latest_best_gesture: Optional[GestureResult] = None
        self.latest_gesture_state: Dict[str, object] = {}
        self.latest_bbox_height_ratio = 0.0
        self.latest_reason = "starting"
        self.latest_safety_reason = "starting"
        self.latest_action = "NONE"
        self.current_command = "NONE"
        self.current_control_source = "SYSTEM"
        self.last_command_time = 0.0
        self.latest_scene: Dict[str, object] = {}
        self.latest_vision_description = ""
        self.latest_agent_error = ""
        self.last_image_time = 0.0
        self.last_target_time = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.battery_percent = None

        self.manual_override_cmd: Optional[Twist] = None
        self.manual_override_gimbal_cmd: Optional[Twist] = None
        self.manual_override_started_at = 0.0
        self.manual_override_until = 0.0
        self.manual_override_action = "NONE"

        self.latest_cmd = Twist()
        self.latest_safe_cmd = Twist()
        self.latest_gimbal_cmd = Twist()
        self.latest_safe_gimbal_cmd = Twist()

        self.locked_target: Optional[LockedTarget] = None
        self.lock_state = "NONE"
        self.lock_message = "No target locked"
        self.last_lock_loss_time = 0.0
        self.follow_requested = False
        self.target_reset_count = 0
        self.locked_target_id = ""
        self.candidate_count = 0

        self.user_request = ""
        self.agent_job_running = False
        self.last_agent_plan_time = 0.0
        self.agent_lock = threading.Lock()
        self.current_agent_source = "LLM_AGENT"

        self.recording = False
        self.recording_started_at = 0.0
        self.video_writer = None
        self.video_output_path = ""
        self.last_snapshot_path = ""
        self.last_reconnect_time = 0.0

        self.dashboard = None
        if bool(self.get_parameter("dashboard_enabled").value):
            self.dashboard = DashboardServer(
                host=str(self.get_parameter("dashboard_host").value),
                port=int(self.get_parameter("dashboard_port").value),
                jpeg_quality=int(self.get_parameter("jpeg_quality").value),
                command_callback=self.handle_dashboard_command,
                settings_callback=self.handle_dashboard_settings,
            )
            self.dashboard.start()
            self.get_logger().info(
                "Flask Dashboard started at http://%s:%s"
                % (self.get_parameter("dashboard_host").value, self.get_parameter("dashboard_port").value)
            )

        self.battery_sub = self.create_subscription(BatteryState, "/battery", self.battery_callback, 10)
        self.robot_command_sub = self.create_subscription(String, self.robot_command_topic, self.robot_command_callback, 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.mode_timer = self.create_timer(0.5, self.publish_robot_mode)

        self.publish_event("SYSTEM", "system boot: dashboard online")

    def _declare_parameters(self) -> None:
        defaults = {
            "camera_topic": "/camera/image_color",
            "cmd_vel_topic": "/cmd_vel",
            "cmd_gimbal_topic": "/cmd_gimbal",
            "robot_mode_topic": "/robot/mode",
            "robot_command_topic": "/robot/command",
            "led_command_topic": "/robot/led_command",
            "robot_ip": "10.10.10.152",
            "record_root": "records",
            "yolo_model": "yolo11n.pt",
            "confidence_threshold": 0.45,
            "imgsz": 640,
            "person_class_id": 0,
            "control_rate_hz": 15.0,
            "center_threshold_px": 45,
            "yaw_gain": 1.2,
            "distance_gain": 0.8,
            "max_linear_speed": 0.30,
            "min_linear_speed": 0.06,
            "max_angular_speed": 1.0,
            "min_angular_speed": 0.10,
            "target_distance_m": 1.5,
            "target_bbox_height_ratio": 0.45,
            "bbox_height_tolerance": 0.08,
            "lost_target_timeout_sec": 0.8,
            "command_timeout_sec": 1.2,
            "image_timeout_sec": 1.0,
            "enable_backward": True,
            "enable_auto_move": True,
            "dashboard_enabled": True,
            "dashboard_host": "0.0.0.0",
            "dashboard_port": 8088,
            "jpeg_quality": 80,
            "gesture.enabled": True,
            "gesture.max_num_hands": 2,
            "gesture.min_detection_confidence": 0.6,
            "gesture.min_tracking_confidence": 0.5,
            "gesture.stable_frame_count": 5,
            "gesture.cooldown_seconds": 2.0,
            "gesture.turn_angular_speed": 0.6,
            "gesture.action_duration_sec": 1.0,
            "gesture.publish_debug_image": False,
            "gesture_actions.open_palm": "STOP",
            "gesture_actions.thumbs_up": "START_FOLLOW",
            "gesture_actions.fist": "PAUSE",
            "gesture_actions.point_left": "TURN_LEFT",
            "gesture_actions.point_right": "TURN_RIGHT",
            "gesture_actions.victory": "PATROL_READY",
            "gesture_actions.ok_sign": "RESUME",
            "control_modes.default_mode": "IDLE",
            "control_modes.follow_mode_enabled": True,
            "control_modes.emergency_stop_enabled": True,
            "llm.enabled": False,
            "llm.base_url": "http://127.0.0.1:8080/v1",
            "llm.model": "qwen3",
            "llm.timeout_sec": 8.0,
            "vlm.enabled": False,
            "vlm.base_url": "http://127.0.0.1:8080/v1",
            "vlm.model": "qwen3-vl",
            "vlm.timeout_sec": 8.0,
            "agent.enabled": False,
            "agent.max_history": 20,
            "agent.planning_interval": 2.0,
            "agent.forward_speed": 0.12,
            "agent.turn_speed": 0.45,
            "agent.action_duration_sec": 0.8,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _control_config(self) -> ControlConfig:
        return ControlConfig(
            center_threshold_px=int(self.get_parameter("center_threshold_px").value),
            yaw_gain=float(self.get_parameter("yaw_gain").value),
            distance_gain=float(self.get_parameter("distance_gain").value),
            max_linear_speed=float(self.get_parameter("max_linear_speed").value),
            min_linear_speed=float(self.get_parameter("min_linear_speed").value),
            max_angular_speed=float(self.get_parameter("max_angular_speed").value),
            min_angular_speed=float(self.get_parameter("min_angular_speed").value),
            target_bbox_height_ratio=float(self.get_parameter("target_bbox_height_ratio").value),
            bbox_height_tolerance=float(self.get_parameter("bbox_height_tolerance").value),
            enable_backward=bool(self.get_parameter("enable_backward").value),
            enable_auto_move=bool(self.get_parameter("enable_auto_move").value),
        )

    def _gesture_actions(self) -> Dict[str, str]:
        names = ["open_palm", "thumbs_up", "fist", "point_left", "point_right", "victory", "ok_sign"]
        return {name: str(self.get_parameter("gesture_actions.%s" % name).value) for name in names}

    def publish_event(self, source: str, message: str, level: str = "info", event_type: str = "status") -> None:
        """发布内部事件，并同步到 ROS 日志。"""
        event = self.event_bus.publish(source, message, level=level, event_type=event_type)
        if level == "error":
            self.get_logger().error("[%s] %s" % (source, message))
        elif level == "warning":
            self.get_logger().warning("[%s] %s" % (source, message))
        else:
            self.get_logger().info("[%s] %s" % (source, message))
        return event

    def robot_command_callback(self, msg: String) -> None:
        """监听 /robot/command，允许外部节点复用同一套控制命令。"""
        command = self._parse_robot_command(msg.data)
        if command:
            self.apply_robot_command(command, source="TOPIC")

    def _parse_robot_command(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                if str(payload.get("source", "")).lower() == "dashboard":
                    return ""
                return str(payload.get("command", "")).strip().upper()
        except Exception:
            pass
        return raw.upper()

    def publish_robot_mode(self) -> None:
        """周期发布当前模式，供其他 ROS2 节点订阅。"""
        self.robot_mode_pub.publish(String(data=str(self.gesture_controller.mode)))

    def publish_robot_command(self, command: str, payload: Optional[Dict[str, object]] = None) -> None:
        """同步发布 Dashboard 指令，便于 ROS2 外部组件感知控制操作。"""
        data = {"command": str(command).upper(), "source": "dashboard", "timestamp": time.time()}
        if payload:
            data["payload"] = payload
        self.robot_command_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def publish_led_command(self, command: str) -> None:
        """给外部 LED 桥接层一个显式的软件休眠状态信号。"""
        self.led_command_pub.publish(String(data=str(command).upper()))

    def _success(self, command: str, message: str, **extra) -> Dict[str, object]:
        result = {"success": True, "ok": True, "command": command, "message": message}
        result.update(extra)
        return result

    def _failure(self, command: str, error: str, **extra) -> Dict[str, object]:
        result = {"success": False, "ok": False, "command": command, "error": error, "message": error}
        result.update(extra)
        return result

    def _mark_command(self, command: str, source: str) -> None:
        """记录最近一次控制来源，供遥测摘要和详情弹窗显示。"""
        self.current_command = command
        self.current_control_source = source
        self.last_command_time = time.time()

    def apply_robot_command(
        self,
        command: str,
        source: str = "DASHBOARD",
        payload: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """统一处理 Dashboard、ROS Topic 和内部 Agent 指令。"""
        payload = payload or {}
        command = str(command).upper()
        self._mark_command(command, source)

        if command == "EMERGENCY_STOP":
            self.follow_requested = False
            self.clear_manual_override()
            self.gesture_controller.force_stop(source=source.lower(), gesture="command")
            self.publish_stop("%s emergency stop" % source)
            self.publish_led_command("OFF")
            self.publish_event("SAFETY_AGENT", "%s -> EMERGENCY_STOP" % source, level="warning", event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Emergency stop applied", mode=self.gesture_controller.mode)

        if self.gesture_controller.mode == ControlMode.SLEEP and command not in (
            "WAKE",
            "SLEEP",
            "EMERGENCY_STOP",
            "CLEAR_LOGS",
        ):
            self.publish_sleep_zero("sleep blocks %s" % command)
            self.publish_event("SAFETY_AGENT", "sleep blocks %s" % command, level="warning", event_type="command")
            return self._failure(command, "SLEEP mode blocks motion and agent commands", mode=self.gesture_controller.mode)

        if command == "SLEEP":
            self.enter_sleep(source)
            return self._success(command, "Robot entered software sleep", mode=self.gesture_controller.mode)
        if command == "WAKE":
            return self.wake_robot(source)
        if command == "IDLE":
            self.follow_requested = False
            self.clear_manual_override()
            self.gesture_controller.handle_web_command("PAUSE")
            self.publish_stop("%s idle" % source)
            self.publish_event("FOLLOW_AGENT", "%s -> IDLE" % source, event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Robot switched to IDLE", mode=self.gesture_controller.mode)
        if command in ("PAUSE_FOLLOW", "PAUSE"):
            self.follow_requested = False
            self.clear_manual_override()
            self.gesture_controller.handle_web_command("PAUSE")
            self.publish_stop("%s pause follow" % source)
            self.publish_event("FOLLOW_AGENT", "%s -> PAUSE_FOLLOW" % source, event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Follow paused", mode=self.gesture_controller.mode)
        if command == "START_FOLLOW":
            if self.locked_target is None:
                self.lock_state = "SCANNING"
                self.lock_message = "No locked person; switched to scanning only"
                self.publish_event("FOLLOW_AGENT", "follow paused due to no locked target", level="warning", event_type="lock")
                return self._failure(command, "Lock a person first before following", lock_state=self.lock_state)
            self.follow_requested = True
            self.gesture_controller.handle_web_command("START_FOLLOW")
            self.publish_event("FOLLOW_AGENT", "%s -> START_FOLLOW" % source, event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Follow started on locked target", mode=self.gesture_controller.mode)
        if command == "AGENT_MODE":
            self.gesture_controller.previous_mode = self.gesture_controller.mode
            self.gesture_controller.mode = ControlMode.AGENT_MODE
            self.publish_event("LLM_AGENT", "%s -> AGENT_MODE" % source, event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Agent mode enabled", mode=self.gesture_controller.mode)
        if command == "GESTURE_MODE":
            self.gesture_controller.previous_mode = self.gesture_controller.mode
            self.gesture_controller.mode = ControlMode.GESTURE_CONTROL
            self.publish_event("GESTURE_AGENT", "%s -> GESTURE_MODE" % source, event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Gesture mode enabled", mode=self.gesture_controller.mode)
        if command == "AGENT_QUERY":
            self.user_request = str(payload.get("text", "")).strip()
            self.gesture_controller.previous_mode = self.gesture_controller.mode
            self.gesture_controller.mode = ControlMode.AGENT_MODE
            self.last_agent_plan_time = 0.0
            self.publish_event("LLM_AGENT", "query: %s" % (self.user_request or "(empty)"), event_type="command")
            self.publish_robot_mode()
            return self._success(command, "Agent query queued", mode=self.gesture_controller.mode, query=self.user_request)
        if command == "LOCK_TARGET":
            x = payload.get("x")
            y = payload.get("y")
            if x is None or y is None:
                return self.lock_best_person(source)
            return self.lock_target_from_point(float(x), float(y), source=source)
        if command == "RESET_TARGET":
            self.unlock_target("%s reset target" % source)
            return self._success(command, "Target reset", lock_state=self.lock_state)
        if command == "SNAPSHOT":
            return self.take_snapshot(source)
        if command == "START_RECORD":
            return self.start_recording(source)
        if command == "STOP_RECORD":
            return self.stop_recording(source)
        if command == "RECONNECT_ROBOT":
            self.last_reconnect_time = time.time()
            self.publish_stop("%s reconnect request" % source)
            self.publish_event("SYSTEM", "%s requested reconnect; waiting for ROS driver" % source, level="warning", event_type="command")
            return self._success(command, "Reconnect signal published to ROS graph")
        if command == "CLEAR_LOGS":
            self.gesture_controller.logs.clear()
            self.robot_executor.logs.clear()
            self.event_bus = InternalEventBus()
            self.publish_event("SYSTEM", "logs cleared", event_type="command")
            return self._success(command, "Logs cleared")
        if command == "STOP":
            self.clear_manual_override()
            self.publish_stop("%s stop" % source)
            self.publish_event("SAFETY_AGENT", "%s -> STOP" % source, event_type="command")
            return self._success(command, "Stop command sent", mode=self.gesture_controller.mode)

        if command in ("FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STRAFE_LEFT", "STRAFE_RIGHT"):
            self.start_manual_override(command)
            self.publish_event("FOLLOW_AGENT", "%s manual chassis %s" % (source, command), event_type="command")
            return self._success(command, "Manual chassis command armed", mode=self.gesture_controller.mode)
        if command in ("GIMBAL_UP", "GIMBAL_DOWN", "GIMBAL_LEFT", "GIMBAL_RIGHT", "GIMBAL_CENTER"):
            self.start_manual_override(command)
            self.publish_event("GESTURE_AGENT", "%s manual gimbal %s" % (source, command), event_type="command")
            return self._success(command, "Manual gimbal command armed", mode=self.gesture_controller.mode)

        return self._failure(command, "Unsupported command")

    def enter_sleep(self, source: str) -> None:
        """进入软件休眠，强制底盘和云台持续零速度。"""
        self.follow_requested = False
        self.clear_manual_override()
        self.robot_executor.update_plan({"action": "STOP", "reason": "sleep mode", "speak": ""})
        self.gesture_controller._apply_action("SLEEP", source=source.lower(), gesture="command")
        self.publish_sleep_zero("%s sleep" % source)
        self.publish_led_command("OFF")
        self.publish_event("SAFETY_AGENT", "%s -> SLEEP" % source, level="warning", event_type="command")
        self.publish_robot_mode()

    def wake_robot(self, source: str) -> Dict[str, object]:
        """从软件休眠恢复；若仍处于急停则拒绝恢复。"""
        if self.gesture_controller.mode == ControlMode.EMERGENCY_STOP:
            self.publish_stop("WAKE blocked by EMERGENCY_STOP")
            self.publish_event("SAFETY_AGENT", "%s wake blocked by emergency stop" % source, level="warning", event_type="command")
            return self._failure("WAKE", "WAKE blocked by EMERGENCY_STOP", mode=self.gesture_controller.mode)
        self.gesture_controller._apply_action("WAKE", source=source.lower(), gesture="command")
        self.publish_led_command("ON")
        self.publish_stop("%s wake" % source)
        self.publish_event("SYSTEM", "%s -> WAKE" % source, event_type="command")
        self.publish_robot_mode()
        return self._success("WAKE", "Robot woke from software sleep", mode=self.gesture_controller.mode)

    def handle_dashboard_command(self, command: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """处理 Dashboard 按钮和视频点击选人。"""
        payload = payload or {}
        self.publish_robot_command(command, payload)
        result = self.apply_robot_command(command, source="DASHBOARD", payload=payload)
        status = self._status_dict()
        result["status"] = status
        return result

    def handle_dashboard_settings(self, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """读取或更新 Dashboard 运行时设置。"""
        if payload is None:
            return {"ok": True, "success": True, "settings": self.current_settings(), "core": self.core_data()}

        settings = payload.get("settings", payload) if isinstance(payload, dict) else {}
        if not isinstance(settings, dict):
            return self._failure("SETTINGS", "invalid settings payload")

        self.robot_ip = str(settings.get("robot_ip", self.robot_ip)).strip() or self.robot_ip
        self.planner_agent.base_url = str(settings.get("llm_base_url", self.planner_agent.base_url)).strip()
        self.planner_agent.model = str(settings.get("llm_model", self.planner_agent.model)).strip()
        self.planner_agent.enabled = bool(settings.get("llm_enabled", self.planner_agent.enabled))
        self.vision_agent.base_url = str(settings.get("vlm_base_url", self.vision_agent.base_url)).strip()
        self.vision_agent.model = str(settings.get("vlm_model", self.vision_agent.model)).strip()
        self.vision_agent.enabled = bool(settings.get("vlm_enabled", self.vision_agent.enabled))
        self.agent_enabled = bool(settings.get("agent_enabled", self.agent_enabled))

        try:
            self.robot_executor.forward_speed = clamp(
                abs(float(settings.get("manual_forward_speed", self.robot_executor.forward_speed))),
                0.0,
                float(self.get_parameter("max_linear_speed").value),
            )
            self.robot_executor.turn_speed = clamp(
                abs(float(settings.get("manual_turn_speed", self.robot_executor.turn_speed))),
                0.0,
                float(self.get_parameter("max_angular_speed").value),
            )
            self.robot_executor.action_duration_sec = clamp(
                float(settings.get("manual_action_duration", self.robot_executor.action_duration_sec)),
                0.1,
                2.0,
            )
        except Exception as exc:
            return self._failure("SETTINGS", "invalid speed setting: %s" % exc)

        self.publish_event("SYSTEM", "settings updated from dashboard", event_type="settings")
        # 汇总每个子智能体的摘要输入/输出，供右侧卡片和详情弹窗共用。
        now_text = time.strftime("%H:%M:%S")
        latest_plan = dict(self.robot_executor.last_plan)
        latest_plan_action = str(latest_plan.get("action", "STOP"))
        gesture_state = dict(self.latest_gesture_state)
        gesture_output = gesture_state.get("stable_gesture", "none")
        return {
            "ok": True,
            "success": True,
            "settings": self.current_settings(),
            "core": self.core_data(),
            "event_logs": self.event_bus.snapshot(),
        }

    def current_settings(self) -> Dict[str, object]:
        """返回可在设置弹窗中编辑的运行时配置。"""
        return {
            "robot_ip": self.robot_ip,
            "llm_enabled": self.planner_agent.enabled,
            "llm_base_url": self.planner_agent.base_url,
            "llm_model": self.planner_agent.model,
            "vlm_enabled": self.vision_agent.enabled,
            "vlm_base_url": self.vision_agent.base_url,
            "vlm_model": self.vision_agent.model,
            "agent_enabled": self.agent_enabled,
            "manual_forward_speed": self.robot_executor.forward_speed,
            "manual_turn_speed": self.robot_executor.turn_speed,
            "manual_action_duration": self.robot_executor.action_duration_sec,
        }

    def core_data(self) -> Dict[str, object]:
        """设置弹窗展示的核心运行数据。"""
        camera_online = (time.time() - self.last_image_time) < 2.0 if self.last_image_time else False
        return {
            "dashboard_version": DASHBOARD_VERSION,
            "camera_topic": self.camera_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "cmd_gimbal_topic": self.cmd_gimbal_topic,
            "robot_mode_topic": self.robot_mode_topic,
            "robot_command_topic": self.robot_command_topic,
            "led_command_topic": self.led_command_topic,
            "robot_ip": self.robot_ip,
            "battery": self.battery_percent,
            "mode": self.gesture_controller.mode,
            "sleeping": self.gesture_controller.mode == ControlMode.SLEEP,
            "fps": round(self.fps, 2),
            "camera_status": "online" if camera_online else "offline",
            "recording": self.recording,
            "lock_state": self.lock_state,
        }

    def start_manual_override(self, command: str) -> None:
        """启动短时手动接管。底盘和云台走同一套时间窗，便于统一安全收口。"""
        chassis = Twist()
        gimbal = Twist()
        linear = abs(float(self.robot_executor.forward_speed))
        angular = abs(float(self.robot_executor.turn_speed))
        gimbal_speed = min(self.max_angular_speed, max(0.2, angular))

        if command == "FORWARD":
            chassis.linear.x = linear
        elif command == "BACKWARD":
            chassis.linear.x = -linear
        elif command == "TURN_LEFT":
            chassis.angular.z = angular
        elif command == "TURN_RIGHT":
            chassis.angular.z = -angular
        elif command == "STRAFE_LEFT":
            chassis.linear.y = linear
        elif command == "STRAFE_RIGHT":
            chassis.linear.y = -linear
        elif command == "GIMBAL_UP":
            gimbal.angular.y = gimbal_speed
        elif command == "GIMBAL_DOWN":
            gimbal.angular.y = -gimbal_speed
        elif command == "GIMBAL_LEFT":
            gimbal.angular.z = gimbal_speed
        elif command == "GIMBAL_RIGHT":
            gimbal.angular.z = -gimbal_speed
        elif command == "GIMBAL_CENTER":
            gimbal = Twist()

        self.manual_override_cmd = chassis
        self.manual_override_gimbal_cmd = gimbal
        self.manual_override_action = command
        self.manual_override_started_at = time.time()
        self.manual_override_until = self.manual_override_started_at + float(self.robot_executor.action_duration_sec)

    def clear_manual_override(self) -> None:
        """停止并清理手动接管状态。"""
        self.manual_override_cmd = None
        self.manual_override_gimbal_cmd = None
        self.manual_override_started_at = 0.0
        self.manual_override_until = 0.0
        self.manual_override_action = "NONE"

    def battery_callback(self, msg: BatteryState) -> None:
        """读取 /battery。robomaster_ros 通常发布 BatteryState。"""
        try:
            if msg.percentage >= 0.0:
                self.battery_percent = int(round(float(msg.percentage) * 100.0))
        except Exception:
            self.battery_percent = None

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error("cv_bridge convert failed: %s" % exc)
            self.publish_stop("cv_bridge failed")
            return

        try:
            people = self.detector.detect(frame)
            self.candidate_count = len(people)
            tracked_target = self._resolve_locked_target(people)
            highlighted = tracked_target if tracked_target is not None else self.detector.select_largest_person(people)
            if self.locked_target is None:
                if people:
                    self.lock_state = "CANDIDATE"
                    self.lock_message = "Detected candidate person; waiting for lock"
                else:
                    self.lock_state = "NONE"
                    self.lock_message = "No person detected"
            annotated = frame.copy()
            self.detector.draw_detections(annotated, people, highlighted)

            gestures: List[GestureResult] = []
            best = None
            sleep_mode = self.gesture_controller.mode == ControlMode.SLEEP
            if self.gesture_detector is not None and not sleep_mode:
                gestures = self.gesture_detector.detect(frame)
                self.gesture_detector.draw(annotated, gestures)
                best = best_gesture(gestures)

            triggered, gesture_state = self.gesture_debouncer.update(best)
            if best is not None and best.gesture == "open_palm" and not sleep_mode:
                self.gesture_controller.force_stop(source="gesture_immediate", gesture="open_palm")
                self.publish_event("GESTURE_AGENT", "open_palm immediate stop", level="warning", event_type="gesture")
                self.publish_stop("open_palm immediate stop")
            if triggered is not None and not sleep_mode:
                self.gesture_controller.handle_stable_gesture(triggered)
                self.publish_event("GESTURE_AGENT", "stable gesture: %s" % triggered, event_type="gesture")

            self.gesture_controller.publish_state(gesture_state, best)
            self._update_fps()
            self._draw_overlay(annotated, gesture_state, tracked_target)

            now = now_sec()
            with self.lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                self.latest_people = people
                self.latest_target = tracked_target
                self.latest_gestures = gestures
                self.latest_best_gesture = best
                self.latest_gesture_state = gesture_state
                self.last_image_time = now
                if tracked_target is not None:
                    self.last_target_time = now

            self._write_recording_frame(annotated)

            if not sleep_mode:
                self._maybe_start_agent_job(frame, people, gestures, tracked_target)

            if self.publish_debug_image:
                self.gesture_debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

            if self.dashboard is not None:
                self.dashboard.update(annotated, self._status_dict())
        except Exception as exc:
            self.publish_event("SYSTEM", "image processing failed: %s" % exc, level="error", event_type="vision")
            self.publish_stop("image processing exception")

    def _draw_overlay(
        self,
        frame,
        gesture_state: Dict[str, object],
        tracked_target: Optional[PersonDetection],
    ) -> None:
        """在 OpenCV 帧上叠加模式、锁定状态和当前动作。"""
        lines = [
            "mode: %s" % self.gesture_controller.mode,
            "follow lock: %s" % self.lock_state,
            "lock msg: %s" % self.lock_message[:48],
            "candidate: %s x%s" % (gesture_state.get("candidate", "none"), gesture_state.get("candidate_count", 0)),
            "stable: %s" % gesture_state.get("stable_gesture", "none"),
            "action: %s" % self.latest_action,
        ]
        y = 24
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24
        if tracked_target is not None:
            cv2.putText(
                frame,
                "LOCKED",
                (int(tracked_target.x1), max(18, int(tracked_target.y1) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _update_fps(self) -> None:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

    def _maybe_start_agent_job(
        self,
        frame,
        people: List[PersonDetection],
        gestures: List[GestureResult],
        target: Optional[PersonDetection],
    ) -> None:
        """按固定间隔启动后台 Agent 推理，避免阻塞图像主回调。"""
        if not self.agent_enabled:
            return
        if self.gesture_controller.mode == ControlMode.SLEEP:
            return
        if self.gesture_controller.mode != ControlMode.AGENT_MODE:
            return
        now = now_sec()
        interval = float(self.get_parameter("agent.planning_interval").value)
        with self.agent_lock:
            if self.agent_job_running or now - self.last_agent_plan_time < interval:
                return
            self.agent_job_running = True
            self.last_agent_plan_time = now

        thread = threading.Thread(
            target=self._run_agent_job,
            args=(frame.copy(), list(people), list(gestures), target),
            name="vision-language-agent",
            daemon=True,
        )
        thread.start()

    def _run_agent_job(
        self,
        frame,
        people: List[PersonDetection],
        gestures: List[GestureResult],
        target: Optional[PersonDetection],
    ) -> None:
        """后台调用 VLM 和 LLM，生成高级动作建议。"""
        try:
            if self.gesture_controller.mode == ControlMode.SLEEP:
                return
            height, width = frame.shape[:2]
            vision = self.vision_agent.describe(frame)
            scene = build_scene_summary(
                people=people,
                gestures=gestures,
                target=target,
                frame_width=width,
                frame_height=height,
                vision_description=vision.description,
            )
            planner = self.planner_agent.plan(
                scene=scene,
                robot_state=self._status_dict(),
                mode=self.gesture_controller.mode,
                user_request=self.user_request,
            )
            self.robot_executor.update_plan(planner.plan, planner.latency_ms, planner.tokens)
            with self.lock:
                self.latest_scene = scene
                self.latest_vision_description = vision.description
                self.latest_agent_error = vision.error or planner.error
            self.publish_event(
                "LLM_AGENT",
                "plan=%s reason=%s" % (planner.plan.get("action"), planner.plan.get("reason")),
                event_type="agent",
            )
        except Exception as exc:
            self.latest_agent_error = str(exc)
            self.robot_executor.update_plan({"action": "STOP", "reason": "agent exception: %s" % exc, "speak": ""})
            self.publish_event("LLM_AGENT", "agent exception: %s" % exc, level="error", event_type="agent")
        finally:
            with self.agent_lock:
                self.agent_job_running = False

    def control_loop(self) -> None:
        with self.lock:
            frame = self.latest_frame
            target = self.latest_target
            last_image_time = self.last_image_time
            last_target_time = self.last_target_time

        mode = self.gesture_controller.mode
        if mode == ControlMode.SLEEP:
            self.publish_sleep_zero("mode SLEEP")
            return
        if mode == ControlMode.EMERGENCY_STOP:
            self.publish_stop("mode EMERGENCY_STOP")
            return
        if frame is None:
            self.publish_stop("no image")
            return

        now = now_sec()
        image_age = now - last_image_time if last_image_time else 999.0
        if image_age > float(self.get_parameter("image_timeout_sec").value):
            self.publish_stop("image timeout")
            return

        if self.manual_override_cmd is not None:
            if time.time() <= self.manual_override_until:
                command_age = max(0.0, now - self.manual_override_started_at)
                safe_cmd, safety_reason = self.safety_guard.filter_cmd(self.manual_override_cmd, True, image_age, command_age)
                gimbal_cmd = self._clamp_gimbal_cmd(self.manual_override_gimbal_cmd or Twist())
                self.latest_action = "MANUAL_%s" % self.manual_override_action
                self._publish_cmd(safe_cmd, self.manual_override_cmd, gimbal_cmd, self.manual_override_gimbal_cmd or Twist(), "manual override", safety_reason)
                return
            self.clear_manual_override()

        override_cmd, action_name = self.gesture_controller.get_override_cmd()
        self.latest_action = action_name
        if mode in (ControlMode.IDLE, ControlMode.PATROL_READY):
            self.publish_stop("mode %s" % mode)
            return

        if mode == ControlMode.GESTURE_CONTROL and override_cmd is not None:
            command_age = max(0.0, now - self.gesture_controller.action_started_at)
            safe_cmd, safety_reason = self.safety_guard.filter_cmd(override_cmd, True, image_age, command_age)
            self._publish_cmd(safe_cmd, override_cmd, Twist(), Twist(), "gesture action", safety_reason)
            return

        if mode == ControlMode.AGENT_MODE:
            next_mode, agent_cmd, agent_reason = self.robot_executor.resolve(mode)
            if next_mode == ControlMode.FOLLOW:
                if self.locked_target is None:
                    self.publish_event("FOLLOW_AGENT", "FOLLOW_PERSON ignored: no locked target", level="warning", event_type="lock")
                    self.publish_stop("agent follow blocked")
                    return
                self.follow_requested = True
                self.gesture_controller.mode = ControlMode.FOLLOW
                mode = ControlMode.FOLLOW
            elif next_mode in (ControlMode.IDLE, ControlMode.PATROL_READY):
                self.gesture_controller.mode = next_mode
                self.publish_stop("agent mode %s: %s" % (next_mode, agent_reason))
                return
            elif agent_cmd is not None:
                command_age = max(0.0, now - self.robot_executor.active_started_at)
                safe_cmd, safety_reason = self.safety_guard.filter_cmd(agent_cmd, True, image_age, command_age)
                self._publish_cmd(safe_cmd, agent_cmd, Twist(), Twist(), "agent action: %s" % agent_reason, safety_reason)
                return
            else:
                self.publish_stop("agent waiting")
                return

        if mode != ControlMode.FOLLOW:
            self.publish_stop("mode %s" % mode)
            return
        if not self.follow_requested:
            self.publish_stop("follow not requested")
            return
        if self.locked_target is None:
            self.publish_stop("no locked target")
            return

        has_fresh_target = target is not None and (now - last_target_time) <= self.lost_target_timeout_sec
        if not has_fresh_target:
            self.publish_stop("target lost")
            if time.time() - self.last_lock_loss_time > 1.0:
                self.last_lock_loss_time = time.time()
                self.lock_state = "LOST"
                self.lock_message = "Locked target lost; robot stopped"
                self.publish_event("FOLLOW_AGENT", "locked target lost; stop", level="warning", event_type="lock")
            return

        height, width = frame.shape[:2]
        cmd, reason, bbox_height_ratio = self.controller.compute_cmd(target, width, height)
        safe_cmd, safety_reason = self.safety_guard.filter_cmd(cmd, True, image_age)
        self.latest_bbox_height_ratio = bbox_height_ratio
        self.lock_state = "LOCKED"
        self.lock_message = "Locked target tracked"
        self._publish_cmd(safe_cmd, cmd, Twist(), Twist(), reason, safety_reason)

    def _publish_cmd(
        self,
        safe_cmd: Twist,
        raw_cmd: Twist,
        safe_gimbal_cmd: Twist,
        raw_gimbal_cmd: Twist,
        reason: str,
        safety_reason: str,
    ) -> None:
        self.cmd_pub.publish(safe_cmd)
        self.gimbal_pub.publish(safe_gimbal_cmd)
        self.latest_cmd = raw_cmd
        self.latest_safe_cmd = safe_cmd
        self.latest_gimbal_cmd = raw_gimbal_cmd
        self.latest_safe_gimbal_cmd = safe_gimbal_cmd
        self.latest_reason = reason
        self.latest_safety_reason = safety_reason

    def _clamp_gimbal_cmd(self, cmd: Twist) -> Twist:
        """把云台指令限制在安全角速度范围内。"""
        safe = Twist()
        safe.angular.y = clamp(float(cmd.angular.y), -self.max_angular_speed, self.max_angular_speed)
        safe.angular.z = clamp(float(cmd.angular.z), -self.max_angular_speed, self.max_angular_speed)
        return safe

    def publish_stop(self, reason: str = "stop") -> None:
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.gimbal_pub.publish(stop)
        self.latest_cmd = stop
        self.latest_safe_cmd = stop
        self.latest_gimbal_cmd = stop
        self.latest_safe_gimbal_cmd = stop
        self.latest_reason = reason
        self.latest_safety_reason = "stop"

    def publish_sleep_zero(self, reason: str = "sleep") -> None:
        """SLEEP 模式持续输出底盘和云台零速度。"""
        self.publish_stop(reason)
        self.latest_safety_reason = "sleep zero"

    def lock_best_person(self, source: str) -> Dict[str, object]:
        """锁定当前帧里最显著的人物目标。"""
        with self.lock:
            people = list(self.latest_people)
        target = self.detector.select_largest_person(people)
        if target is None:
            self.lock_state = "CANDIDATE" if people else "NONE"
            self.lock_message = "No person detected to lock"
            return self._failure("LOCK_TARGET", "No detected person available")
        self._store_locked_target(target, selected_by=source)
        self.publish_event("FOLLOW_AGENT", "%s locked best person" % source, event_type="lock")
        return self._success("LOCK_TARGET", "Best detected person locked", lock_state=self.lock_state)

    def lock_target_from_point(self, x: float, y: float, source: str = "DASHBOARD") -> Dict[str, object]:
        """根据视频点击位置锁定人物。x/y 使用 0..1 归一化坐标。"""
        with self.lock:
            people = list(self.latest_people)
        if not people:
            return self._failure("LOCK_TARGET", "No detected person available")

        width = max(1.0, float(self.latest_frame.shape[1])) if self.latest_frame is not None else 1.0
        height = max(1.0, float(self.latest_frame.shape[0])) if self.latest_frame is not None else 1.0
        px = clamp(float(x), 0.0, 1.0) * width
        py = clamp(float(y), 0.0, 1.0) * height

        containing = [person for person in people if person.x1 <= px <= person.x2 and person.y1 <= py <= person.y2]
        if containing:
            target = max(containing, key=lambda item: item.area)
        else:
            target = min(people, key=lambda item: ((item.center_x - px) ** 2 + (item.center_y - py) ** 2))

        self._store_locked_target(target, selected_by=source)
        self.publish_event("FOLLOW_AGENT", "%s clicked and locked target" % source, event_type="lock")
        return self._success("LOCK_TARGET", "Target locked from video click", lock_state=self.lock_state)

    def unlock_target(self, reason: str) -> None:
        """清空锁定目标，但保留最后一次失锁说明。"""
        self.locked_target = None
        self.locked_target_id = ""
        self.follow_requested = False
        self.lock_state = "NONE"
        self.lock_message = reason
        self.publish_event("FOLLOW_AGENT", reason, level="warning", event_type="lock")

    def _store_locked_target(self, target: PersonDetection, selected_by: str) -> None:
        """把当前检测结果固化为锁定目标。"""
        self.locked_target = LockedTarget(
            bbox=(target.x1, target.y1, target.x2, target.y2),
            selected_by=selected_by,
            locked_at=time.time(),
            label="person",
            confidence=float(target.confidence),
        )
        self.locked_target_id = "person-%s" % int(self.locked_target.locked_at * 1000)
        self.lock_state = "LOCKED"
        self.lock_message = "Locked target ready for follow"

    def _resolve_locked_target(self, people: List[PersonDetection]) -> Optional[PersonDetection]:
        """根据上一帧锁定框，在本帧里继续匹配同一人物。"""
        if self.locked_target is None:
            self.lock_state = "NONE"
            self.lock_message = "No target locked"
            return None
        if not people:
            self.lock_state = "LOST"
            self.lock_message = "Locked target missing in current frame"
            return None

        best = None
        best_score = 0.0
        for person in people:
            score = self._iou(self.locked_target.bbox, (person.x1, person.y1, person.x2, person.y2))
            if score > best_score:
                best = person
                best_score = score
        if best is None:
            self.lock_state = "LOST"
            self.lock_message = "Locked target missing in current frame"
            return None

        if best_score < 0.05:
            previous_cx = (self.locked_target.bbox[0] + self.locked_target.bbox[2]) / 2.0
            previous_cy = (self.locked_target.bbox[1] + self.locked_target.bbox[3]) / 2.0
            best = min(
                people,
                key=lambda item: ((item.center_x - previous_cx) ** 2 + (item.center_y - previous_cy) ** 2),
            )
        self.locked_target.bbox = (best.x1, best.y1, best.x2, best.y2)
        self.locked_target.confidence = float(best.confidence)
        self.lock_state = "LOCKED"
        self.lock_message = "Locked target tracked"
        return best

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """计算两个检测框的 IoU，用于轻量目标重关联。"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = max(1e-6, a_area + b_area - inter_area)
        return inter_area / denom

    def _write_recording_frame(self, frame) -> None:
        """在录制开启时把最新画面写入视频文件。"""
        if not self.recording or self.video_writer is None:
            return
        self.video_writer.write(frame)

    def take_snapshot(self, source: str) -> Dict[str, object]:
        """保存当前标注画面到 records/snapshots。"""
        with self.lock:
            frame = None if self.latest_annotated is None else self.latest_annotated.copy()
        if frame is None:
            return self._failure("SNAPSHOT", "No frame available")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / ("snapshot_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
        cv2.imwrite(str(path), frame)
        self.last_snapshot_path = str(path)
        self.publish_event("SYSTEM", "%s saved snapshot %s" % (source, path.name), event_type="record")
        return self._success("SNAPSHOT", "Snapshot saved", path=str(path))

    def start_recording(self, source: str) -> Dict[str, object]:
        """启动视频录制，保存到 records/videos。"""
        if self.recording:
            return self._success("START_RECORD", "Recording already running", path=self.video_output_path)
        with self.lock:
            frame = None if self.latest_annotated is None else self.latest_annotated.copy()
        if frame is None:
            return self._failure("START_RECORD", "No frame available for recording")

        self.video_dir.mkdir(parents=True, exist_ok=True)
        path = self.video_dir / ("record_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(5.0, float(self.fps) or 15.0),
            (width, height),
        )
        if not writer.isOpened():
            return self._failure("START_RECORD", "Failed to open video writer")
        self.video_writer = writer
        self.recording = True
        self.recording_started_at = time.time()
        self.video_output_path = str(path)
        self.publish_event("SYSTEM", "%s started recording %s" % (source, path.name), event_type="record")
        return self._success("START_RECORD", "Recording started", path=str(path))

    def stop_recording(self, source: str) -> Dict[str, object]:
        """停止录制并释放 VideoWriter。"""
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        if not self.recording:
            return self._success("STOP_RECORD", "Recording already stopped")
        duration = round(time.time() - self.recording_started_at, 2)
        path = self.video_output_path
        self.recording = False
        self.recording_started_at = 0.0
        self.video_output_path = ""
        self.publish_event("SYSTEM", "%s stopped recording" % source, event_type="record")
        return self._success("STOP_RECORD", "Recording stopped", path=path, duration_seconds=duration)

    def _agent_status_map(self) -> Dict[str, Dict[str, object]]:
        """汇总每个子智能体当前状态，供 Dashboard 渲染。"""
        return {
            "FOLLOW_AGENT": {
                "status": self.lock_state if self.follow_requested or self.lock_state not in ("NONE", "") else "READY",
                "enabled": True,
                "last_event": self.lock_message,
                "last_update": now_text,
                "lock_state": self.lock_state,
                "message": self.lock_message,
                "input": {
                    "people_count": len(self.latest_people),
                    "candidate_count": self.candidate_count,
                    "locked_target_id": self.locked_target_id or None,
                },
                "output": {
                    "follow_enabled": self.follow_requested,
                    "follow_active": self.gesture_controller.mode == ControlMode.FOLLOW and self.follow_requested,
                    "current_target": "person" if self.latest_target is not None else "none",
                },
                "current_action": "FOLLOW" if self.follow_requested else "SCAN",
                "last_error": "target lost" if self.lock_state == "LOST" else "",
            },
            "GESTURE_AGENT": {
                "status": "ONLINE" if self.gesture_enabled and self.gesture_detector is not None else "IDLE",
                "enabled": self.gesture_enabled,
                "last_event": gesture_output,
                "last_update": now_text,
                "message": gesture_output,
                "input": {
                    "candidate": gesture_state.get("candidate", "none"),
                    "candidate_count": gesture_state.get("candidate_count", 0),
                    "cooldown_remaining": gesture_state.get("cooldown_remaining", 0.0),
                },
                "output": {
                    "stable_gesture": gesture_output,
                    "command_mode": str(self.gesture_controller.mode),
                },
                "current_action": gesture_output,
                "last_error": "",
            },
            "VLM_AGENT": {
                "status": "ONLINE" if self.vision_agent.enabled else "STANDBY",
                "enabled": self.vision_agent.enabled,
                "last_event": self.latest_vision_description or "No VLM description yet",
                "last_update": now_text,
                "message": self.latest_vision_description or "No VLM description yet",
                "input": {
                    "scene_enabled": self.vision_agent.enabled,
                    "camera_online": (time.time() - self.last_image_time) < 2.0 if self.last_image_time else False,
                },
                "output": {
                    "scene_text": self.latest_scene.get("description", ""),
                    "vision_description": self.latest_vision_description,
                },
                "current_action": "DESCRIBE_SCENE",
                "last_error": self.latest_agent_error if self.vision_agent.enabled else "",
            },
            "LLM_AGENT": {
                "status": "THINKING" if self.agent_job_running else ("ONLINE" if self.agent_enabled else "STANDBY"),
                "enabled": self.agent_enabled,
                "last_event": latest_plan_action,
                "last_update": now_text,
                "message": latest_plan_action,
                "input": {
                    "user_request": self.user_request,
                    "scene_summary": self.latest_scene.get("description", ""),
                    "current_source": self.current_agent_source,
                },
                "output": {
                    "last_plan": latest_plan,
                    "last_tokens": dict(self.robot_executor.last_tokens),
                    "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                },
                "current_action": latest_plan_action,
                "last_error": self.latest_agent_error if self.agent_enabled else "",
            },
            "SAFETY_AGENT": {
                "status": "ACTIVE",
                "enabled": True,
                "last_event": self.latest_safety_reason,
                "last_update": now_text,
                "message": self.latest_safety_reason,
                "input": {
                    "raw_cmd": twist_to_dict(self.latest_cmd),
                    "raw_gimbal_cmd": twist_to_dict(self.latest_gimbal_cmd),
                },
                "output": {
                    "safe_cmd": twist_to_dict(self.latest_safe_cmd),
                    "safe_gimbal_cmd": twist_to_dict(self.latest_safe_gimbal_cmd),
                },
                "current_action": self.current_command,
                "last_error": "",
            },
            "MANUAL_AGENT": {
                "status": "ACTIVE" if self.manual_override_cmd is not None else "IDLE",
                "enabled": True,
                "last_event": self.manual_override_action,
                "last_update": now_text,
                "message": self.manual_override_action,
                "input": {
                    "control_source": self.current_control_source,
                    "override_until": round(self.manual_override_until, 3) if self.manual_override_until else None,
                },
                "output": {
                    "action": self.manual_override_action,
                    "cmd_active": self.manual_override_cmd is not None,
                },
                "current_action": self.manual_override_action,
                "last_error": "",
            },
            "DASHBOARD_AGENT": {
                "status": "ONLINE",
                "enabled": True,
                "last_event": self.current_command,
                "last_update": now_text,
                "message": self.current_control_source,
                "input": {
                    "last_command": self.current_command,
                    "last_command_time": round(self.last_command_time, 3) if self.last_command_time else None,
                },
                "output": {
                    "dashboard_version": DASHBOARD_VERSION,
                    "detail_endpoints": [
                        "/api/detail/telemetry",
                        "/api/detail/agent",
                        "/api/detail/logs",
                        "/api/detail/models",
                        "/api/detail/sub_agents",
                    ],
                },
                "current_action": self.current_command,
                "last_error": "",
            },
        }

    def _status_dict(self) -> Dict[str, object]:
        """输出给 Dashboard 的完整运行状态。"""
        target = self.latest_target
        target_info = None
        if target is not None:
            target_info = {
                "bbox": [round(target.x1, 1), round(target.y1, 1), round(target.x2, 1), round(target.y2, 1)],
                "confidence": round(target.confidence, 3),
                "area": round(target.area, 1),
                "center_x": round(target.center_x, 1),
                "center_y": round(target.center_y, 1),
                "bbox_height_ratio": round(self.latest_bbox_height_ratio, 3),
            }

        gesture = self.latest_best_gesture
        gesture_info = {
            "current": gesture.gesture if gesture is not None else "none",
            "confidence": round(gesture.confidence, 3) if gesture is not None else 0.0,
            "candidate": self.latest_gesture_state.get("candidate", "none"),
            "candidate_count": self.latest_gesture_state.get("candidate_count", 0),
            "stable_gesture": self.latest_gesture_state.get("stable_gesture", "none"),
            "cooldown_remaining": self.latest_gesture_state.get("cooldown_remaining", 0.0),
        }
        camera_online = (time.time() - self.last_image_time) < 2.0 if self.last_image_time else False
        sub_agents = self._agent_status_map()
        model_name = self.planner_agent.model or self.vision_agent.model or "unknown"
        summary_scene = self.latest_vision_description or self.latest_scene.get("description", "暂无场景报告")
        summary_plan = str(self.robot_executor.last_plan.get("action", "STOP"))
        telemetry_summary = {
            "connection": "CONNECTED" if camera_online else "DISCONNECTED",
            "battery": self.battery_percent,
            "mode": self.gesture_controller.mode,
            "linear_x": round(float(self.latest_safe_cmd.linear.x), 3),
            "angular_z": round(float(self.latest_safe_cmd.angular.z), 3),
            "control_source": self.current_control_source,
            "target_summary": self.lock_state,
            "model_name_short": (model_name[:12] + "...") if len(model_name) > 15 else model_name,
        }
        agent_summary = {
            "status": sub_agents["LLM_AGENT"]["status"],
            "scene_summary": summary_scene[:96],
            "action_summary": summary_plan,
            "current_action": summary_plan,
            "latency_ms": round(self.robot_executor.last_latency_ms, 1),
            "token_summary": dict(self.robot_executor.last_tokens),
        }
        logs_snapshot = self.event_bus.snapshot(limit=100)
        event_summary = [
            {
                "time": item.get("time", ""),
                "source": item.get("source", "SYSTEM"),
                "message": str(item.get("message", ""))[:80],
                "level": item.get("level", "info"),
            }
            for item in logs_snapshot[:5]
        ]
        model_runtime = {
            "YOLO": {"status": "online" if self.detector is not None else "offline", "model_name": str(self.detector.model_path)},
            "HAND": {"status": "online" if self.gesture_enabled and self.gesture_detector is not None else "offline", "model_name": "MediaPipe Hands"},
            "VLM": {
                "status": sub_agents["VLM_AGENT"]["status"],
                "model_name": self.vision_agent.model,
                "base_url": self.vision_agent.base_url,
                "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                "gpu": "unknown",
                "torch_version": "unknown",
                "cuda_status": "unknown",
                "last_error": self.latest_agent_error,
            },
            "LLM": {
                "status": sub_agents["LLM_AGENT"]["status"],
                "model_name": self.planner_agent.model,
                "base_url": self.planner_agent.base_url,
                "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                "gpu": "unknown",
                "torch_version": "unknown",
                "cuda_status": "unknown",
                "last_error": self.latest_agent_error,
            },
            "AGENT": {"status": sub_agents["LLM_AGENT"]["status"], "model_name": self.planner_agent.model},
        }
        detail_payload = {
            "telemetry": {
                "battery": self.battery_percent,
                "connection": "CONNECTED" if camera_online else "DISCONNECTED",
                "mode": self.gesture_controller.mode,
                "control_source": self.current_control_source,
                "linear_x": round(float(self.latest_safe_cmd.linear.x), 3),
                "linear_y": round(float(self.latest_safe_cmd.linear.y), 3),
                "angular_z": round(float(self.latest_safe_cmd.angular.z), 3),
                "gimbal_yaw": round(float(self.latest_safe_gimbal_cmd.angular.z), 3),
                "gimbal_pitch": round(float(self.latest_safe_gimbal_cmd.angular.y), 3),
                "target": target_info,
                "gesture": gesture_info,
                "current_command": self.current_command,
                "last_command_time": round(self.last_command_time, 3) if self.last_command_time else None,
                "model_name": model_name,
                "llm_status": sub_agents["LLM_AGENT"]["status"],
                "camera_status": "online" if camera_online else "offline",
                "yolo_status": "online" if self.detector is not None else "offline",
                "hand_status": "online" if self.gesture_enabled and self.gesture_detector is not None else "offline",
                "agent_status": sub_agents["LLM_AGENT"]["status"],
                "raw_status_json": {
                    "connected": camera_online,
                    "camera_status": "online" if camera_online else "offline",
                    "follow_lock_state": self.lock_state,
                    "follow_lock_message": self.lock_message,
                    "current_command": self.current_command,
                    "current_control_source": self.current_control_source,
                    "recording": self.recording,
                },
            },
            "agent": {
                "scene_text": self.latest_scene.get("description", ""),
                "action_plan_text": summary_plan,
                "llm_request": self.user_request,
                "llm_response": dict(self.robot_executor.last_plan),
                "current_action": summary_plan,
                "reason": str(self.robot_executor.last_plan.get("reason", "")),
                "tokens": dict(self.robot_executor.last_tokens),
                "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                "events": list(self.robot_executor.logs)[:20],
                "raw_agent_json": {
                    "scene": self.latest_scene,
                    "agent": {
                        "enabled": self.agent_enabled,
                        "status": sub_agents["LLM_AGENT"]["status"],
                        "vision_description": self.latest_vision_description,
                        "last_plan": dict(self.robot_executor.last_plan),
                        "logs": list(self.robot_executor.logs),
                        "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                        "tokens": dict(self.robot_executor.last_tokens),
                        "job_running": self.agent_job_running,
                        "error": self.latest_agent_error,
                        "user_request": self.user_request,
                    },
                },
            },
            "logs": {
                "items": logs_snapshot,
                "filters": [
                    "ALL",
                    "DASHBOARD",
                    "SYSTEM",
                    "SAFETY_AGENT",
                    "FOLLOW_AGENT",
                    "GESTURE_AGENT",
                    "VLM_AGENT",
                    "LLM_AGENT",
                    "MANUAL_AGENT",
                    "ERROR",
                ],
            },
            "models": model_runtime,
            "sub_agents": {
                "items": sub_agents,
                "follow_meta": {
                    "candidate_count": self.candidate_count,
                    "locked_target_id": self.locked_target_id,
                    "locked_bbox": [round(v, 1) for v in self.locked_target.bbox] if self.locked_target else None,
                    "target_confidence": round(self.locked_target.confidence, 3) if self.locked_target else None,
                    "target_lost_seconds": round(time.time() - self.last_target_time, 2) if self.last_target_time else None,
                    "follow_enabled": self.follow_requested,
                    "follow_active": self.gesture_controller.mode == ControlMode.FOLLOW and self.follow_requested,
                    "lock_state": self.lock_state,
                },
            },
        }
        return {
            "connected": camera_online,
            "dashboard_version": DASHBOARD_VERSION,
            "battery": self.battery_percent,
            "gesture_name": gesture_info["current"],
            "target_name": "person" if target is not None else "none",
            "linear_x": round(float(self.latest_safe_cmd.linear.x), 3),
            "linear_y": round(float(self.latest_safe_cmd.linear.y), 3),
            "angular_z": round(float(self.latest_safe_cmd.angular.z), 3),
            "gimbal_pitch": round(float(self.latest_safe_gimbal_cmd.angular.y), 3),
            "gimbal_yaw": round(float(self.latest_safe_gimbal_cmd.angular.z), 3),
            "yolo_status": "online" if self.detector is not None else "offline",
            "hand_status": "online" if self.gesture_enabled and self.gesture_detector is not None else "standby",
            "vlm_status": sub_agents["VLM_AGENT"]["status"],
            "llm_status": sub_agents["LLM_AGENT"]["status"],
            "agent_status": sub_agents["LLM_AGENT"]["status"],
            "nav_status": "standby",
            "camera_status": "online" if camera_online else "offline",
            "fps": round(self.fps, 2),
            "mode": self.gesture_controller.mode,
            "camera_topic": self.camera_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "cmd_gimbal_topic": self.cmd_gimbal_topic,
            "robot_mode_topic": self.robot_mode_topic,
            "robot_command_topic": self.robot_command_topic,
            "led_command_topic": self.led_command_topic,
            "robot_ip": self.robot_ip,
            "people_count": len(self.latest_people),
            "target": target_info,
            "target_distance_m": self.target_distance_m,
            "gesture": gesture_info,
            "gesture_enabled": self.gesture_enabled,
            "gesture_logs": list(self.gesture_controller.logs),
            "runtime_settings": self.current_settings(),
            "sleeping": self.gesture_controller.mode == ControlMode.SLEEP,
            "scene": self.latest_scene,
            "follow_lock_state": self.lock_state,
            "follow_lock_message": self.lock_message,
            "follow_enabled": self.follow_requested,
            "can_start_follow": self.locked_target is not None,
            "locked_target": None
            if self.locked_target is None
            else {
                "selected_by": self.locked_target.selected_by,
                "locked_at": round(self.locked_target.locked_at, 3),
                "confidence": round(self.locked_target.confidence, 3),
                "bbox": [round(v, 1) for v in self.locked_target.bbox],
            },
            "sub_agents": sub_agents,
            "agent": {
                "enabled": self.agent_enabled,
                "status": sub_agents["LLM_AGENT"]["status"],
                "vision_description": self.latest_vision_description,
                "last_plan": dict(self.robot_executor.last_plan),
                "logs": list(self.robot_executor.logs),
                "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                "tokens": dict(self.robot_executor.last_tokens),
                "job_running": self.agent_job_running,
                "error": self.latest_agent_error,
                "user_request": self.user_request,
            },
            "ai_compute": {
                "latency_ms": round(self.robot_executor.last_latency_ms, 1),
                "tokens": dict(self.robot_executor.last_tokens),
                "job_running": self.agent_job_running,
                "model": self.planner_agent.model,
                "vlm_model": self.vision_agent.model,
            },
            "recording": {
                "active": self.recording,
                "video_path": self.video_output_path,
                "last_snapshot_path": self.last_snapshot_path,
                "started_at": round(self.recording_started_at, 3) if self.recording_started_at else None,
            },
            "telemetry_summary": telemetry_summary,
            "agent_summary": agent_summary,
            "event_summary": event_summary,
            "model_runtime": model_runtime,
            "details": detail_payload,
            "raw_cmd": twist_to_dict(self.latest_cmd),
            "safe_cmd": twist_to_dict(self.latest_safe_cmd),
            "raw_gimbal_cmd": twist_to_dict(self.latest_gimbal_cmd),
            "safe_gimbal_cmd": twist_to_dict(self.latest_safe_gimbal_cmd),
            "control_reason": self.latest_reason,
            "safety_reason": self.latest_safety_reason,
            "event_logs": self.event_bus.snapshot(),
            "last_image_age_sec": round(time.time() - self.last_image_time, 2) if self.last_image_time else None,
            "last_target_age_sec": round(time.time() - self.last_target_time, 2) if self.last_target_time else None,
            "last_reconnect_time": round(self.last_reconnect_time, 3) if self.last_reconnect_time else None,
        }

    def destroy_node(self) -> bool:
        """确保 Ctrl+C 或节点退出时一定回零速。"""
        self.publish_stop("destroy node")
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.gesture_detector is not None:
            self.gesture_detector.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PersonFollowerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("KeyboardInterrupt, stopping robot")
    except Exception as exc:
        if node is not None:
            node.get_logger().error("Unhandled exception: %s" % exc)
            node.publish_stop("unhandled exception")
        else:
            print("person_follower_node failed before startup: %s" % exc)
    finally:
        if node is not None:
            node.publish_stop("shutdown")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
