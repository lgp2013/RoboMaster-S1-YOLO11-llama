"""Documentation."""

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
from rclpy.action import ActionClient
from rclpy.node import Node
from robomaster_msgs.action import RecenterGimbal
from robomaster_msgs.msg import GimbalCommand
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import Bool, String

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
    """Documentation."""

    bbox: Tuple[float, float, float, float]
    center_x: float
    center_y: float
    area: float
    selected_by: str
    locked_at: float
    label: str
    confidence: float


class InternalEventBus:
    """Documentation."""

    def __init__(self, maxlen: int = 80) -> None:
        self._events: Deque[Dict[str, object]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def publish(self, source: str, message: str, level: str = "info", event_type: str = "status") -> Dict[str, object]:
        """Documentation."""
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
        """Documentation."""
        with self._lock:
            return list(list(self._events)[:limit])


class PersonFollowerNode(Node):
    """Documentation."""

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
        self.control_rate_hz = float(self.get_parameter("follow_control.control_rate_hz").value)
        self.lost_target_timeout_sec = float(self.get_parameter("follow_control.target_lost_timeout").value)
        self.follow_enabled = bool(self.get_parameter("follow_control.enabled").value)
        self.require_locked_target = bool(self.get_parameter("follow_control.require_locked_target").value)
        self.follow_person_class_id = int(self.get_parameter("follow_control.yolo_class_person").value)
        self.follow_iou_match_threshold = float(self.get_parameter("follow_control.iou_match_threshold").value)
        self.follow_center_distance_threshold_ratio = float(
            self.get_parameter("follow_control.center_distance_threshold_ratio").value
        )
        self.follow_max_area_change_ratio = float(self.get_parameter("follow_control.max_area_change_ratio").value)
        self.target_distance_m = 1.0
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
        self.gimbal_pub = self.create_publisher(GimbalCommand, self.cmd_gimbal_topic, 10)
        self.gimbal_engage_pub = self.create_publisher(Bool, "gimbal/engage", 10)
        self.robot_mode_pub = self.create_publisher(String, self.robot_mode_topic, 10)
        self.robot_command_pub = self.create_publisher(String, self.robot_command_topic, 10)
        self.led_command_pub = self.create_publisher(String, self.led_command_topic, 10)
        self.gesture_state_pub = self.create_publisher(String, "/gesture/state", 10)
        self.gesture_command_pub = self.create_publisher(String, "/gesture/command", 10)
        self.gesture_debug_pub = self.create_publisher(Image, "/gesture/debug_image", 5)
        self.recenter_gimbal_client = ActionClient(self, RecenterGimbal, "recenter_gimbal")

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
        self.last_people_time = 0.0
        self.lock_match_stable_count = 0
        self.follow_error_started_at = 0.0
        self.last_follow_cmd = Twist()
        self.last_follow_gimbal_cmd = Twist()
        self.follow_debug: Dict[str, object] = {
            "follow_state": "NONE",
            "locked_target_id": "",
            "locked_bbox": None,
            "matched_bbox": None,
            "iou": 0.0,
            "center_distance": 0.0,
            "error_x": 0.0,
            "error_x_normalized": 0.0,
            "target_height_ratio": 0.0,
            "linear_x": 0.0,
            "angular_z": 0.0,
            "gimbal_yaw_speed": 0.0,
            "gimbal_yaw_invert": bool(self.get_parameter("follow_control.gimbal_yaw_invert").value),
            "chassis_yaw_invert": bool(self.get_parameter("follow_control.chassis_yaw_invert").value),
        }

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
        self.last_command_result: Dict[str, object] = {
            "success": True,
            "command": "BOOT",
            "message": "Dashboard initialized",
            "timestamp": round(time.time(), 3),
        }

        self.dashboard = None
        if bool(self.get_parameter("dashboard_enabled").value):
            self.dashboard = DashboardServer(
                host=str(self.get_parameter("dashboard_host").value),
                port=int(self.get_parameter("dashboard_port").value),
                jpeg_quality=int(self.get_parameter("jpeg_quality").value),
                command_callback=self.handle_dashboard_command,
                settings_callback=self.handle_dashboard_settings,
                media_callback=self.media_data,
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
            "chassis_move_center_ratio": 0.20,
            "yaw_gain": 1.2,
            "distance_gain": 1.1,
            "max_linear_speed": 0.35,
            "min_linear_speed": 0.07,
            "max_angular_speed": 1.0,
            "min_angular_speed": 0.10,
            "target_distance_m": 1.0,
            "target_bbox_height_ratio": 0.60,
            "bbox_height_tolerance": 0.04,
            "enable_chassis_yaw_assist": False,
            "lost_target_timeout_sec": 3.0,
            "command_timeout_sec": 5.0,
            "image_timeout_sec": 5.0,
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
            # Gimbal auto-tracking parameters
            "gimbal_target_y_ratio": 0.38,
            "gimbal_frame_center_y_ratio": 0.42,
            "gimbal_yaw_gain": 3.2,
            "gimbal_pitch_gain": 3.0,
            "max_gimbal_yaw_speed": 3.0,
            "max_gimbal_pitch_speed": 2.4,
            "min_gimbal_speed": 0.05,
            "gimbal_deadzone_px": 8,
            "gimbal_vertical_deadzone_px": 6,
            "follow_control.enabled": True,
            "follow_control.require_locked_target": True,
            "follow_control.yolo_class_person": 0,
            "follow_control.iou_match_threshold": 0.25,
            "follow_control.center_distance_threshold_ratio": 0.18,
            "follow_control.max_area_change_ratio": 2.5,
            "follow_control.target_lost_timeout": 1.0,
            "follow_control.center_dead_zone_ratio": 0.08,
            "follow_control.too_far_height_ratio": 0.32,
            "follow_control.too_close_height_ratio": 0.58,
            "follow_control.max_linear_x": 0.10,
            "follow_control.max_angular_z": 0.30,
            "follow_control.max_gimbal_yaw_speed": 0.25,
            "follow_control.kp_chassis_yaw": 0.25,
            "follow_control.kp_gimbal_yaw": 0.20,
            "follow_control.chassis_yaw_invert": False,
            "follow_control.gimbal_yaw_invert": True,
            "follow_control.smooth_alpha": 0.4,
            "follow_control.control_rate_hz": 10.0,
            "follow_control.gimbal_first": True,
            "follow_control.chassis_rotate_delay_seconds": 0.5,
            "follow_control.debug_follow_only": False,
            "follow_control.enable_gimbal_control": True,
            "follow_control.enable_chassis_rotation": True,
            "follow_control.enable_distance_control": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _control_config(self) -> ControlConfig:
        return ControlConfig(
            center_dead_zone_ratio=float(self.get_parameter("follow_control.center_dead_zone_ratio").value),
            too_far_height_ratio=float(self.get_parameter("follow_control.too_far_height_ratio").value),
            too_close_height_ratio=float(self.get_parameter("follow_control.too_close_height_ratio").value),
            max_linear_x=float(self.get_parameter("follow_control.max_linear_x").value),
            max_angular_z=float(self.get_parameter("follow_control.max_angular_z").value),
            max_gimbal_yaw_speed=float(self.get_parameter("follow_control.max_gimbal_yaw_speed").value),
            kp_chassis_yaw=float(self.get_parameter("follow_control.kp_chassis_yaw").value),
            kp_gimbal_yaw=float(self.get_parameter("follow_control.kp_gimbal_yaw").value),
            chassis_yaw_invert=bool(self.get_parameter("follow_control.chassis_yaw_invert").value),
            gimbal_yaw_invert=bool(self.get_parameter("follow_control.gimbal_yaw_invert").value),
            smooth_alpha=float(self.get_parameter("follow_control.smooth_alpha").value),
            gimbal_first=bool(self.get_parameter("follow_control.gimbal_first").value),
            chassis_rotate_delay_seconds=float(self.get_parameter("follow_control.chassis_rotate_delay_seconds").value),
            debug_follow_only=bool(self.get_parameter("follow_control.debug_follow_only").value),
            enable_gimbal_control=bool(self.get_parameter("follow_control.enable_gimbal_control").value),
            enable_chassis_rotation=bool(self.get_parameter("follow_control.enable_chassis_rotation").value),
            enable_distance_control=bool(self.get_parameter("follow_control.enable_distance_control").value),
            gimbal_target_y_ratio=float(self.get_parameter("gimbal_target_y_ratio").value),
            gimbal_frame_center_y_ratio=float(self.get_parameter("gimbal_frame_center_y_ratio").value),
            gimbal_yaw_gain=float(self.get_parameter("gimbal_yaw_gain").value),
            gimbal_pitch_gain=float(self.get_parameter("gimbal_pitch_gain").value),
            max_gimbal_pitch_speed=float(self.get_parameter("max_gimbal_pitch_speed").value),
            min_gimbal_speed=float(self.get_parameter("min_gimbal_speed").value),
            gimbal_deadzone_px=int(self.get_parameter("gimbal_deadzone_px").value),
            gimbal_vertical_deadzone_px=int(self.get_parameter("gimbal_vertical_deadzone_px").value),
        )

    def _gesture_actions(self) -> Dict[str, str]:
        names = ["open_palm", "thumbs_up", "fist", "point_left", "point_right", "victory", "ok_sign"]
        return {name: str(self.get_parameter("gesture_actions.%s" % name).value) for name in names}

    def publish_event(self, source: str, message: str, level: str = "info", event_type: str = "status") -> None:
        """Documentation."""
        event = self.event_bus.publish(source, message, level=level, event_type=event_type)
        if level == "error":
            self.get_logger().error("[%s] %s" % (source, message))
        elif level == "warning":
            self.get_logger().warning("[%s] %s" % (source, message))
        else:
            self.get_logger().info("[%s] %s" % (source, message))
        return event

    def robot_command_callback(self, msg: String) -> None:
        """Documentation."""
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
        """Documentation."""
        self.robot_mode_pub.publish(String(data=str(self.gesture_controller.mode)))

    def publish_robot_command(self, command: str, payload: Optional[Dict[str, object]] = None) -> None:
        """Documentation."""
        data = {"command": str(command).upper(), "source": "dashboard", "timestamp": time.time()}
        if payload:
            data["payload"] = payload
        self.robot_command_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def publish_led_command(self, command: str) -> None:
        """Documentation."""
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
        """Documentation."""
        self.current_command = command
        self.current_control_source = source
        self.last_command_time = time.time()

    def apply_robot_command(
        self,
        command: str,
        source: str = "DASHBOARD",
        payload: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Documentation."""
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
            if self.require_locked_target and (self.locked_target is None or self.lock_state != "LOCKED"):
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
            raw_width = payload.get("raw_width")
            raw_height = payload.get("raw_height")
            return self.lock_target_from_point(
                float(x),
                float(y),
                source=source,
                raw_width=float(raw_width) if raw_width is not None else None,
                raw_height=float(raw_height) if raw_height is not None else None,
            )
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
        if command in ("GIMBAL_UP", "GIMBAL_DOWN", "GIMBAL_LEFT", "GIMBAL_RIGHT"):
            self.start_manual_override(command)
            self.publish_event("GESTURE_AGENT", "%s manual gimbal %s" % (source, command), event_type="command")
            return self._success(command, "Manual gimbal command armed", mode=self.gesture_controller.mode)
        if command == "GIMBAL_CENTER":
            self.clear_manual_override()
            self.publish_stop("%s gimbal center" % source)
            self.recenter_gimbal(source)
            self.publish_event("GESTURE_AGENT", "%s manual gimbal center" % source, event_type="command")
            return self._success(command, "Gimbal recenter requested", mode=self.gesture_controller.mode)

        return self._failure(command, "Unsupported command")

    def enter_sleep(self, source: str) -> None:
        """Documentation."""
        self.follow_requested = False
        self.clear_manual_override()
        self.robot_executor.update_plan({"action": "STOP", "reason": "sleep mode", "speak": ""})
        self.gesture_controller._apply_action("SLEEP", source=source.lower(), gesture="command")
        self.publish_sleep_zero("%s sleep" % source)
        self.publish_led_command("OFF")
        self.publish_event("SAFETY_AGENT", "%s -> SLEEP" % source, level="warning", event_type="command")
        self.publish_robot_mode()

    def wake_robot(self, source: str) -> Dict[str, object]:
        """Documentation."""
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
        """Documentation."""
        payload = payload or {}
        self.publish_robot_command(command, payload)
        result = self.apply_robot_command(command, source="DASHBOARD", payload=payload)
        self.last_command_result = {
            "success": bool(result.get("success", result.get("ok", False))),
            "command": str(result.get("command", command)),
            "message": str(result.get("message", result.get("error", ""))),
            "path": str(result.get("path", "")),
            "duration_seconds": result.get("duration_seconds"),
            "timestamp": round(time.time(), 3),
        }
        status = self._status_dict()
        result["status"] = status
        return result

    def handle_dashboard_settings(self, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Documentation."""
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
        return {
            "ok": True,
            "success": True,
            "settings": self.current_settings(),
            "core": self.core_data(),
            "event_logs": self.event_bus.snapshot(),
        }

    def current_settings(self) -> Dict[str, object]:
        """Documentation."""
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
        """Documentation."""
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
        """Documentation."""
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
            gimbal.angular.y = -gimbal_speed
        elif command == "GIMBAL_DOWN":
            gimbal.angular.y = gimbal_speed
        elif command == "GIMBAL_LEFT":
            gimbal.angular.z = -gimbal_speed
        elif command == "GIMBAL_RIGHT":
            gimbal.angular.z = gimbal_speed

        self.manual_override_cmd = chassis
        self.manual_override_gimbal_cmd = gimbal
        self.manual_override_action = command
        self.manual_override_started_at = time.time()
        self.manual_override_until = self.manual_override_started_at + float(self.robot_executor.action_duration_sec)

    def clear_manual_override(self) -> None:
        """Documentation."""
        self.manual_override_cmd = None
        self.manual_override_gimbal_cmd = None
        self.manual_override_started_at = 0.0
        self.manual_override_until = 0.0
        self.manual_override_action = "NONE"

    def battery_callback(self, msg: BatteryState) -> None:
        """Documentation."""
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
            tracked_target = self.match_locked_target_frame(people)
            highlighted = tracked_target if tracked_target is not None else self.detector.select_largest_person(people)
            if self.locked_target is None:
                if people:
                    self.lock_state = "CANDIDATE"
                    self.lock_message = "Detected candidate person; waiting for lock"
                else:
                    self.lock_state = "SCANNING"
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
                if people:
                    self.last_people_time = now
                if tracked_target is not None:
                    self.last_target_time = now

            self._write_recording_frame(annotated)

            if not sleep_mode:
                self._maybe_start_agent_job(frame, people, gestures, tracked_target)

            if self.publish_debug_image:
                self.gesture_debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

            if self.dashboard is not None:
                self.dashboard.update(frame, annotated, self._status_dict())
        except Exception as exc:
            self.publish_event("SYSTEM", "image processing failed: %s" % exc, level="error", event_type="vision")
            self.publish_stop("image processing exception")

    def _draw_overlay(
        self,
        frame,
        gesture_state: Dict[str, object],
        tracked_target: Optional[PersonDetection],
    ) -> None:
        """Documentation."""
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
        """Documentation."""
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
        """Documentation."""
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
        now = now_sec()
        image_age = now - last_image_time if last_image_time else 999.0

        if self.manual_override_cmd is not None:
            if time.time() <= self.manual_override_until:
                command_age = max(0.0, now - self.manual_override_started_at)
                safe_cmd, safety_reason = self.safety_guard.filter_cmd(self.manual_override_cmd, True, 0.0, command_age)
                gimbal_cmd = self._clamp_gimbal_cmd(self.manual_override_gimbal_cmd or Twist())
                self.latest_action = "MANUAL_%s" % self.manual_override_action
                self._publish_cmd(safe_cmd, self.manual_override_cmd, gimbal_cmd, self.manual_override_gimbal_cmd or Twist(), "manual override", safety_reason)
                return
            self.clear_manual_override()

        if frame is None:
            self.publish_stop("no image")
            return
        if image_age > float(self.get_parameter("image_timeout_sec").value):
            self.publish_stop("image timeout")
            return

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

        if image_age > 1.0:
            self._set_follow_lost("image timeout, stop following")
            self.publish_stop("image timeout")
            return

        has_fresh_target = target is not None and (now - last_target_time) <= self.lost_target_timeout_sec
        if not has_fresh_target:
            self._set_follow_lost("target lost, stop following")
            self.publish_stop("target lost")
            if time.time() - self.last_lock_loss_time > 1.0:
                self.last_lock_loss_time = time.time()
                self.publish_event("FOLLOW_AGENT", "target lost, stop following", level="warning", event_type="lock")
            return

        if self.lock_match_stable_count < 2:
            self.lock_state = "LOCKED"
            self.lock_message = "Locked target stabilizing"
            self.follow_debug.update({"follow_state": self.lock_state})
            self.publish_stop("waiting stable target")
            return

        height, width = frame.shape[:2]
        metrics = self.controller.compute_metrics(target, width, height)
        self.latest_bbox_height_ratio = metrics["target_height_ratio"]
        self.follow_debug.update(
            {
                "follow_state": "FOLLOW_ACTIVE",
                "locked_target_id": self.locked_target_id,
                "locked_bbox": [round(v, 1) for v in self.locked_target.bbox] if self.locked_target else None,
                "error_x": round(metrics["error_x"], 2),
                "error_x_normalized": round(metrics["error_x_normalized"], 3),
                "target_height_ratio": round(metrics["target_height_ratio"], 3),
            }
        )

        gimbal_raw, _ = self.controller.compute_gimbal_cmd(target, width, height)
        gimbal_yaw_speed = float(gimbal_raw.angular.z)
        gimbal_near_limit = abs(gimbal_yaw_speed) >= (self.controller.config.max_gimbal_yaw_speed * 0.92)
        if abs(metrics["error_x"]) >= metrics["dead_zone_x"]:
            if self.follow_error_started_at <= 0.0:
                self.follow_error_started_at = now
        else:
            self.follow_error_started_at = 0.0

        allow_chassis_rotation = bool(self.controller.config.enable_chassis_rotation)
        if allow_chassis_rotation and self.controller.config.gimbal_first:
            elapsed = max(0.0, now - self.follow_error_started_at) if self.follow_error_started_at > 0.0 else 0.0
            allow_chassis_rotation = elapsed >= self.controller.config.chassis_rotate_delay_seconds or gimbal_near_limit

        cmd_raw, cmd_metrics = self.controller.compute_cmd(target, width, height, allow_chassis_rotation)
        safe_cmd, safety_reason = self.safety_guard.filter_cmd(cmd_raw, True, image_age)
        safe_cmd = self._smooth_follow_cmd(safe_cmd)
        safe_gimbal_cmd = self._smooth_follow_gimbal_cmd(gimbal_raw)
        safe_gimbal_cmd = self._clamp_gimbal_cmd(safe_gimbal_cmd)
        self.follow_debug.update(
            {
                "matched_bbox": [round(v, 1) for v in self.locked_target.bbox] if self.locked_target else None,
                "linear_x": round(float(safe_cmd.linear.x), 3),
                "angular_z": round(float(safe_cmd.angular.z), 3),
                "gimbal_yaw_speed": round(float(safe_gimbal_cmd.angular.z), 3),
                "gimbal_yaw_invert": bool(self.controller.config.gimbal_yaw_invert),
                "chassis_yaw_invert": bool(self.controller.config.chassis_yaw_invert),
            }
        )
        self.lock_state = "FOLLOW_ACTIVE"
        self.lock_message = "Locked target tracked"
        reason = "follow: height=%s yaw=%s gimbal=%s" % (
            round(float(cmd_metrics["target_height_ratio"]), 3),
            round(float(cmd_raw.angular.z), 3),
            round(float(gimbal_raw.angular.z), 3),
        )
        self._publish_cmd(safe_cmd, cmd_raw, safe_gimbal_cmd, gimbal_raw, reason, safety_reason)

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
        self.publish_gimbal_speed(safe_gimbal_cmd)
        self.latest_cmd = raw_cmd
        self.latest_safe_cmd = safe_cmd
        self.latest_gimbal_cmd = raw_gimbal_cmd
        self.latest_safe_gimbal_cmd = safe_gimbal_cmd
        self.latest_reason = reason
        self.latest_safety_reason = safety_reason

    def _smooth_follow_cmd(self, cmd: Twist) -> Twist:
        """Documentation."""
        alpha = max(0.0, min(1.0, float(self.controller.config.smooth_alpha)))
        if self.last_follow_cmd is None:
            self.last_follow_cmd = Twist()
        smooth = Twist()
        smooth.linear.x = alpha * float(cmd.linear.x) + (1.0 - alpha) * float(self.last_follow_cmd.linear.x)
        smooth.linear.y = 0.0
        smooth.linear.z = 0.0
        smooth.angular.x = 0.0
        smooth.angular.y = 0.0
        smooth.angular.z = alpha * float(cmd.angular.z) + (1.0 - alpha) * float(self.last_follow_cmd.angular.z)
        self.last_follow_cmd = smooth
        return smooth

    def _smooth_follow_gimbal_cmd(self, cmd: Twist) -> Twist:
        """Documentation."""
        alpha = max(0.0, min(1.0, float(self.controller.config.smooth_alpha)))
        if self.last_follow_gimbal_cmd is None:
            self.last_follow_gimbal_cmd = Twist()
        smooth = Twist()
        smooth.linear.x = 0.0
        smooth.linear.y = 0.0
        smooth.linear.z = 0.0
        smooth.angular.x = 0.0
        smooth.angular.y = alpha * float(cmd.angular.y) + (1.0 - alpha) * float(self.last_follow_gimbal_cmd.angular.y)
        smooth.angular.z = alpha * float(cmd.angular.z) + (1.0 - alpha) * float(self.last_follow_gimbal_cmd.angular.z)
        self.last_follow_gimbal_cmd = smooth
        return smooth

    def publish_gimbal_speed(self, cmd: Twist) -> None:
        msg = GimbalCommand()
        msg.pitch_speed = float(cmd.angular.y)
        msg.yaw_speed = float(cmd.angular.z)
        self.gimbal_engage_pub.publish(Bool(data=True))
        self.gimbal_pub.publish(msg)

    def _clamp_gimbal_cmd(self, cmd: Twist) -> Twist:
        """Documentation."""
        safe = Twist()
        safe.angular.y = clamp(float(cmd.angular.y), -self.max_angular_speed, self.max_angular_speed)
        safe.angular.z = clamp(float(cmd.angular.z), -self.max_angular_speed, self.max_angular_speed)
        return safe

    def publish_stop(self, reason: str = "stop") -> None:
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.publish_gimbal_speed(stop)
        self.latest_cmd = stop
        self.latest_safe_cmd = stop
        self.latest_gimbal_cmd = stop
        self.latest_safe_gimbal_cmd = stop
        self.latest_reason = reason
        self.latest_safety_reason = "stop"
        self.last_follow_cmd = Twist()
        self.last_follow_gimbal_cmd = Twist()

    def recenter_gimbal(self, source: str) -> None:
        if not self.recenter_gimbal_client.wait_for_server(timeout_sec=0.2):
            self.publish_event("GESTURE_AGENT", "%s recenter_gimbal action unavailable" % source, level="warning", event_type="command")
            return
        goal = RecenterGimbal.Goal()
        goal.yaw_speed = float(self.max_angular_speed)
        goal.pitch_speed = float(self.max_angular_speed)
        self.gimbal_engage_pub.publish(Bool(data=True))
        self.recenter_gimbal_client.send_goal_async(goal)

    def publish_sleep_zero(self, reason: str = "sleep") -> None:
        """Documentation."""
        self.publish_stop(reason)
        self.latest_safety_reason = "sleep zero"

    def lock_best_person(self, source: str) -> Dict[str, object]:
        """Documentation."""
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

    def lock_target_from_point(
        self,
        x: float,
        y: float,
        source: str = "DASHBOARD",
        raw_width: Optional[float] = None,
        raw_height: Optional[float] = None,
    ) -> Dict[str, object]:
        """Documentation."""
        with self.lock:
            people = list(self.latest_people)
        if not people:
            return self._failure("LOCK_TARGET", "No detected person available")

        width = max(1.0, float(self.latest_frame.shape[1])) if self.latest_frame is not None else 1.0
        height = max(1.0, float(self.latest_frame.shape[0])) if self.latest_frame is not None else 1.0
        source_width = max(1.0, float(raw_width if raw_width is not None else width))
        source_height = max(1.0, float(raw_height if raw_height is not None else height))
        px = clamp(float(x), 0.0, source_width - 1.0) * (width / source_width)
        py = clamp(float(y), 0.0, source_height - 1.0) * (height / source_height)

        containing = [person for person in people if person.x1 <= px <= person.x2 and person.y1 <= py <= person.y2]
        if not containing:
            self.publish_event("FOLLOW_AGENT", "%s click missed all person boxes" % source, level="warning", event_type="lock")
            return self._failure("LOCK_TARGET", "Click point is outside every person bbox")
        target = min(containing, key=lambda item: ((item.center_x - px) ** 2 + (item.center_y - py) ** 2))

        self._store_locked_target(target, selected_by=source)
        self.publish_event("FOLLOW_AGENT", "%s clicked and locked target" % source, event_type="lock")
        return self._success("LOCK_TARGET", "Target locked from video click", lock_state=self.lock_state)

    def unlock_target(self, reason: str) -> None:
        """Documentation."""
        self.locked_target = None
        self.locked_target_id = ""
        self.follow_requested = False
        self.lock_state = "NONE"
        self.lock_message = reason
        self.publish_event("FOLLOW_AGENT", reason, level="warning", event_type="lock")
        self.last_follow_cmd = Twist()
        self.last_follow_gimbal_cmd = Twist()

    def _store_locked_target(self, target: PersonDetection, selected_by: str) -> None:
        """Documentation."""
        self.locked_target = LockedTarget(
            bbox=(target.x1, target.y1, target.x2, target.y2),
            center_x=float(target.center_x),
            center_y=float(target.center_y),
            area=float(target.area),
            selected_by=selected_by,
            locked_at=time.time(),
            label="person",
            confidence=float(target.confidence),
        )
        self.locked_target_id = "person-%s" % int(self.locked_target.locked_at * 1000)
        self.lock_state = "LOCKED"
        self.lock_message = "Locked target ready for follow"
        self.lock_match_stable_count = 0
        self.follow_error_started_at = 0.0
        self.last_follow_cmd = Twist()
        self.last_follow_gimbal_cmd = Twist()
        self.follow_debug.update(
            {
                "follow_state": self.lock_state,
                "locked_target_id": self.locked_target_id,
                "locked_bbox": [round(v, 1) for v in self.locked_target.bbox],
                "matched_bbox": None,
                "iou": 0.0,
                "center_distance": 0.0,
                "error_x": 0.0,
                "error_x_normalized": 0.0,
                "target_height_ratio": 0.0,
                "linear_x": 0.0,
                "angular_z": 0.0,
                "error_y": 0.0,
                "error_y_normalized": 0.0,
                "gimbal_yaw_speed": 0.0,
            }
        )

    def _resolve_locked_target(self, people: List[PersonDetection]) -> Optional[PersonDetection]:
        """Documentation."""
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

    def match_locked_target_frame(self, people: List[PersonDetection]) -> Optional[PersonDetection]:
        """Documentation."""
        if self.locked_target is None:
            self.lock_state = "NONE"
            self.lock_message = "No target locked"
            return None
        if not people:
            self.follow_debug.update({"iou": 0.0, "center_distance": 0.0, "matched_bbox": None})
            self._set_follow_lost("Locked target missing in current frame")
            return None

        best_iou_target = None
        best_iou = 0.0
        for person in people:
            score = self._iou(self.locked_target.bbox, (person.x1, person.y1, person.x2, person.y2))
            if score > best_iou:
                best_iou = score
                best_iou_target = person

        def area_ratio_ok(person: PersonDetection) -> bool:
            area_ratio = max(person.area, self.locked_target.area) / max(1.0, min(person.area, self.locked_target.area))
            return area_ratio <= self.follow_max_area_change_ratio

        match = None
        center_distance = 0.0
        if best_iou_target is not None and best_iou >= self.follow_iou_match_threshold and area_ratio_ok(best_iou_target):
            match = best_iou_target
            center_distance = ((match.center_x - self.locked_target.center_x) ** 2 + (match.center_y - self.locked_target.center_y) ** 2) ** 0.5
        else:
            frame_width = max(1.0, float(self.latest_frame.shape[1])) if self.latest_frame is not None else 1.0
            max_center_distance = frame_width * self.follow_center_distance_threshold_ratio
            center_candidates = [
                person
                for person in people
                if area_ratio_ok(person)
                and (((person.center_x - self.locked_target.center_x) ** 2 + (person.center_y - self.locked_target.center_y) ** 2) ** 0.5) <= max_center_distance
            ]
            if center_candidates:
                match = min(
                    center_candidates,
                    key=lambda item: ((item.center_x - self.locked_target.center_x) ** 2 + (item.center_y - self.locked_target.center_y) ** 2),
                )
                center_distance = ((match.center_x - self.locked_target.center_x) ** 2 + (match.center_y - self.locked_target.center_y) ** 2) ** 0.5

        if match is None:
            self.follow_debug.update({"iou": round(best_iou, 3), "center_distance": round(center_distance, 2), "matched_bbox": None})
            self._set_follow_lost("target lost, stop following")
            return None

        self.locked_target.bbox = (match.x1, match.y1, match.x2, match.y2)
        self.locked_target.center_x = float(match.center_x)
        self.locked_target.center_y = float(match.center_y)
        self.locked_target.area = float(match.area)
        self.locked_target.confidence = float(match.confidence)
        self.lock_match_stable_count += 1
        self.lock_state = "LOCKED"
        self.lock_message = "Locked target tracked"
        self.follow_debug.update(
            {
                "follow_state": self.lock_state,
                "locked_target_id": self.locked_target_id,
                "locked_bbox": [round(v, 1) for v in self.locked_target.bbox],
                "matched_bbox": [round(match.x1, 1), round(match.y1, 1), round(match.x2, 1), round(match.y2, 1)],
                "iou": round(best_iou, 3),
                "center_distance": round(center_distance, 2),
            }
        )
        return match

    def _set_follow_lost(self, reason: str) -> None:
        """Documentation."""
        self.follow_requested = False
        self.lock_match_stable_count = 0
        self.last_follow_cmd = Twist()
        self.last_follow_gimbal_cmd = Twist()
        self.follow_error_started_at = 0.0
        self.lock_state = "LOST"
        self.lock_message = reason
        self.follow_debug.update({"follow_state": self.lock_state})

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """Documentation."""
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
        """Documentation."""
        if not self.recording or self.video_writer is None:
            return
        self.video_writer.write(frame)

    def take_snapshot(self, source: str) -> Dict[str, object]:
        """Documentation."""
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
        """Documentation."""
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
        """Documentation."""
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

    def _media_items(self, directory: Path, kind: str, limit: int = 16) -> List[Dict[str, object]]:
        if not directory.exists():
            return []
        items: List[Dict[str, object]] = []
        for path in sorted(directory.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            stat = path.stat()
            relative_path = path.relative_to(self.record_root).as_posix()
            items.append(
                {
                    "kind": kind,
                    "name": path.name,
                    "relative_path": relative_path,
                    "url": "/api/media/file?path=%s" % relative_path,
                    "size_bytes": int(stat.st_size),
                    "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                }
            )
            if len(items) >= limit:
                break
        return items

    def media_data(self) -> Dict[str, object]:
        snapshots = self._media_items(self.snapshot_dir, "snapshot")
        videos = self._media_items(self.video_dir, "video")
        latest_snapshot = snapshots[0] if snapshots else None
        latest_video = videos[0] if videos else None
        return {
            "record_root": str(self.record_root.resolve()),
            "snapshot_dir": str(self.snapshot_dir.resolve()),
            "video_dir": str(self.video_dir.resolve()),
            "snapshot_count": len(snapshots),
            "video_count": len(videos),
            "latest_snapshot": latest_snapshot,
            "latest_video": latest_video,
            "last_snapshot_path": self.last_snapshot_path,
            "active_recording": self.recording,
            "video_output_path": self.video_output_path,
            "recording_started_at": round(self.recording_started_at, 3) if self.recording_started_at else None,
            "snapshots": snapshots,
            "videos": videos,
        }

    def _agent_status_map(self) -> Dict[str, Dict[str, object]]:
        """Documentation."""
        now_text = time.strftime("%H:%M:%S")
        latest_plan = dict(self.robot_executor.last_plan)
        latest_plan_action = str(latest_plan.get("action", "STOP"))
        gesture_state = dict(self.latest_gesture_state)
        gesture_output = str(gesture_state.get("stable_gesture", "none"))
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
                        "/api/detail/media",
                    ],
                },
                "current_action": self.current_command,
                "last_error": "",
            },
        }

    def _status_dict(self) -> Dict[str, object]:
        """Documentation."""
        target = self.latest_target
        frame_width = int(self.latest_frame.shape[1]) if self.latest_frame is not None else 0
        frame_height = int(self.latest_frame.shape[0]) if self.latest_frame is not None else 0
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
        summary_scene = self.latest_vision_description or self.latest_scene.get("description", "鏆傛棤鍦烘櫙鎶ュ憡")
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
        media_payload = self.media_data()
        media_summary = {
            "snapshot_count": media_payload["snapshot_count"],
            "video_count": media_payload["video_count"],
            "latest_snapshot_name": media_payload["latest_snapshot"]["name"] if media_payload["latest_snapshot"] else "none",
            "latest_video_name": media_payload["latest_video"]["name"] if media_payload["latest_video"] else "none",
            "recording_active": self.recording,
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
                    "follow_state": self.lock_state,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
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
                    "follow_state": self.lock_state,
                    "follow_debug": dict(self.follow_debug),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "target_lost_seconds": round(time.time() - self.last_target_time, 2) if self.last_target_time else None,
                    "follow_enabled": self.follow_requested,
                    "follow_active": self.gesture_controller.mode == ControlMode.FOLLOW and self.follow_requested,
                    "lock_state": self.lock_state,
                },
            },
            "media": media_payload,
        }
        return {
            "follow_state": self.lock_state,
            "frame_width": frame_width,
            "frame_height": frame_height,
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
            "last_command_result": dict(self.last_command_result),
            "telemetry_summary": telemetry_summary,
            "agent_summary": agent_summary,
            "event_summary": event_summary,
            "media_summary": media_summary,
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
        """Documentation."""
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
