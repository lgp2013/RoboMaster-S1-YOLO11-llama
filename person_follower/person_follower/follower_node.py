"""ROS2 Foxy node for RoboMaster S1 YOLO person following and gesture control."""

from collections import deque
import threading
import time
from typing import Deque, Dict, List, Optional

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
from .utils import clamp, json_string_msg, now_sec, twist_to_dict
from .vision_agent import VisionAgent
from .web_dashboard import DASHBOARD_VERSION, DashboardServer
from .yolo_detector import PersonDetection, YoloPersonDetector


class PersonFollowerNode(Node):
    """第二阶段节点：人体跟随 + 手势模式控制。"""

    def __init__(self) -> None:
        super().__init__("person_follower")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.lost_target_timeout_sec = float(self.get_parameter("lost_target_timeout_sec").value)
        self.target_distance_m = float(self.get_parameter("target_distance_m").value)
        self.gesture_enabled = bool(self.get_parameter("gesture.enabled").value)
        self.publish_debug_image = bool(self.get_parameter("gesture.publish_debug_image").value)
        self.agent_enabled = bool(self.get_parameter("agent.enabled").value)

        self.detector = YoloPersonDetector(
            model_path=str(self.get_parameter("yolo_model").value),
            confidence=float(self.get_parameter("confidence_threshold").value),
            imgsz=int(self.get_parameter("imgsz").value),
            person_class_id=int(self.get_parameter("person_class_id").value),
        )
        self.controller = PersonFollowerController(self._control_config())
        self.safety_guard = SafetyGuard(
            max_linear_speed=float(self.get_parameter("max_linear_speed").value),
            max_angular_speed=float(self.get_parameter("max_angular_speed").value),
            command_timeout_sec=float(self.get_parameter("command_timeout_sec").value),
            image_timeout_sec=float(self.get_parameter("image_timeout_sec").value),
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
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
        self.latest_cmd = Twist()
        self.latest_safe_cmd = Twist()
        self.latest_action = "NONE"
        self.latest_scene: Dict[str, object] = {}
        self.latest_vision_description = ""
        self.latest_agent_error = ""
        self.user_request = ""
        self.agent_job_running = False
        self.last_agent_plan_time = 0.0
        self.agent_lock = threading.Lock()
        self.event_logs: Deque[Dict[str, object]] = deque(maxlen=20)
        self.battery_percent = None
        self.manual_override_cmd: Optional[Twist] = None
        self.manual_override_started_at = 0.0
        self.manual_override_until = 0.0
        self.manual_override_action = "NONE"
        self.last_image_time = 0.0
        self.last_target_time = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

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

        self.get_logger().info("Person follower stage2 node started")
        self.get_logger().info("camera_topic=%s cmd_vel_topic=%s" % (self.camera_topic, self.cmd_vel_topic))
        self.battery_sub = self.create_subscription(BatteryState, "/battery", self.battery_callback, 10)
        # 所有状态字段初始化完成后再订阅图像，避免回调抢先触发时访问未创建的属性。
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.log_event("system boot: tactical dashboard online")

    def _declare_parameters(self) -> None:
        defaults = {
            "camera_topic": "/camera/image_color",
            "cmd_vel_topic": "/cmd_vel",
            "robot_ip": "10.10.10.152",
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

    def handle_dashboard_command(self, command: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """处理 Dashboard 按钮和 Agent 查询。"""
        payload = payload or {}
        command = command.upper()
        if command == "AGENT_MODE":
            self.gesture_controller.previous_mode = self.gesture_controller.mode
            self.gesture_controller.mode = ControlMode.AGENT_MODE
            self.log_event("dashboard: enter AGENT_MODE")
            return {"ok": True, "mode": self.gesture_controller.mode, "event_logs": list(self.event_logs)}
        if command == "AGENT_QUERY":
            self.user_request = str(payload.get("text", "")).strip()
            self.gesture_controller.previous_mode = self.gesture_controller.mode
            self.gesture_controller.mode = ControlMode.AGENT_MODE
            self.last_agent_plan_time = 0.0
            self.log_event("dashboard query: %s" % (self.user_request or "(empty)"))
            return {"ok": True, "mode": self.gesture_controller.mode, "query": self.user_request, "event_logs": list(self.event_logs)}
        if command == "EMERGENCY_STOP":
            self.clear_manual_override()
            self.gesture_controller.force_stop(source="web", gesture="button")
            self.publish_stop("dashboard %s" % command)
            self.log_event("dashboard: %s" % command)
            return {"ok": True, "mode": self.gesture_controller.mode, "action": command, "event_logs": list(self.event_logs)}
        if command == "STOP":
            self.clear_manual_override()
            self.gesture_controller.handle_web_command("PAUSE")
            self.publish_stop("dashboard STOP")
            self.log_event("dashboard: STOP")
            return {"ok": True, "mode": self.gesture_controller.mode, "action": command, "event_logs": list(self.event_logs)}
        if command in ("PAUSE_FOLLOW", "PAUSE"):
            result = self.gesture_controller.handle_web_command("PAUSE")
            self.log_event("dashboard: PAUSE_FOLLOW")
            result["event_logs"] = list(self.event_logs)
            return result
        if command == "START_FOLLOW":
            result = self.gesture_controller.handle_web_command("START_FOLLOW")
            self.log_event("dashboard: START_FOLLOW")
            result["event_logs"] = list(self.event_logs)
            return result
        if command in ("FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"):
            self.start_manual_override(command)
            self.log_event("dashboard manual motion: %s" % command)
            return {"ok": True, "mode": self.gesture_controller.mode, "action": command, "event_logs": list(self.event_logs)}
        if command == "CLEAR_LOGS":
            self.gesture_controller.handle_web_command("CLEAR_LOGS")
            self.robot_executor.logs.clear()
            self.event_logs.clear()
            return {"ok": True, "message": "logs cleared", "event_logs": []}
        return self.gesture_controller.handle_web_command(command)

    def handle_dashboard_settings(self, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """读取或更新 Dashboard 运行时设置。不会直接写磁盘配置文件。"""
        if payload is None:
            return {"ok": True, "settings": self.current_settings(), "core": self.core_data()}

        settings = payload.get("settings", payload) if isinstance(payload, dict) else {}
        if not isinstance(settings, dict):
            return {"ok": False, "message": "invalid settings payload"}

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
            return {"ok": False, "message": "invalid speed setting: %s" % exc}

        self.log_event("settings updated from dashboard")
        return {"ok": True, "settings": self.current_settings(), "core": self.core_data(), "event_logs": list(self.event_logs)}

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
        yolo_status = "online" if self.detector is not None else "offline"
        hand_status = "online" if self.gesture_enabled and self.gesture_detector is not None else "standby"
        vlm_status = "online" if self.vision_agent.enabled and not self.latest_agent_error else "standby"
        llm_status = "online" if self.agent_enabled and not self.latest_agent_error else "standby"
        return {
            "dashboard_version": DASHBOARD_VERSION,
            "camera_topic": self.camera_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "robot_ip": self.robot_ip,
            "battery": self.battery_percent,
            "mode": self.gesture_controller.mode,
            "fps": round(self.fps, 2),
            "yolo_status": yolo_status,
            "hand_status": hand_status,
            "vlm_status": vlm_status,
            "llm_status": llm_status,
            "agent_status": "thinking" if self.agent_job_running else ("online" if self.agent_enabled else "standby"),
            "nav_status": "standby",
            "camera_status": "online" if camera_online else "offline",
            "last_error": self.latest_agent_error,
        }

    def start_manual_override(self, command: str) -> None:
        """Web 手动接管：固定低速短动作，优先级高于 Agent/FOLLOW。"""
        cmd = Twist()
        linear = float(self.get_parameter("agent.forward_speed").value)
        angular = float(self.get_parameter("agent.turn_speed").value)
        if command == "FORWARD":
            cmd.linear.x = abs(linear)
        elif command == "BACKWARD":
            cmd.linear.x = -abs(linear)
        elif command == "TURN_LEFT":
            cmd.angular.z = abs(angular)
        elif command == "TURN_RIGHT":
            cmd.angular.z = -abs(angular)
        self.manual_override_cmd = cmd
        self.manual_override_action = command
        self.manual_override_started_at = time.time()
        self.manual_override_until = self.manual_override_started_at + float(self.get_parameter("agent.action_duration_sec").value)

    def clear_manual_override(self) -> None:
        """停止并清除手动接管动作。"""
        self.manual_override_cmd = None
        self.manual_override_started_at = 0.0
        self.manual_override_until = 0.0
        self.manual_override_action = "NONE"

    def battery_callback(self, msg: BatteryState) -> None:
        """读取 /battery。robomaster_ros 通常发布 sensor_msgs/BatteryState。"""
        try:
            if msg.percentage >= 0.0:
                self.battery_percent = int(round(float(msg.percentage) * 100.0))
        except Exception:
            self.battery_percent = None

    def log_event(self, message: str) -> None:
        """记录 Dashboard 最近事件，不参与 ROS 控制。"""
        self.event_logs.appendleft({"time": time.strftime("%H:%M:%S"), "message": message})

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error("cv_bridge convert failed: %s" % exc)
            self.publish_stop("cv_bridge failed")
            return

        try:
            people = self.detector.detect(frame)
            target = self.detector.select_largest_person(people)
            annotated = frame.copy()
            self.detector.draw_detections(annotated, people, target)

            gestures: List[GestureResult] = []
            best = None
            if self.gesture_detector is not None:
                gestures = self.gesture_detector.detect(frame)
                self.gesture_detector.draw(annotated, gestures)
                best = best_gesture(gestures)

            triggered, gesture_state = self.gesture_debouncer.update(best)
            if best is not None and best.gesture == "open_palm":
                if self.gesture_controller.mode != ControlMode.EMERGENCY_STOP:
                    self.gesture_controller.force_stop(source="gesture_immediate", gesture="open_palm")
                    self.log_event("gesture: open_palm immediate stop")
                self.publish_stop("open_palm immediate stop")
            if triggered is not None:
                self.gesture_controller.handle_stable_gesture(triggered)
                self.log_event("stable gesture: %s" % triggered)
            self.gesture_controller.publish_state(gesture_state, best)
            self._draw_overlay(annotated, gesture_state)
            self._update_fps()

            now = now_sec()
            with self.lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                self.latest_people = people
                self.latest_target = target
                self.latest_gestures = gestures
                self.latest_best_gesture = best
                self.latest_gesture_state = gesture_state
                self.last_image_time = now
                if target is not None:
                    self.last_target_time = now

            self._maybe_start_agent_job(frame, people, gestures, target)

            if self.publish_debug_image:
                self.gesture_debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

            if self.dashboard is not None:
                self.dashboard.update(annotated, self._status_dict())

        except Exception as exc:
            self.get_logger().error("image processing failed, stopping robot: %s" % exc)
            self.log_event("image processing failed: %s" % exc)
            self.publish_stop("image processing exception")

    def _draw_overlay(self, frame, gesture_state: Dict[str, object]) -> None:
        lines = [
            "mode: %s" % self.gesture_controller.mode,
            "candidate: %s x%s" % (gesture_state.get("candidate", "none"), gesture_state.get("candidate_count", 0)),
            "stable: %s" % gesture_state.get("stable_gesture", "none"),
            "action: %s" % self.latest_action,
        ]
        y = 24
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

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
        """按固定间隔启动后台 Agent 推理，避免阻塞图像回调。"""
        if not self.agent_enabled:
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

        frame_copy = frame.copy()
        people_copy = list(people)
        gestures_copy = list(gestures)
        target_copy = target
        thread = threading.Thread(
            target=self._run_agent_job,
            args=(frame_copy, people_copy, gestures_copy, target_copy),
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
        """后台调用 VLM 和 LLM，生成动作计划。"""
        try:
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
            robot_state = self._status_dict()
            planner = self.planner_agent.plan(
                scene=scene,
                robot_state=robot_state,
                mode=self.gesture_controller.mode,
                user_request=self.user_request,
            )
            self.robot_executor.update_plan(planner.plan, planner.latency_ms, planner.tokens)
            self.log_event("agent plan: %s - %s" % (planner.plan.get("action"), planner.plan.get("reason")))
            with self.lock:
                self.latest_scene = scene
                self.latest_vision_description = vision.description
                self.latest_agent_error = vision.error or planner.error
            if planner.plan.get("action") == "STOP":
                self.publish_stop("agent stop: %s" % planner.plan.get("reason", ""))
        except Exception as exc:
            self.latest_agent_error = str(exc)
            self.robot_executor.update_plan({"action": "STOP", "reason": "agent exception: %s" % exc, "speak": ""})
            self.log_event("agent exception: %s" % exc)
            self.publish_stop("agent exception")
        finally:
            with self.agent_lock:
                self.agent_job_running = False

    def control_loop(self) -> None:
        with self.lock:
            frame = self.latest_frame
            target = self.latest_target
            last_image_time = self.last_image_time
            last_target_time = self.last_target_time

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
                self.latest_action = "MANUAL_%s" % self.manual_override_action
                self._publish_cmd(
                    safe_cmd,
                    self.manual_override_cmd,
                    "manual override %s" % self.manual_override_action,
                    safety_reason,
                )
                return
            self.clear_manual_override()

        override_cmd, action_name = self.gesture_controller.get_override_cmd()
        mode = self.gesture_controller.mode
        self.latest_action = action_name

        if mode in (ControlMode.IDLE, ControlMode.EMERGENCY_STOP, ControlMode.PATROL_READY):
            self.publish_stop("mode %s" % mode)
            return

        if mode == ControlMode.GESTURE_CONTROL and override_cmd is not None:
            command_age = max(0.0, now - self.gesture_controller.action_started_at)
            safe_cmd, safety_reason = self.safety_guard.filter_cmd(override_cmd, True, image_age, command_age)
            self._publish_cmd(safe_cmd, override_cmd, "gesture action %s" % action_name, safety_reason)
            return

        if mode == ControlMode.AGENT_MODE:
            next_mode, agent_cmd, agent_reason = self.robot_executor.resolve(mode)
            if next_mode == ControlMode.FOLLOW:
                self.gesture_controller.mode = ControlMode.FOLLOW
                mode = ControlMode.FOLLOW
            elif next_mode in (ControlMode.IDLE, ControlMode.PATROL_READY):
                self.gesture_controller.mode = next_mode
                self.publish_stop("agent mode %s: %s" % (next_mode, agent_reason))
                return
            elif agent_cmd is not None:
                command_age = max(0.0, now - self.robot_executor.active_started_at)
                safe_cmd, safety_reason = self.safety_guard.filter_cmd(agent_cmd, True, image_age, command_age)
                self._publish_cmd(safe_cmd, agent_cmd, "agent action: %s" % agent_reason, safety_reason)
                return
            else:
                self.publish_stop("agent waiting")
                return

        if mode != ControlMode.FOLLOW:
            self.publish_stop("mode %s" % mode)
            return

        has_fresh_target = target is not None and (now - last_target_time) <= self.lost_target_timeout_sec
        if not has_fresh_target:
            self.publish_stop("target lost")
            return

        height, width = frame.shape[:2]
        cmd, reason, bbox_height_ratio = self.controller.compute_cmd(target, width, height)
        safe_cmd, safety_reason = self.safety_guard.filter_cmd(cmd, True, image_age)
        self.latest_bbox_height_ratio = bbox_height_ratio
        self._publish_cmd(safe_cmd, cmd, reason, safety_reason)

    def _publish_cmd(self, safe_cmd: Twist, raw_cmd: Twist, reason: str, safety_reason: str) -> None:
        self.cmd_pub.publish(safe_cmd)
        self.latest_cmd = raw_cmd
        self.latest_safe_cmd = safe_cmd
        self.latest_reason = reason
        self.latest_safety_reason = safety_reason

    def publish_stop(self, reason: str = "stop") -> None:
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.latest_cmd = stop
        self.latest_safe_cmd = stop
        self.latest_reason = reason
        self.latest_safety_reason = "stop"

    def _status_dict(self) -> Dict[str, object]:
        target = self.latest_target
        target_info = None
        if target is not None:
            target_info = {
                "bbox": [round(target.x1, 1), round(target.y1, 1), round(target.x2, 1), round(target.y2, 1)],
                "confidence": round(target.confidence, 3),
                "area": round(target.area, 1),
                "center_x": round(target.center_x, 1),
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
        yolo_status = "online" if self.detector is not None else "offline"
        hand_status = "online" if self.gesture_enabled and self.gesture_detector is not None else "standby"
        vlm_status = "online" if self.vision_agent.enabled and not self.latest_agent_error else "standby"
        llm_status = "online" if self.agent_enabled and not self.latest_agent_error else "standby"
        agent_status = "thinking" if self.agent_job_running else ("online" if self.agent_enabled else "standby")
        return {
            "connected": camera_online,
            "dashboard_version": DASHBOARD_VERSION,
            "battery": self.battery_percent,
            "gesture_name": gesture_info["current"],
            "target_name": "person" if target is not None else "none",
            "linear_x": round(float(self.latest_safe_cmd.linear.x), 3),
            "angular_z": round(float(self.latest_safe_cmd.angular.z), 3),
            "yolo_status": yolo_status,
            "hand_status": hand_status,
            "vlm_status": vlm_status,
            "llm_status": llm_status,
            "agent_status": agent_status,
            "nav_status": "standby",
            "camera_status": "online" if camera_online else "offline",
            "fps": round(self.fps, 2),
            "mode": self.gesture_controller.mode,
            "camera_topic": self.camera_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "robot_ip": self.robot_ip,
            "people_count": len(self.latest_people),
            "target": target_info,
            "target_distance_m": self.target_distance_m,
            "gesture": gesture_info,
            "gesture_enabled": self.gesture_enabled,
            "gesture_logs": list(self.gesture_controller.logs),
            "scene": self.latest_scene,
            "agent": {
                "enabled": self.agent_enabled,
                "status": agent_status,
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
            "raw_cmd": twist_to_dict(self.latest_cmd),
            "safe_cmd": twist_to_dict(self.latest_safe_cmd),
            "control_reason": self.latest_reason,
            "safety_reason": self.latest_safety_reason,
            "event_logs": list(self.event_logs),
            "last_image_age_sec": round(time.time() - self.last_image_time, 2) if self.last_image_time else None,
            "last_target_age_sec": round(time.time() - self.last_target_time, 2) if self.last_target_time else None,
        }

    def destroy_node(self) -> bool:
        self.publish_stop("destroy node")
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
