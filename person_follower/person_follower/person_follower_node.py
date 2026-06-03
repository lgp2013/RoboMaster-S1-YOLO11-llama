"""ROS2 Foxy node: YOLO11 person tracking and RoboMaster S1 following.

流程：
1. 订阅 /camera/image_color；
2. YOLO11 检测 person；
3. 多人时选择面积最大的 person；
4. 根据目标中心和检测框高度生成 /cmd_vel；
5. SafetyGuard 做最终速度限制；
6. Flask Dashboard 显示实时视频和状态。
"""

import threading
import time
from typing import Dict, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

from .flask_dashboard import DashboardServer
from .follower_control import ControlConfig, PersonFollowerController, SafetyGuard, twist_to_dict
from .yolo_detector import PersonDetection, YoloPersonDetector


class PersonFollowerNode(Node):
    """RoboMaster S1 人体跟随节点。"""

    def __init__(self) -> None:
        super().__init__("person_follower")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.camera_topic = self.get_parameter("camera_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.lost_target_timeout_sec = float(self.get_parameter("lost_target_timeout_sec").value)
        self.target_distance_m = float(self.get_parameter("target_distance_m").value)

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
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.latest_frame = None
        self.latest_annotated = None
        self.latest_people = []
        self.latest_target: Optional[PersonDetection] = None
        self.latest_bbox_height_ratio = 0.0
        self.latest_reason = "starting"
        self.latest_safety_reason = "starting"
        self.latest_cmd = Twist()
        self.latest_safe_cmd = Twist()
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
            )
            self.dashboard.start()
            self.get_logger().info(
                "Flask Dashboard started at http://%s:%s"
                % (self.get_parameter("dashboard_host").value, self.get_parameter("dashboard_port").value)
            )

        self.get_logger().info("Person follower node started")
        self.get_logger().info("camera_topic=%s cmd_vel_topic=%s" % (self.camera_topic, self.cmd_vel_topic))

    def _declare_parameters(self) -> None:
        defaults = {
            "camera_topic": "/camera/image_color",
            "cmd_vel_topic": "/cmd_vel",
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
            "command_timeout_sec": 0.5,
            "enable_backward": True,
            "enable_auto_move": True,
            "dashboard_enabled": True,
            "dashboard_host": "0.0.0.0",
            "dashboard_port": 8088,
            "jpeg_quality": 80,
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

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error("cv_bridge convert failed: %s" % exc)
            return

        try:
            people = self.detector.detect(frame)
            target = self.detector.select_largest_person(people)
            annotated = frame.copy()
            self.detector.draw_detections(annotated, people, target)
            self._update_fps()

            with self.lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                self.latest_people = people
                self.latest_target = target
                self.last_image_time = time.time()
                if target is not None:
                    self.last_target_time = time.time()

            if self.dashboard is not None:
                self.dashboard.update(annotated, self._status_dict())

        except Exception as exc:
            self.get_logger().error("image processing failed: %s" % exc)
            self.publish_stop()

    def _update_fps(self) -> None:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

    def control_loop(self) -> None:
        with self.lock:
            frame = self.latest_frame
            target = self.latest_target
            last_target_time = self.last_target_time

        if frame is None:
            self.publish_stop()
            return

        now = time.time()
        has_fresh_target = target is not None and (now - last_target_time) <= self.lost_target_timeout_sec
        if not has_fresh_target:
            safe_cmd, safety_reason = self.safety_guard.filter_cmd(Twist(), False, 0.0)
            self.cmd_pub.publish(safe_cmd)
            self.latest_cmd = Twist()
            self.latest_safe_cmd = safe_cmd
            self.latest_reason = "target lost"
            self.latest_safety_reason = safety_reason
            return

        height, width = frame.shape[:2]
        cmd, reason, bbox_height_ratio = self.controller.compute_cmd(target, width, height)
        safe_cmd, safety_reason = self.safety_guard.filter_cmd(cmd, True, 0.0)
        self.cmd_pub.publish(safe_cmd)

        self.latest_cmd = cmd
        self.latest_safe_cmd = safe_cmd
        self.latest_bbox_height_ratio = bbox_height_ratio
        self.latest_reason = reason
        self.latest_safety_reason = safety_reason

    def publish_stop(self) -> None:
        stop = Twist()
        self.cmd_pub.publish(stop)
        self.latest_cmd = stop
        self.latest_safe_cmd = stop
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
        return {
            "fps": round(self.fps, 2),
            "camera_topic": self.camera_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "people_count": len(self.latest_people),
            "target": target_info,
            "target_distance_m": self.target_distance_m,
            "raw_cmd": twist_to_dict(self.latest_cmd),
            "safe_cmd": twist_to_dict(self.latest_safe_cmd),
            "control_reason": self.latest_reason,
            "safety_reason": self.latest_safety_reason,
            "last_image_age_sec": round(time.time() - self.last_image_time, 2) if self.last_image_time else None,
            "last_target_age_sec": round(time.time() - self.last_target_time, 2) if self.last_target_time else None,
        }

    def destroy_node(self) -> bool:
        self.publish_stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, stopping robot")
    except Exception as exc:
        node.get_logger().error("Unhandled exception: %s" % exc)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
