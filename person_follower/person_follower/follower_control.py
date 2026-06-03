"""Tracking and safety control logic for RoboMaster S1 person follower."""

from dataclasses import dataclass
from typing import Optional, Tuple

from geometry_msgs.msg import Twist

from .yolo_detector import PersonDetection


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class ControlConfig:
    center_threshold_px: int
    yaw_gain: float
    distance_gain: float
    max_linear_speed: float
    min_linear_speed: float
    max_angular_speed: float
    min_angular_speed: float
    target_bbox_height_ratio: float
    bbox_height_tolerance: float
    enable_backward: bool
    enable_auto_move: bool


class PersonFollowerController:
    """完全基于 YOLO 检测框的跟随控制器，不使用 OpenCV Tracker。"""

    def __init__(self, config: ControlConfig) -> None:
        self.config = config

    def compute_cmd(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Twist, str, float]:
        cmd = Twist()
        if target is None:
            return cmd, "no target", 0.0

        # 横向误差：目标在右侧时 error > 0。ROS 中 angular.z 正值为左转，
        # 所以向右转需要负角速度。
        error_x = target.center_x - frame_width / 2.0
        if abs(error_x) > self.config.center_threshold_px:
            normalized_error = error_x / max(1.0, frame_width / 2.0)
            angular = -self.config.yaw_gain * normalized_error
            angular = clamp(angular, -self.config.max_angular_speed, self.config.max_angular_speed)
            if 0 < abs(angular) < self.config.min_angular_speed:
                angular = self.config.min_angular_speed if angular > 0 else -self.config.min_angular_speed
            cmd.angular.z = angular

        bbox_height_ratio = target.height / max(1.0, float(frame_height))
        distance_error = self.config.target_bbox_height_ratio - bbox_height_ratio
        if self.config.enable_auto_move and abs(distance_error) > self.config.bbox_height_tolerance:
            linear = self.config.distance_gain * distance_error
            linear = clamp(linear, -self.config.max_linear_speed, self.config.max_linear_speed)
            if linear < 0 and not self.config.enable_backward:
                linear = 0.0
            if 0 < abs(linear) < self.config.min_linear_speed:
                linear = self.config.min_linear_speed if linear > 0 else -self.config.min_linear_speed
            cmd.linear.x = linear

        reason = (
            f"error_x={error_x:.1f}, bbox_height_ratio={bbox_height_ratio:.3f}, "
            f"linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}"
        )
        return cmd, reason, bbox_height_ratio


class SafetyGuard:
    """最终安全层：限制速度、处理丢失目标和异常命令。"""

    def __init__(
        self,
        max_linear_speed: float,
        max_angular_speed: float,
        command_timeout_sec: float,
    ) -> None:
        self.max_linear_speed = abs(max_linear_speed)
        self.max_angular_speed = abs(max_angular_speed)
        self.command_timeout_sec = command_timeout_sec

    def filter_cmd(self, cmd: Twist, has_target: bool, command_age_sec: float) -> Tuple[Twist, str]:
        safe = Twist()
        if not has_target:
            return safe, "stop: no target"
        if command_age_sec > self.command_timeout_sec:
            return safe, "stop: command timeout"

        safe.linear.x = clamp(cmd.linear.x, -self.max_linear_speed, self.max_linear_speed)
        safe.angular.z = clamp(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)

        # S1 跟随只需要前后和原地旋转，禁止横移和其他轴。
        safe.linear.y = 0.0
        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        return safe, "safe"


def twist_to_dict(cmd: Twist):
    return {
        "linear_x": round(float(cmd.linear.x), 3),
        "angular_z": round(float(cmd.angular.z), 3),
    }
