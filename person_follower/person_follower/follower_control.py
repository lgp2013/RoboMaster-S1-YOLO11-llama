"""Tracking and follow control helpers for the RoboMaster S1."""

from dataclasses import dataclass
from typing import Optional, Tuple

from geometry_msgs.msg import Twist

from .utils import clamp
from .yolo_detector import PersonDetection


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
    target_distance_m: float
    enable_backward: bool
    enable_auto_move: bool
    gimbal_target_y_ratio: float = 0.38
    gimbal_frame_center_y_ratio: float = 0.42
    gimbal_yaw_gain: float = 2.0
    gimbal_pitch_gain: float = 1.5
    max_gimbal_yaw_speed: float = 2.0
    max_gimbal_pitch_speed: float = 1.5
    min_gimbal_speed: float = 0.1
    gimbal_deadzone_px: int = 20
    gimbal_vertical_deadzone_px: int = 12


class PersonFollowerController:
    """Compute chassis and gimbal commands from YOLO person detections."""

    def __init__(self, config: ControlConfig) -> None:
        self.config = config

    def _desired_bbox_height_ratio(self) -> float:
        reference_distance_m = 1.0
        desired_ratio = self.config.target_bbox_height_ratio * (
            reference_distance_m / max(0.6, float(self.config.target_distance_m))
        )
        return clamp(desired_ratio, 0.32, 0.78)

    def compute_cmd(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Twist, str, float]:
        cmd = Twist()
        if target is None:
            return cmd, "no target", 0.0

        error_x = target.center_x - frame_width / 2.0
        if abs(error_x) > self.config.center_threshold_px:
            normalized_error = error_x / max(1.0, frame_width / 2.0)
            angular = -self.config.yaw_gain * normalized_error
            angular = clamp(angular, -self.config.max_angular_speed, self.config.max_angular_speed)
            if 0 < abs(angular) < self.config.min_angular_speed:
                angular = self.config.min_angular_speed if angular > 0 else -self.config.min_angular_speed
            cmd.angular.z = angular

        bbox_height_ratio = target.height / max(1.0, float(frame_height))
        desired_bbox_height_ratio = self._desired_bbox_height_ratio()
        distance_error = desired_bbox_height_ratio - bbox_height_ratio
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
            f"target_bbox_ratio={desired_bbox_height_ratio:.3f}, "
            f"linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}"
        )
        return cmd, reason, bbox_height_ratio

    def compute_gimbal_cmd(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Twist, str]:
        gimbal_cmd = Twist()
        if target is None:
            return gimbal_cmd, "no target for gimbal"

        aim_x = target.center_x
        aim_y = target.y1 + target.height * self.config.gimbal_target_y_ratio
        desired_frame_y = frame_height * self.config.gimbal_frame_center_y_ratio

        error_x = aim_x - frame_width / 2.0
        error_y = aim_y - desired_frame_y

        if abs(error_x) > self.config.gimbal_deadzone_px:
            normalized_error_x = error_x / max(1.0, frame_width / 2.0)
            yaw_speed = self.config.gimbal_yaw_gain * normalized_error_x
            yaw_speed = clamp(yaw_speed, -self.config.max_gimbal_yaw_speed, self.config.max_gimbal_yaw_speed)
            if 0 < abs(yaw_speed) < self.config.min_gimbal_speed:
                yaw_speed = self.config.min_gimbal_speed if yaw_speed > 0 else -self.config.min_gimbal_speed
            gimbal_cmd.angular.z = yaw_speed

        if abs(error_y) > self.config.gimbal_vertical_deadzone_px:
            normalized_error_y = error_y / max(1.0, frame_height / 2.0)
            pitch_speed = self.config.gimbal_pitch_gain * normalized_error_y
            pitch_speed = clamp(pitch_speed, -self.config.max_gimbal_pitch_speed, self.config.max_gimbal_pitch_speed)
            if 0 < abs(pitch_speed) < self.config.min_gimbal_speed:
                pitch_speed = self.config.min_gimbal_speed if pitch_speed > 0 else -self.config.min_gimbal_speed
            gimbal_cmd.angular.y = pitch_speed

        reason = (
            f"gimbal_error_x={error_x:.1f}, gimbal_error_y={error_y:.1f}, "
            f"yaw_speed={gimbal_cmd.angular.z:.2f}, pitch_speed={gimbal_cmd.angular.y:.2f}"
        )
        return gimbal_cmd, reason


class SafetyGuard:
    """Legacy local safety helper kept for compatibility."""

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
