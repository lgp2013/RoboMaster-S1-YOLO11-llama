"""FOLLOW_AGENT control helpers for locked-target follow."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from geometry_msgs.msg import Twist

from .utils import clamp
from .yolo_detector import PersonDetection


@dataclass
class ControlConfig:
    center_dead_zone_ratio: float
    too_far_height_ratio: float
    too_close_height_ratio: float
    max_linear_x: float
    max_angular_z: float
    max_gimbal_yaw_speed: float
    kp_chassis_yaw: float
    kp_gimbal_yaw: float
    chassis_yaw_invert: bool
    gimbal_yaw_invert: bool
    smooth_alpha: float
    gimbal_first: bool
    chassis_rotate_delay_seconds: float
    debug_follow_only: bool
    enable_gimbal_control: bool
    enable_chassis_rotation: bool
    enable_distance_control: bool
    gimbal_target_y_ratio: float
    gimbal_frame_center_y_ratio: float
    gimbal_yaw_gain: float
    gimbal_pitch_gain: float
    max_gimbal_pitch_speed: float
    min_gimbal_speed: float
    gimbal_deadzone_px: int
    gimbal_vertical_deadzone_px: int


class PersonFollowerController:
    """Compute follow commands from a locked person target."""

    def __init__(self, config: ControlConfig) -> None:
        self.config = config

    def compute_metrics(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, float]:
        if target is None:
            return {
                "error_x": 0.0,
                "error_x_normalized": 0.0,
                "dead_zone_x": frame_width * self.config.center_dead_zone_ratio,
                "target_height_ratio": 0.0,
                "error_y": 0.0,
                "error_y_normalized": 0.0,
            }

        error_x = target.center_x - frame_width / 2.0
        error_x_normalized = clamp(error_x / max(1.0, frame_width / 2.0), -1.0, 1.0)
        dead_zone_x = max(8.0, frame_width * self.config.center_dead_zone_ratio)
        target_height_ratio = target.height / max(1.0, float(frame_height))

        aim_y = target.y1 + target.height * self.config.gimbal_target_y_ratio
        desired_frame_y = frame_height * self.config.gimbal_frame_center_y_ratio
        error_y = aim_y - desired_frame_y
        error_y_normalized = clamp(error_y / max(1.0, frame_height / 2.0), -1.0, 1.0)

        return {
            "error_x": float(error_x),
            "error_x_normalized": float(error_x_normalized),
            "dead_zone_x": float(dead_zone_x),
            "target_height_ratio": float(target_height_ratio),
            "error_y": float(error_y),
            "error_y_normalized": float(error_y_normalized),
        }

    def compute_cmd(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
        allow_chassis_rotation: bool,
    ) -> Tuple[Twist, Dict[str, float]]:
        cmd = Twist()
        metrics = self.compute_metrics(target, frame_width, frame_height)
        if target is None:
            return cmd, metrics

        if self.config.enable_distance_control:
            target_height_ratio = metrics["target_height_ratio"]
            if target_height_ratio < self.config.too_far_height_ratio:
                cmd.linear.x = min(0.08, self.config.max_linear_x)
            elif target_height_ratio > self.config.too_close_height_ratio:
                cmd.linear.x = max(-0.06, -self.config.max_linear_x)

        if self.config.enable_chassis_rotation and allow_chassis_rotation:
            if abs(metrics["error_x"]) >= metrics["dead_zone_x"]:
                angular = -self.config.kp_chassis_yaw * metrics["error_x_normalized"]
                if self.config.chassis_yaw_invert:
                    angular *= -1.0
                cmd.angular.z = clamp(angular, -self.config.max_angular_z, self.config.max_angular_z)

        return cmd, metrics

    def compute_gimbal_cmd(
        self,
        target: Optional[PersonDetection],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Twist, Dict[str, float]]:
        gimbal_cmd = Twist()
        metrics = self.compute_metrics(target, frame_width, frame_height)
        if target is None or not self.config.enable_gimbal_control:
            return gimbal_cmd, metrics

        if abs(metrics["error_x"]) >= max(metrics["dead_zone_x"], float(self.config.gimbal_deadzone_px)):
            yaw_speed = self.config.gimbal_yaw_gain * metrics["error_x_normalized"]
            if self.config.gimbal_yaw_invert:
                yaw_speed *= -1.0
            yaw_speed = clamp(yaw_speed, -self.config.max_gimbal_yaw_speed, self.config.max_gimbal_yaw_speed)
            if 0 < abs(yaw_speed) < self.config.min_gimbal_speed:
                yaw_speed = self.config.min_gimbal_speed if yaw_speed > 0 else -self.config.min_gimbal_speed
            gimbal_cmd.angular.z = yaw_speed

        if abs(metrics["error_y"]) >= float(self.config.gimbal_vertical_deadzone_px):
            pitch_speed = self.config.gimbal_pitch_gain * metrics["error_y_normalized"]
            pitch_speed = clamp(pitch_speed, -self.config.max_gimbal_pitch_speed, self.config.max_gimbal_pitch_speed)
            if 0 < abs(pitch_speed) < self.config.min_gimbal_speed:
                pitch_speed = self.config.min_gimbal_speed if pitch_speed > 0 else -self.config.min_gimbal_speed
            gimbal_cmd.angular.y = pitch_speed

        return gimbal_cmd, metrics


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
