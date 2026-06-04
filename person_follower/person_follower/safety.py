"""Final safety layer for all RoboMaster S1 velocity commands."""

from typing import Tuple

from geometry_msgs.msg import Twist

from .utils import clamp


class SafetyGuard:
    """最终安全层，负责限速、超时停机和统一零速度回退。"""

    def __init__(
        self,
        max_linear_speed: float,
        max_angular_speed: float,
        command_timeout_sec: float,
        image_timeout_sec: float = 1.0,
    ) -> None:
        self.max_linear_speed = abs(float(max_linear_speed))
        self.max_angular_speed = abs(float(max_angular_speed))
        self.command_timeout_sec = float(command_timeout_sec)
        self.image_timeout_sec = float(image_timeout_sec)

    def stop_cmd(self) -> Twist:
        """生成零速度命令。"""
        return Twist()

    def filter_cmd(
        self,
        cmd: Twist,
        allow_motion: bool,
        image_age_sec: float,
        command_age_sec: float = 0.0,
    ) -> Tuple[Twist, str]:
        """对任意底盘命令做最终过滤。"""
        safe = Twist()
        if image_age_sec > self.image_timeout_sec:
            return safe, "stop: image timeout"
        if command_age_sec > self.command_timeout_sec:
            return safe, "stop: command timeout"
        if not allow_motion:
            return safe, "stop: motion disabled by mode"

        safe.linear.x = clamp(cmd.linear.x, -self.max_linear_speed, self.max_linear_speed)
        safe.linear.y = clamp(cmd.linear.y, -self.max_linear_speed, self.max_linear_speed)
        safe.angular.z = clamp(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)

        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        return safe, "safe"
