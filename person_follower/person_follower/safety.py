"""Final safety layer for all RoboMaster S1 velocity commands."""

from typing import Tuple

from geometry_msgs.msg import Twist

from .utils import clamp


class SafetyGuard:
    """最终安全层：限速、图像超时停机、异常停机。

    这里不会做真正的避障，因为普通 RGB 摄像头无法可靠判断墙、桌子、
    柜子的距离；它负责保证任何上层输出都不会变成高速或持续危险动作。
    """

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
        """对任意控制器输出做最终过滤。

        allow_motion=False 时只允许 stop。图像超过 1 秒未更新时必须 stop。
        """
        safe = Twist()
        if image_age_sec > self.image_timeout_sec:
            return safe, "stop: image timeout"
        if command_age_sec > self.command_timeout_sec:
            return safe, "stop: command timeout"
        if not allow_motion:
            return safe, "stop: motion disabled by mode"

        safe.linear.x = clamp(cmd.linear.x, -self.max_linear_speed, self.max_linear_speed)
        safe.angular.z = clamp(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)

        # S1 跟随系统只允许前后移动和原地旋转，不允许横移或其它轴速度。
        safe.linear.y = 0.0
        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        return safe, "safe"
