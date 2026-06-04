"""Common helpers for the RoboMaster S1 follower package."""

import json
import time
from typing import Any, Dict

from geometry_msgs.msg import Twist
from std_msgs.msg import String


def clamp(value: float, min_value: float, max_value: float) -> float:
    """把数值限制在安全范围内。"""
    return max(min_value, min(max_value, value))


def now_sec() -> float:
    """返回当前 wall-clock 时间，便于非 ROS 线程共享。"""
    return time.time()


def twist_to_dict(cmd: Twist) -> Dict[str, float]:
    """把 Twist 转成 Dashboard 友好的字典。"""
    return {
        "linear_x": round(float(cmd.linear.x), 3),
        "linear_y": round(float(cmd.linear.y), 3),
        "linear_z": round(float(cmd.linear.z), 3),
        "angular_x": round(float(cmd.angular.x), 3),
        "angular_y": round(float(cmd.angular.y), 3),
        "angular_z": round(float(cmd.angular.z), 3),
    }


def make_twist(linear_x: float = 0.0, angular_z: float = 0.0, linear_y: float = 0.0) -> Twist:
    """生成底盘常用 Twist，支持前后、侧移和原地转向。"""
    cmd = Twist()
    cmd.linear.x = float(linear_x)
    cmd.linear.y = float(linear_y)
    cmd.angular.z = float(angular_z)
    return cmd


def json_string_msg(payload: Dict[str, Any]) -> String:
    """把字典编码成 std_msgs/String，供状态和命令 topic 复用。"""
    msg = String()
    msg.data = json.dumps(payload, ensure_ascii=False)
    return msg
