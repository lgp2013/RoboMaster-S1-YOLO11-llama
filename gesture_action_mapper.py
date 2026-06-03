"""Map normalized hand gestures to conservative robot action suggestions."""

from __future__ import annotations

from typing import Optional, Tuple


GESTURE_ACTIONS = {
    "open_palm": "stop",
    "fist": "stop",
    "peace": "gimbal_up",
    "point_left": "gimbal_left",
    "point_right": "gimbal_right",
    "thumbs_up": "gimbal_up",
    "thumbs_down": "gimbal_down",
}


def map_gesture_to_action(gesture: str) -> Tuple[Optional[str], str]:
    """Return (action, reason) for a recognized gesture.

    Chassis forward/backward is intentionally not mapped by default. Gesture
    output is still only a suggestion and must pass through SafetyGuard.
    """
    normalized = (gesture or "none").strip().lower()
    action = GESTURE_ACTIONS.get(normalized)
    if action is None:
        return None, f"gesture '{normalized}' has no mapped action"
    return action, f"gesture '{normalized}' mapped to {action}"


if __name__ == "__main__":
    for name in ["open_palm", "fist", "peace", "point_left", "point_right", "none"]:
        print(f"{name}: {map_gesture_to_action(name)}")
