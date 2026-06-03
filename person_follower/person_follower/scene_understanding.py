"""Scene information extraction for the Vision-Language-Agent stage."""

from typing import Dict, List, Optional

from .gesture_detector import GestureResult
from .yolo_detector import PersonDetection


def build_scene_summary(
    people: List[PersonDetection],
    gestures: List[GestureResult],
    target: Optional[PersonDetection],
    frame_width: int,
    frame_height: int,
    vision_description: str = "",
) -> Dict[str, object]:
    """把 YOLO、手势和 VLM 描述整理成结构化场景信息。

    这里不直接做决策，只负责给 Agent 和 Dashboard 提供统一输入。
    """
    gesture_names = [item.gesture for item in gestures if item.gesture not in ("unknown", "none")]
    scene = {
        "people": len(people),
        "gestures": gesture_names,
        "objects": extract_known_objects(vision_description),
        "scene": infer_scene_type(vision_description),
        "description": vision_description or "当前还没有视觉语言模型描述。",
        "target": None,
    }
    if target is not None:
        area_ratio = target.area / max(1.0, float(frame_width * frame_height))
        scene["target"] = {
            "class": "person",
            "confidence": round(target.confidence, 3),
            "center_x": round(target.center_x, 1),
            "center_y": round(target.center_y, 1),
            "area_ratio": round(area_ratio, 4),
            "bbox_height_ratio": round(target.height / max(1.0, float(frame_height)), 4),
            "horizontal_position": horizontal_position(target.center_x, frame_width),
        }
    return scene


def horizontal_position(center_x: float, frame_width: int) -> str:
    if center_x < frame_width * 0.4:
        return "left"
    if center_x > frame_width * 0.6:
        return "right"
    return "center"


def infer_scene_type(description: str) -> str:
    text = description.lower()
    if any(word in text for word in ["office", "desk", "chair", "computer"]):
        return "office"
    if any(word in text for word in ["outdoor", "road", "street"]):
        return "outdoor"
    if any(word in text for word in ["room", "indoor"]):
        return "indoor"
    return "unknown"


def extract_known_objects(description: str) -> List[str]:
    """从 VLM 文本里粗略抽取常见物体，后续可替换为开放词表检测。"""
    text = description.lower()
    candidates = [
        "chair",
        "table",
        "desk",
        "bottle",
        "cup",
        "wall",
        "door",
        "cabinet",
        "sofa",
        "computer",
        "phone",
    ]
    return [item for item in candidates if item in text]
