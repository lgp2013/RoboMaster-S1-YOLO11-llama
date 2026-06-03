"""YOLO11 person detector.

只保留 person 类别，并在多人时选择面积最大的目标。
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2


@dataclass
class PersonDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "person"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


class YoloPersonDetector:
    """封装 YOLO11 检测，避免主节点堆太多视觉代码。"""

    def __init__(self, model_path: str, confidence: float, imgsz: int, person_class_id: int = 0) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Python 依赖 ultralytics。请在 ROS2 工作空间中执行：\n"
                "  cd ~/rm_ws\n"
                "  python3 -m pip install -r src/person_follower/requirements.txt\n"
                "如果使用 root 运行 ROS2，也必须用 root 的 python3 安装依赖。"
            ) from exc

        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.person_class_id = person_class_id
        self.model = YOLO(model_path)

    def detect(self, frame) -> List[PersonDetection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        people: List[PersonDetection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if class_id != self.person_class_id:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            people.append(PersonDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))
        return people

    @staticmethod
    def select_largest_person(people: List[PersonDetection]) -> Optional[PersonDetection]:
        if not people:
            return None
        return max(people, key=lambda item: item.area)

    @staticmethod
    def draw_detections(frame, people: List[PersonDetection], target: Optional[PersonDetection]) -> None:
        for person in people:
            is_target = target is not None and person is target
            color = (0, 255, 0) if is_target else (80, 180, 255)
            thickness = 3 if is_target else 2
            cv2.rectangle(
                frame,
                (int(person.x1), int(person.y1)),
                (int(person.x2), int(person.y2)),
                color,
                thickness,
            )
            label = f"person {person.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (int(person.x1), max(20, int(person.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        if target is not None:
            cv2.circle(frame, (int(target.center_x), int(target.center_y)), 5, (0, 255, 255), -1)
