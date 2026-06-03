"""MediaPipe Hands gesture detector.

识别手部关键点，并用规则分类常见手势。规则分类不是深度手势模型，
但足够用于 RoboMaster 调试阶段的安全模式切换。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    handedness: str
    landmarks: List[Tuple[float, float, float]]
    points: List[Tuple[int, int]]


class GestureDetector:
    """MediaPipe Hands 封装，输出手势名称、置信度和关键点。"""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Python 依赖 mediapipe。请执行：\n"
                "  python3 -m pip install -r src/person_follower/requirements.txt"
            ) from exc

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    def detect(self, frame_bgr) -> List[GestureResult]:
        """识别当前帧中的手势。"""
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        output: List[GestureResult] = []
        handedness_list = results.multi_handedness or []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Unknown"
            score = 0.6
            if idx < len(handedness_list):
                item = handedness_list[idx].classification[0]
                handedness = item.label
                score = float(item.score)

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
            gesture, gesture_conf = self._classify(landmarks, handedness, score)
            output.append(
                GestureResult(
                    gesture=gesture,
                    confidence=gesture_conf,
                    handedness=handedness,
                    landmarks=landmarks,
                    points=points,
                )
            )
        return output

    def draw(self, frame_bgr, gestures: List[GestureResult]) -> None:
        """在画面上绘制手部关键点和手势名称。"""
        for result in gestures:
            for start, end in self.hand_connections:
                p1 = result.points[start]
                p2 = result.points[end]
                cv2.line(frame_bgr, p1, p2, (255, 190, 80), 2, cv2.LINE_AA)
            for point in result.points:
                cv2.circle(frame_bgr, point, 4, (80, 220, 255), -1)
            if result.points:
                x, y = result.points[0]
                cv2.putText(
                    frame_bgr,
                    "%s %.2f" % (result.gesture, result.confidence),
                    (max(0, x - 20), max(24, y - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 255, 160),
                    2,
                    cv2.LINE_AA,
                )

    def close(self) -> None:
        self.hands.close()

    def _classify(self, landmarks: List[Tuple[float, float, float]], handedness: str, score: float) -> Tuple[str, float]:
        """基于关键点几何关系做手势分类。"""
        if len(landmarks) < 21:
            return "unknown", 0.0

        fingers = self._extended_fingers(landmarks)
        extended_count = sum(1 for value in fingers.values() if value)

        thumb_index_distance = self._distance_2d(landmarks[4], landmarks[8])
        if thumb_index_distance < 0.055 and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            return "ok_sign", min(0.98, score)

        if extended_count >= 4:
            return "open_palm", min(0.95, score)
        if extended_count == 0:
            return "fist", min(0.95, score)
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "victory", min(0.95, score)
        if fingers["thumb"] and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "thumbs_up", min(0.90, score)
        if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            wrist_x = landmarks[0][0]
            tip_x = landmarks[8][0]
            if tip_x < wrist_x - 0.08:
                return "point_left", min(0.90, score)
            if tip_x > wrist_x + 0.08:
                return "point_right", min(0.90, score)
            return "point", min(0.75, score)

        return "unknown", min(0.50, score)

    def _extended_fingers(self, landmarks: List[Tuple[float, float, float]]):
        """判断五根手指是否伸直。y 越小表示越靠近画面上方。"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        # 拇指用横向距离判断，兼容左右手时只关心是否明显离开手掌中心。
        thumb_extended = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0]) + 0.025
        return {
            "thumb": thumb_extended,
            "index": landmarks[8][1] < landmarks[6][1] - 0.015,
            "middle": landmarks[12][1] < landmarks[10][1] - 0.015,
            "ring": landmarks[16][1] < landmarks[14][1] - 0.015,
            "pinky": landmarks[20][1] < landmarks[18][1] - 0.015,
        }

    @staticmethod
    def _distance_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5


def best_gesture(gestures: List[GestureResult]) -> Optional[GestureResult]:
    """选择置信度最高的非 unknown 手势。"""
    valid = [item for item in gestures if item.gesture != "unknown"]
    if not valid:
        return None
    return max(valid, key=lambda item: item.confidence)
