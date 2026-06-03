"""MediaPipe Hands based gesture detector for the RoboMaster demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from config import (
    MEDIAPIPE_MAX_NUM_HANDS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime dependency message.
    mp = None


@dataclass
class GestureResult:
    gesture: str = "none"
    confidence: float = 0.0
    handedness: str = "unknown"
    landmarks: Optional[object] = None


class GestureDetector:
    """Detect simple static hand gestures from BGR OpenCV frames."""

    def __init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe is not installed. Run: pip install mediapipe")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )

    def close(self) -> None:
        self.hands.close()

    def detect(self, frame_bgr) -> GestureResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return GestureResult()

        landmarks = results.multi_hand_landmarks[0]
        handedness = "unknown"
        confidence = 0.0
        if results.multi_handedness:
            handedness_info = results.multi_handedness[0].classification[0]
            handedness = handedness_info.label
            confidence = float(handedness_info.score)

        fingers = self._extended_fingers(landmarks)
        gesture = self._classify_gesture(landmarks, fingers)
        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            handedness=handedness,
            landmarks=landmarks,
        )

    def draw(self, frame_bgr, result: GestureResult) -> None:
        if result.landmarks is None:
            return
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            result.landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        cv2.putText(
            frame_bgr,
            f"gesture: {result.gesture}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )

    def _extended_fingers(self, landmarks) -> List[bool]:
        lm = landmarks.landmark
        # Thumb uses x-axis relation because it bends sideways in image space.
        thumb_extended = abs(lm[4].x - lm[2].x) > abs(lm[3].x - lm[2].x) * 1.2
        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_extended = lm[16].y < lm[14].y
        pinky_extended = lm[20].y < lm[18].y
        return [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

    def _classify_gesture(self, landmarks, fingers: List[bool]) -> str:
        thumb, index, middle, ring, pinky = fingers
        count = sum(1 for value in fingers if value)
        lm = landmarks.landmark

        if count >= 4:
            return "open_palm"
        if count == 0:
            return "fist"
        if index and middle and not ring and not pinky:
            return "peace"
        if index and not middle and not ring and not pinky:
            wrist_x = lm[0].x
            tip_x = lm[8].x
            if tip_x < wrist_x - 0.08:
                return "point_left"
            if tip_x > wrist_x + 0.08:
                return "point_right"
            return "point"
        if thumb and not index and not middle and not ring and not pinky:
            return "thumbs_up" if lm[4].y < lm[2].y else "thumbs_down"
        return "unknown"


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.detect(frame)
            detector.draw(frame, result)
            cv2.imshow("Gesture Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
