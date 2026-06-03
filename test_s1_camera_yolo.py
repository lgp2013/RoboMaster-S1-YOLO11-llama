"""Test RoboMaster S1 camera streaming and YOLO11 detection.

Press q in the OpenCV window to quit.
"""

from __future__ import annotations

import cv2
from robomaster import robot
from ultralytics import YOLO

from config import ROBOT_CONN_TYPE, YOLO_CONF, YOLO_IMGSZ, YOLO_MODEL


def main() -> int:
    ep_robot = robot.Robot()
    camera = None

    try:
        print(f"Loading YOLO model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)

        print(f"Connecting RoboMaster S1: conn_type={ROBOT_CONN_TYPE}")
        ep_robot.initialize(conn_type=ROBOT_CONN_TYPE)
        camera = ep_robot.camera

        print("Starting video stream...")
        camera.start_video_stream(display=False)

        print("Press q to quit.")
        while True:
            frame = camera.read_cv2_image(strategy="newest", timeout=5)
            if frame is None:
                print("[WARN] No camera frame received.")
                continue

            results = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )
            annotated = results[0].plot()
            cv2.imshow("RoboMaster S1 YOLO11", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 0
    except Exception as exc:
        print(f"[FAIL] Camera/YOLO test failed: {exc}")
        return 1
    finally:
        if camera is not None:
            try:
                camera.stop_video_stream()
            except Exception as exc:
                print(f"[WARN] camera.stop_video_stream failed: {exc}")
        try:
            ep_robot.close()
        except Exception as exc:
            print(f"[WARN] ep_robot.close failed: {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
