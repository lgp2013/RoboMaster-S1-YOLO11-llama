"""Central configuration for the RoboMaster S1 YOLO11 llama.cpp demo."""

ROBOT_CONN_TYPE = "sta"

LLM_BASE_URL = "http://10.10.10.156:8080"
LLM_MODEL = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"

YOLO_MODEL = "yolo11n.pt"
YOLO_CONF = 0.45
YOLO_IMGSZ = 640

LLM_INTERVAL_SECONDS = 3

SAFE_FORWARD_SPEED = 0.2
SAFE_BACKWARD_SPEED = -0.2
SAFE_TURN_SPEED = 20
SAFE_GIMBAL_YAW_SPEED = 20
SAFE_GIMBAL_PITCH_SPEED = 12.5

PERSON_CLASS_ID = 0

# Web target-lock demo settings.
LOCKED_PERSON_DIR = "media_locked_people"
TARGET_FOLLOW_CONTROL_INTERVAL_SECONDS = 0.2
TARGET_CENTER_DEADZONE_X = 45
TARGET_CENTER_DEADZONE_Y = 40

# Area-ratio thresholds are camera based approximations for roughly keeping
# a person in a comfortable 1-2 meter band. Tune them in the web page logs for
# your room, lens angle, and person size.
TARGET_FAR_AREA_RATIO = 0.055
TARGET_NEAR_AREA_RATIO = 0.18

# Locked target search settings.
TARGET_MATCH_MIN_SCORE = 0.35
TARGET_LOST_GIMBAL_SEARCH_SECONDS = 8
TARGET_LOST_CHASSIS_SEARCH_SECONDS = 20
TARGET_SEARCH_TURN_SPEED = 12
TARGET_SEARCH_FRONT_WALL_MM = 450

# Safety Guard configuration for obstacle avoidance and movement control
AUTO_MOVE_ENABLED = False  # Disable LLM direct forward/backward control
MAX_FORWARD_DURATION = 0.3  # Maximum forward movement duration in seconds
FORWARD_COOLDOWN = 0.8  # Minimum time between forward movements
MAX_PERSON_AREA_RATIO_FOR_FORWARD = 0.12  # Max person area ratio to allow forward
MIN_PERSON_AREA_RATIO_FOR_FORWARD = 0.02  # Min person area ratio to allow forward
ALLOW_TURN_IN_PLACE = True  # Allow in-place turning when no forward movement
EMERGENCY_STOP = False  # Emergency stop flag

# OpenCV dashboard layout.
DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720
VIDEO_PANEL_WIDTH = 820
VIDEO_PANEL_HEIGHT = 480
STATUS_PANEL_WIDTH = 400
STATUS_PANEL_HEIGHT = 480
LOG_PANEL_HEIGHT = 130
MAX_LOG_LINES = 6
ENABLE_DASHBOARD = True
DASHBOARD_COLOR_BG = (18, 24, 31)
DASHBOARD_COLOR_PANEL = (28, 38, 49)
DASHBOARD_COLOR_PANEL_DARK = (16, 23, 31)
DASHBOARD_COLOR_TEXT = (235, 242, 248)
DASHBOARD_COLOR_MUTED = (166, 181, 194)
DASHBOARD_COLOR_GREEN = (80, 210, 130)
DASHBOARD_COLOR_YELLOW = (0, 210, 255)
DASHBOARD_COLOR_RED = (70, 90, 240)
DASHBOARD_COLOR_BLUE = (230, 150, 70)
DASHBOARD_COLOR_BORDER = (74, 95, 116)

# MediaPipe Hands gesture detection.
ENABLE_GESTURE_DETECTION = True
MEDIAPIPE_MAX_NUM_HANDS = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.6
GESTURE_ACTION_ENABLED = True
