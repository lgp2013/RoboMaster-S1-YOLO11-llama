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
