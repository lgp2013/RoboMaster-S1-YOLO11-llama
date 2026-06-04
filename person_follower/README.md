# RoboMaster S1 ROS2 Dashboard Upgrade

This package extends the RoboMaster S1 ROS2 stack with:

- `FOLLOW_AGENT`: lock target first, then follow
- `GESTURE_AGENT`: gesture recognition and mode switching
- `VLM_AGENT`: scene understanding
- `LLM_AGENT`: action planning
- `SAFETY_AGENT`: command filtering, sleep, emergency stop
- `MANUAL_AGENT`: manual chassis and gimbal override
- `DASHBOARD_AGENT`: web control and detail APIs

The dashboard is designed around fixed summary cards on the right side and a shared detail modal for all long text and JSON.

## Directory

```text
person_follower/
|- README.md
|- launch/
|  |- follower.launch.py
|- config/
|  |- config.yaml
|- person_follower/
|  |- follower_node.py
|  |- follower_control.py
|  |- gesture_controller.py
|  |- gesture_detector.py
|  |- planner_agent.py
|  |- robot_executor.py
|  |- safety.py
|  |- scene_understanding.py
|  |- utils.py
|  |- vision_agent.py
|  |- web_dashboard.py
|  |- yolo_detector.py
|  |- static/
|  |  |- css/dashboard.css
|  |  |- js/dashboard.js
|  |- templates/
|     |- index.html
|- records/
   |- snapshots/
   |- videos/
```

`records/` is created automatically when snapshots or video recording are used.

## Install

```bash
sudo apt update
sudo apt install -y ros-foxy-cv-bridge python3-pip
```

```bash
cd ~/rm_ws
python3 -m pip install --upgrade pip
python3 -m pip install -r src/person_follower/requirements.txt
```

Optional MediaPipe support:

```bash
python3 -m pip install -r src/person_follower/requirements-mediapipe-optional.txt
```

## Build

```bash
cd ~/rm_ws
colcon build --packages-select person_follower
source install/setup.bash
```

## Exact Launch Commands

1. Start the RoboMaster S1 ROS driver:

```bash
ros2 launch robomaster_ros s1.launch conn_type:=sta
```

2. Start this package:

```bash
cd ~/rm_ws
source install/setup.bash
ros2 launch person_follower follower.launch.py
```

3. Open the dashboard:

```text
http://127.0.0.1:8088
```

For another machine on the LAN, replace `127.0.0.1` with the Ubuntu host IP.

## Dashboard Layout

Left side:

- live video
- HUD with `MODE`, `LOCK`, `TARGET`, `GESTURE`, `FPS`, velocity vector
- click video to lock a person

Right side:

- `Robot Telemetry`
- `Agent Intelligence`
- `Sub-Agent Status`
- `Model Runtime`
- `Event Log`

Each card shows only short summaries. Long text and JSON are opened in the shared detail modal.

## Follow Lock State

The follow lock state is restricted to:

- `NONE`
- `SCANNING`
- `CANDIDATE`
- `LOCKED`
- `LOST`

Rules:

- `START_FOLLOW` is disabled until a target is locked
- click the video to lock a person manually
- `AUTO LOCK` locks the best detected person
- if the locked target is lost, the robot stops and an event is logged

## Manual Override Zones

The manual control area is split into 4 zones:

1. Safety: `EMERGENCY_STOP`, `SLEEP`, `WAKE`
2. Chassis: `FORWARD`, `BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STRAFE_LEFT`, `STRAFE_RIGHT`, `STOP`
3. Gimbal: `GIMBAL_UP`, `GIMBAL_DOWN`, `GIMBAL_LEFT`, `GIMBAL_RIGHT`, `GIMBAL_CENTER`
4. Mode and utility: `START_FOLLOW`, `PAUSE_FOLLOW`, `AGENT_MODE`, `GESTURE_MODE`, `IDLE`, `SNAPSHOT`, `START_RECORD`, `STOP_RECORD`, `CLEAR_LOGS`, `RESET_TARGET`, `RECONNECT_ROBOT`

## ROS2 Topics

- chassis: `/cmd_vel`
- gimbal: `/cmd_gimbal`
- mode: `/robot/mode`
- control command: `/robot/command`
- led bridge: `/robot/led_command`
- gesture state: `/gesture/state`
- gesture command: `/gesture/command`

## HTTP APIs

Control API:

```text
POST /api/control
Content-Type: application/json
```

Example success response:

```json
{
  "success": true,
  "command": "START_FOLLOW",
  "message": "Follow started on locked target"
}
```

Supported control commands:

- `EMERGENCY_STOP`
- `SLEEP`
- `WAKE`
- `FORWARD`
- `BACKWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`
- `STRAFE_LEFT`
- `STRAFE_RIGHT`
- `GIMBAL_UP`
- `GIMBAL_DOWN`
- `GIMBAL_LEFT`
- `GIMBAL_RIGHT`
- `GIMBAL_CENTER`
- `START_FOLLOW`
- `PAUSE_FOLLOW`
- `AGENT_MODE`
- `GESTURE_MODE`
- `IDLE`
- `LOCK_TARGET`
- `RESET_TARGET`
- `SNAPSHOT`
- `START_RECORD`
- `STOP_RECORD`
- `CLEAR_LOGS`
- `RECONNECT_ROBOT`

Detail APIs:

```text
GET /api/status
GET /api/detail/telemetry
GET /api/detail/agent
GET /api/detail/logs
GET /api/detail/models
GET /api/detail/sub_agents
POST /api/logs/clear
```

Detail API response format:

```json
{
  "success": true,
  "data": {},
  "timestamp": 1710000000.123
}
```

## Exact API Commands

Auto-lock the best detected person:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"LOCK_TARGET"}'
```

Lock by clicking a point on the video:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"LOCK_TARGET","x":0.52,"y":0.41}'
```

Start follow:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"START_FOLLOW"}'
```

Sleep:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"SLEEP"}'
```

Wake:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"WAKE"}'
```

Take a snapshot:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"SNAPSHOT"}'
```

Start recording:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"START_RECORD"}'
```

Stop recording:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"STOP_RECORD"}'
```

Fetch telemetry detail:

```bash
curl http://127.0.0.1:8088/api/detail/telemetry
```

Fetch sub-agent detail:

```bash
curl http://127.0.0.1:8088/api/detail/sub_agents
```

Clear logs:

```bash
curl -X POST http://127.0.0.1:8088/api/logs/clear
```

## Validation

1. Dashboard status:

```bash
curl http://127.0.0.1:8088/api/status
```

Check that the payload includes:

- `follow_lock_state`
- `follow_enabled`
- `sub_agents`
- `recording`
- `safe_gimbal_cmd`
- `details`

2. Follow must require lock first:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"START_FOLLOW"}'
```

Expected failure:

```json
{
  "success": false,
  "command": "START_FOLLOW",
  "error": "Lock a person first before following"
}
```

3. Sleep blocks motion:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"SLEEP"}'
```

Then:

```bash
curl -X POST http://127.0.0.1:8088/api/control \
  -H "Content-Type: application/json" \
  -d '{"command":"FORWARD"}'
```

Expected result: the movement command is rejected while sleep mode is active.

4. Snapshot and recording outputs:

- snapshots are written under `records/snapshots/`
- videos are written under `records/videos/`

## Safety Notes

- `EMERGENCY_STOP` has the highest priority
- `SLEEP` blocks motion commands and agent motion output
- `Ctrl+C` publishes zero chassis and gimbal velocity
- target loss stops the robot immediately
- all agent output passes through `SAFETY_AGENT` before publish
