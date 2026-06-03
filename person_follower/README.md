# RoboMaster S1 ROS2 YOLO11 + MediaPipe 手势控制系统

这是第二阶段 RoboMaster S1 人体跟踪工程，运行环境面向 Ubuntu 20.04 + ROS2 Foxy + `robomaster_ros`。

系统流程：

```text
/camera/image_color
  -> YOLO11 检测 person
  -> MediaPipe Hands 识别手势
  -> 手势防抖与模式切换
  -> SafetyGuard 最终安全过滤
  -> /cmd_vel 控制 RoboMaster S1
```

Web Dashboard 地址：

```text
http://0.0.0.0:8088
```

## 功能

- 订阅 `/camera/image_color` 获取 RoboMaster S1 视频流。
- 使用 YOLO11 只检测 `person`。
- 多人场景下选择检测框面积最大的 person 作为跟随目标。
- 根据目标中心控制机器人左右旋转。
- 根据检测框高度估算远近，控制低速前进或后退。
- 使用 MediaPipe Hands 识别手势。
- 通过手势切换 `IDLE / FOLLOW / GESTURE_CONTROL / PATROL_READY / EMERGENCY_STOP`。
- 发布 `/gesture/state` 和 `/gesture/command`。
- Dashboard 显示视频、YOLO 框、手部关键点、手势、模式、速度和最近 10 条手势日志。
- 图像超时、异常、未知动作都会 stop。

## 目录结构

```text
person_follower/
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
├── README.md
├── resource/
│   └── person_follower
├── launch/
│   └── follower.launch.py
├── config/
│   └── config.yaml
└── person_follower/
    ├── __init__.py
    ├── follower_node.py
    ├── person_follower_node.py
    ├── yolo_detector.py
    ├── gesture_detector.py
    ├── gesture_controller.py
    ├── web_dashboard.py
    ├── flask_dashboard.py
    ├── follower_control.py
    ├── safety.py
    └── utils.py
```

`person_follower_node.py` 和 `flask_dashboard.py` 是兼容入口，新的主实现分别在 `follower_node.py` 和 `web_dashboard.py`。

## 环境说明

已验证 Topic：

```text
/cmd_vel
/odom
/imu
/battery
/camera/image_color
```

建议先确认 RoboMaster ROS 驱动正常：

```bash
ros2 run robomaster_ros discover
ros2 launch robomaster_ros s1.launch conn_type:=sta
ros2 topic hz /camera/image_color
```

## 安装依赖

安装 ROS 依赖：

```bash
sudo apt update
sudo apt install -y ros-foxy-cv-bridge python3-pip
```

安装 Python 依赖。注意：你如果用 `root` 运行 ROS2，就必须在 `root` 用户下安装这些依赖。

```bash
cd ~/rm_ws
python3 -m pip install --upgrade pip
python3 -m pip install -r src/person_follower/requirements.txt
```

国内网络慢可以使用清华源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/person_follower/requirements.txt
```

依赖检查：

```bash
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

说明：Ubuntu 20.04 + Python3.8 下，`requirements.txt` 固定使用 `mediapipe==0.10.11`，避免安装到不兼容版本。

## 构建

```bash
cd ~/rm_ws
colcon build --packages-select person_follower
source install/setup.bash
```

## 启动 RoboMaster ROS 驱动

```bash
ros2 launch robomaster_ros s1.launch conn_type:=sta
```

## 启动第二阶段程序

```bash
ros2 launch person_follower follower.launch.py
```

打开 Dashboard：

```text
http://<Ubuntu主机IP>:8088
```

本机访问：

```text
http://127.0.0.1:8088
```

## 手势说明

| 手势 | 命令 | 效果 |
| --- | --- | --- |
| open_palm | STOP | 立即停止机器人，进入紧急停止或静止模式 |
| thumbs_up | START_FOLLOW | 启动人体跟随模式 |
| fist | PAUSE | 暂停跟随，进入 IDLE |
| point_left | TURN_LEFT | 原地左转 1 秒 |
| point_right | TURN_RIGHT | 原地右转 1 秒 |
| victory | PATROL_READY | 进入巡逻预备模式，只显示状态，不导航 |
| ok_sign | RESUME | 恢复上一个模式 |

防抖规则：

- 同一手势连续识别至少 5 帧才触发。
- 同一手势触发后冷却 2 秒。
- Dashboard 会显示当前候选手势、稳定手势和冷却状态。

## 控制模式

| 模式 | 说明 |
| --- | --- |
| IDLE | 机器人静止，不跟随，不执行动作 |
| FOLLOW | 启用第一阶段 YOLO 人体跟随逻辑 |
| GESTURE_CONTROL | 执行一次性手势动作，例如左转或右转 |
| PATROL_READY | 巡逻预留模式，只显示状态，不做导航 |
| EMERGENCY_STOP | 紧急停止模式，优先级最高 |

手势优先级高于人体跟随。识别到 `open_palm` 时，节点会立即发布一次零速度 `/cmd_vel`。

## ROS2 Topic

手势状态：

```bash
ros2 topic echo /gesture/state
```

示例：

```json
{"gesture":"thumbs_up","stable":true,"confidence":0.92,"mode":"FOLLOW"}
```

手势命令：

```bash
ros2 topic echo /gesture/command
```

示例：

```json
{"command":"START_FOLLOW","source":"gesture","timestamp":123456789}
```

可选调试图像：

```yaml
gesture:
  publish_debug_image: true
```

开启后发布 `/gesture/debug_image`。

## Dashboard

页面包含：

- 实时视频画面；
- YOLO 人体检测框；
- MediaPipe 手部关键点；
- 当前手势；
- 稳定手势；
- 当前模式；
- 当前机器人速度；
- 最近 10 条手势指令日志；
- 手动停止、启动跟随、暂停跟随、清空日志按钮。

按钮通过 Flask API 调用后端，不依赖复杂前端框架。

## 配置

配置文件：

```text
config/config.yaml
```

关键配置：

```yaml
gesture:
  enabled: true
  max_num_hands: 2
  min_detection_confidence: 0.6
  min_tracking_confidence: 0.5
  stable_frame_count: 5
  cooldown_seconds: 2.0

gesture_actions:
  open_palm: "STOP"
  thumbs_up: "START_FOLLOW"
  fist: "PAUSE"
  point_left: "TURN_LEFT"
  point_right: "TURN_RIGHT"
  victory: "PATROL_READY"
  ok_sign: "RESUME"

control_modes:
  default_mode: "IDLE"
  follow_mode_enabled: true
  emergency_stop_enabled: true
```

## 安全注意事项

- `open_palm` 永远优先停止机器人。
- 图像超过 1 秒未更新，机器人必须停止。
- YOLO 或 MediaPipe 异常时机器人必须停止。
- Flask Web 异常不会影响 ROS2 控制线程。
- Ctrl+C 退出时会发布零速度 `/cmd_vel`。
- YOLO11 不是避障传感器，普通 RGB 摄像头无法可靠判断墙、桌子、柜子的真实距离。
- 第一次运行请架空底盘，或把 S1 放在开阔区域。
- 不要在拥挤、靠墙、靠桌边环境直接开启 FOLLOW。

## 常见问题

### ModuleNotFoundError: No module named 'ultralytics'

当前启动 ROS2 节点的 Python 环境没有安装 YOLO 依赖：

```bash
cd ~/rm_ws
python3 -m pip install -r src/person_follower/requirements.txt
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```

如果你以 root 运行：

```bash
sudo -H python3 -m pip install -r /root/rm_ws/src/person_follower/requirements.txt
sudo -H python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```

### ModuleNotFoundError: No module named 'mediapipe'

安装 MediaPipe：

```bash
python3 -m pip install mediapipe==0.10.11
```

### YOLO 模型下载失败

手动下载 `yolo11n.pt`，放到运行目录，或把 `config.yaml` 中的 `yolo_model` 改成绝对路径。

### Dashboard 打不开

检查端口：

```bash
ss -lntp | grep 8088
```

远程访问时确认防火墙允许 8088。

### 机器人不动

检查图像和速度 Topic：

```bash
ros2 topic hz /camera/image_color
ros2 topic echo /cmd_vel
```

如果 `/cmd_vel` 有输出但机器人不动，检查 `robomaster_ros` 是否正常连接 S1。

### 手势不触发

先观察 Dashboard 的候选手势和稳定手势。如果候选手势跳动，说明识别不稳定，可以调高光照、让手靠近画面中心，或降低：

```yaml
gesture:
  stable_frame_count: 4
```

## 推荐测试顺序

```bash
ros2 run robomaster_ros discover
ros2 launch robomaster_ros s1.launch conn_type:=sta
ros2 topic hz /camera/image_color
cd ~/rm_ws
python3 -m pip install -r src/person_follower/requirements.txt
colcon build --packages-select person_follower
source install/setup.bash
ros2 launch person_follower follower.launch.py
```
