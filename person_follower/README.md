# RoboMaster S1 ROS2 YOLO11 + 手势 + Vision-Language-Agent

这是 RoboMaster S1 智能控制 Demo 的第三阶段工程，运行环境面向 Ubuntu 20.04 + ROS2 Foxy + `robomaster_ros`。

系统能力：

```text
摄像头
  -> YOLO11 人体检测
  -> MediaPipe Hands 手势识别
  -> Qwen3-VL 场景描述
  -> Llama Server 行为规划
  -> FSM 状态机
  -> SafetyGuard 安全过滤
  -> /cmd_vel 控制 RoboMaster S1
```

重要原则：

```text
LLM/VLM 只给建议
Action Plan 必须经过白名单和 SafetyGuard
模型不能直接控制底盘速度
```

## 当前阶段

第一阶段：

- YOLO11 人体检测；
- 最大 person 目标选择；
- 人体跟随；
- Web Dashboard。

第二阶段：

- MediaPipe Hands；
- open_palm / thumbs_up / fist / point_left / point_right / victory / ok_sign；
- IDLE / FOLLOW / GESTURE_CONTROL / PATROL_READY / EMERGENCY_STOP。

第三阶段：

- Qwen3-VL 视觉语言描述；
- Llama Server 行为规划；
- Agent 面板；
- AGENT_MODE；
- 支持自然语言任务输入；
- 支持目标搜索、场景描述、跟随意图、危险停止。

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
    ├── scene_understanding.py
    ├── vision_agent.py
    ├── planner_agent.py
    ├── robot_executor.py
    ├── web_dashboard.py
    ├── flask_dashboard.py
    ├── follower_control.py
    ├── safety.py
    └── utils.py
```

`person_follower_node.py` 和 `flask_dashboard.py` 是兼容入口，新的主实现分别在 `follower_node.py` 和 `web_dashboard.py`。

## 环境说明

已验证 ROS Topic：

```text
/camera/image_color
/cmd_vel
/odom
/imu
/battery
```

先确认 RoboMaster ROS 驱动正常：

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

安装 Python 依赖：

```bash
cd ~/rm_ws
python3 -m pip install --upgrade pip
python3 -m pip install -r src/person_follower/requirements.txt
```

国内网络慢可以使用清华源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r src/person_follower/requirements.txt
```

如果你以 `root` 用户运行 ROS2，依赖也必须安装到 `root` 用户的 Python 环境。

依赖检查：

```bash
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
python3 -c "import mediapipe as mp; print(mp.__version__)"
python3 -c "import requests; print('requests ok')"
```

说明：Ubuntu 20.04 + Python3.8 下，`requirements.txt` 固定使用 `mediapipe==0.10.11`。

## 构建

```bash
cd ~/rm_ws
colcon build --packages-select person_follower
source install/setup.bash
```

## 启动

先启动 RoboMaster ROS 驱动：

```bash
ros2 launch robomaster_ros s1.launch conn_type:=sta
```

再启动第三阶段程序：

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

## 模型服务配置

配置文件：

```text
config/config.yaml
```

Llama Server 和 Qwen3-VL 均使用 OpenAI 兼容接口：

```yaml
llm:
  enabled: true
  base_url: "http://10.10.10.156:8080/v1"
  model: "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
  timeout_sec: 8.0

vlm:
  enabled: true
  base_url: "http://10.10.10.156:8080/v1"
  model: "qwen3-vl"
  timeout_sec: 8.0

agent:
  enabled: true
  max_history: 20
  planning_interval: 2.0
  forward_speed: 0.12
  turn_speed: 0.45
  action_duration_sec: 0.8
```

接口路径由代码拼接为：

```text
<base_url>/chat/completions
```

如果你的服务地址是：

```text
http://10.10.10.156:8080/v1
```

最终请求地址就是：

```text
http://10.10.10.156:8080/v1/chat/completions
```

## 状态机

系统模式：

| 模式 | 说明 |
| --- | --- |
| IDLE | 静止，不跟随，不执行 Agent 动作 |
| FOLLOW | 使用 YOLO11 人体跟随 |
| GESTURE_CONTROL | 执行一次性手势动作 |
| PATROL_READY | 巡逻预留模式 |
| AGENT_MODE | Vision-Language-Agent 模式 |
| EMERGENCY_STOP | 紧急停止，最高优先级 |

手势优先级高于 Agent。`open_palm` 任何时候都会立即 stop。

## Agent 动作白名单

PlannerAgent 只能输出以下 action：

```text
STOP
FORWARD
BACKWARD
TURN_LEFT
TURN_RIGHT
FOLLOW_PERSON
SEARCH_TARGET
SPEAK
PATROL
```

映射关系：

| Agent action | 执行方式 |
| --- | --- |
| STOP | 发布零速度，保持安全停止 |
| FOLLOW_PERSON | 切换到 FOLLOW，复用 YOLO 人体跟随 |
| SEARCH_TARGET | 原地低速扫描 |
| TURN_LEFT | 原地左转短动作 |
| TURN_RIGHT | 原地右转短动作 |
| FORWARD | 短时低速前进 |
| BACKWARD | 短时低速后退 |
| SPEAK | 当前只记录到 Dashboard，不调用音频 |
| PATROL | 进入 PATROL_READY，不做导航 |

## Dashboard 使用

页面包含：

- 实时视频；
- YOLO 检测框；
- MediaPipe 手部关键点；
- 当前模式；
- 当前手势；
- 当前安全速度；
- Agent 场景描述；
- Agent 行动计划；
- Agent 最近 20 条日志；
- Token 使用量；
- 模型响应时间；
- 手动停止、启动跟随、暂停跟随、Agent 模式按钮；
- 自然语言任务输入框。

示例输入：

```text
你看到了什么？
帮我找水杯
跟着我
停止
```

## 典型能力

环境描述：

```text
用户输入：你看到了什么？
VLM 输出：前方有一个人，旁边有一张椅子。
Planner 输出：SPEAK 或 STOP
```

目标搜索：

```text
用户输入：帮我找水杯
Agent 输出：SEARCH_TARGET
机器人：原地低速扫描，发现目标后 STOP
```

目标跟随：

```text
用户输入：跟着我
Agent 输出：FOLLOW_PERSON
机器人：进入 FOLLOW 模式
```

手势与语言混合控制：

```text
thumbs_up -> FOLLOW
自然语言“停止” -> STOP
open_palm -> 立即 stop
```

危险检测：

```text
识别到跌倒、障碍物、遮挡、目标丢失、模型超时 -> STOP
```

## ROS2 Topic

手势状态：

```bash
ros2 topic echo /gesture/state
```

手势命令：

```bash
ros2 topic echo /gesture/command
```

可选调试图像：

```yaml
gesture:
  publish_debug_image: true
```

开启后发布：

```text
/gesture/debug_image
```

## 安全注意事项

- EMERGENCY_STOP 最高优先级。
- 图像超过 1 秒未更新，机器人停止。
- Agent 响应超时，机器人停止。
- LLM/VLM 异常，机器人停止。
- 目标丢失，机器人停止。
- LLM 不能直接控制底盘速度。
- Action Plan 必须经过白名单和 SafetyGuard。
- YOLO11 和 VLM 都不是可靠避障传感器，普通 RGB 摄像头无法准确判断墙、桌子、柜子的距离。
- 第一次运行请架空底盘，或放在开阔区域低速测试。

## 常见问题

### No module named 'ultralytics'

```bash
cd ~/rm_ws
python3 -m pip install -r src/person_follower/requirements.txt
```

### No module named 'mediapipe'

```bash
python3 -m pip install mediapipe==0.10.11
```

### Agent 没有响应

检查 Llama Server：

```bash
curl http://10.10.10.156:8080/v1/models
```

确认 `config.yaml` 里的 `base_url` 和 `model` 名称正确。

### Dashboard 打不开

```bash
ss -lntp | grep 8088
```

远程访问时确认防火墙允许 8088。

### 机器人不动

```bash
ros2 topic hz /camera/image_color
ros2 topic echo /cmd_vel
```

如果 `/cmd_vel` 有输出但机器人不动，检查 `robomaster_ros` 是否正常连接 S1。

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

## 后续扩展方向

第四阶段：SLAM

第五阶段：Nav2

第六阶段：自主巡逻

第七阶段：多机器人协同
