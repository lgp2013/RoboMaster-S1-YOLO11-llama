# RoboMaster S1 ROS2 YOLO11 人体跟随系统

这个工程基于 Ubuntu 20.04 + ROS2 Foxy + `robomaster_ros`，实现：

```text
/camera/image_color -> YOLO11 person 检测 -> 选择最大 person -> /cmd_vel 自动跟随
```

同时提供 Flask Dashboard：

```text
http://0.0.0.0:8088
```

Dashboard 可以查看实时视频、YOLO 检测框、FPS、当前目标、速度命令和安全状态。

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
    ├── person_follower_node.py
    ├── yolo_detector.py
    ├── follower_control.py
    └── flask_dashboard.py
```

## 前置条件

你已经验证以下命令和 Topic 正常：

```bash
ros2 run robomaster_ros discover
ros2 launch robomaster_ros s1.launch conn_type:=sta
ros2 topic echo /battery
ros2 topic echo /odom
ros2 topic hz /camera/image_color
```

机器人 IP 示例：`10.10.10.152`

## 安装依赖

在 ROS2 工作空间中放置本包，例如：

```bash
mkdir -p ~/rm_ws/src
cp -r person_follower ~/rm_ws/src/
cd ~/rm_ws/src/person_follower
python3 -m pip install -r requirements.txt
```

如果你的系统还没有 `cv_bridge`：

```bash
sudo apt update
sudo apt install ros-foxy-cv-bridge
```

构建：

```bash
cd ~/rm_ws
colcon build --packages-select person_follower
source install/setup.bash
```

## 启动方式

先启动 RoboMaster ROS 驱动：

```bash
ros2 launch robomaster_ros s1.launch conn_type:=sta
```

再启动人体跟随：

```bash
ros2 launch person_follower follower.launch.py
```

打开 Dashboard：

```text
http://<你的电脑IP>:8088
```

如果在本机浏览器打开：

```text
http://127.0.0.1:8088
```

## 配置文件

配置在：

```text
config/config.yaml
```

关键参数：

```yaml
camera_topic: "/camera/image_color"
cmd_vel_topic: "/cmd_vel"
yolo_model: "yolo11n.pt"
confidence_threshold: 0.45
max_linear_speed: 0.30
max_angular_speed: 1.0
target_bbox_height_ratio: 0.45
bbox_height_tolerance: 0.08
dashboard_host: "0.0.0.0"
dashboard_port: 8088
```

## 跟踪逻辑

1. YOLO11 只保留 `person` 类别；
2. 如果画面中有多人，选择检测框面积最大的 person；
3. 根据目标中心点和画面中心点计算横向误差；
4. 目标在右侧时机器人向右转，目标在左侧时向左转；
5. 根据检测框高度占画面高度比例估算距离；
6. 框太小表示人较远，机器人低速前进；
7. 框太大表示人较近，机器人低速后退；
8. 框大小合适时停止前后运动；
9. 目标丢失时立即发布 `cmd_vel=0`。

## 速度与安全

默认限制：

```text
线速度: -0.3 ~ 0.3 m/s
角速度: -1.0 ~ 1.0 rad/s
```

SafetyGuard 会做最终限幅：

- 没有目标时 stop；
- 图像超时或目标丢失时 stop；
- 禁止横移和无关轴速度；
- 所有速度限制在 `config.yaml` 范围内。

重要说明：

YOLO11 不是真正的避障传感器，单目摄像头不能可靠判断墙、桌子、柜子的真实距离。这个 Demo 通过低速、目标丢失停机、检测框高度估距来降低风险。第一次运行请架空底盘，或放在开阔区域。

## Dashboard 功能

Dashboard 显示：

- 实时视频；
- YOLO 检测框；
- FPS；
- 当前跟踪目标；
- 当前速度命令；
- 安全过滤后的速度；
- 目标距离估算；
- 控制原因和安全原因。

## 常见问题

### 1. YOLO 模型下载失败

手动下载 `yolo11n.pt`，放到运行目录，或把 `config.yaml` 里的 `yolo_model` 改成绝对路径。

### 2. Dashboard 打不开

确认节点已启动，并检查端口：

```bash
ss -lntp | grep 8088
```

如果远程访问，确认防火墙允许 `8088`。

### 3. 机器人不动

检查：

```bash
ros2 topic hz /camera/image_color
ros2 topic echo /cmd_vel
```

如果 `/cmd_vel` 有输出但机器人不动，检查 `robomaster_ros` 驱动是否正常连接。

### 4. 机器人方向反了

ROS 中 `angular.z > 0` 通常表示左转。本工程已按这个约定实现。如果你的底盘方向相反，可以在 `follower_control.py` 中调整角速度符号。

### 5. 距离不准

本工程用检测框高度估算距离，不是真实深度。请调整：

```yaml
target_bbox_height_ratio: 0.45
bbox_height_tolerance: 0.08
```

## 测试顺序

```bash
ros2 run robomaster_ros discover
ros2 launch robomaster_ros s1.launch conn_type:=sta
ros2 topic hz /camera/image_color
ros2 launch person_follower follower.launch.py
```
