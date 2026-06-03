# RoboMaster S1 + YOLO11 + 本地 llama.cpp 模型智能控制 Demo

这个工程演示如何用 PC 端 RoboMaster Python SDK 连接 RoboMaster S1，读取摄像头视频流，用 Ultralytics YOLO11 做实时目标检测，并把结构化场景描述发送给本地 llama.cpp OpenAI 兼容接口，让模型返回安全动作建议。

所有机器人动作都限制在白名单内，速度从 `config.py` 读取。模型不能直接指定任意速度。

## 架构说明

```text
RoboMaster S1 Camera
        |
        v
RoboMaster Python SDK video stream
        |
        v
OpenCV frame -> YOLO11 detection -> scene_text
        |
        v
llama.cpp /v1/chat/completions
        |
        v
JSON {"action": "...", "reason": "..."}
        |
        v
safe action whitelist -> gimbal/chassis low-speed command
```

## 环境准备

1. RoboMaster S1 已经开启 PC SDK 支持，并且 PC 能连接到机器人。
2. 本地 llama.cpp server 已启动：

```text
http://10.10.10.156:8080
```

3. llama.cpp OpenAI 兼容接口可访问：

```text
http://10.10.10.156:8080/v1/chat/completions
```

4. 第一次运行 YOLO11 时，`yolo11n.pt` 可能需要下载。

## 安装依赖

建议在项目根目录创建虚拟环境后安装：

```powershell
cd D:\codex\dji-s1\RoboMaster-S1-YOLO11-llama-demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你继续使用上一级已有环境，可以在上一级环境中安装：

```powershell
cd D:\codex\dji-s1\RoboMaster-S1-YOLO11-llama-demo
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置

所有主要参数在 [config.py](config.py)：

```python
ROBOT_CONN_TYPE = "sta"
LLM_BASE_URL = "http://10.10.10.156:8080"
LLM_MODEL = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
YOLO_MODEL = "yolo11n.pt"
YOLO_CONF = 0.45
YOLO_IMGSZ = 640
```

## 测试本地模型接口

```powershell
python test_llm_api.py
```

成功时会打印模型返回内容。失败时会显示明确的 HTTP 或 JSON 错误。

## 测试 S1 摄像头和 YOLO

```powershell
python test_s1_camera_yolo.py
```

功能：

- 连接 S1；
- 启动摄像头；
- YOLO11 实时检测；
- OpenCV 显示检测框；
- 按 `q` 退出。

退出时会停止视频流、关闭机器人、释放窗口。

## 运行云台跟随 Demo

```powershell
python s1_yolo_gimbal_follow.py
```

功能：

- 检测置信度最高的 `person`；
- 根据目标中心与画面中心偏移，低速控制云台；
- 没检测到 `person` 时云台停止；
- 按 `q` 退出。

云台控制死区：

- x 方向小于 40 像素不转 yaw；
- y 方向小于 35 像素不转 pitch；
- 控制间隔约 0.15 秒。

## 运行完整 LLM Agent Demo

```powershell
python s1_yolo_llm_agent.py
```

工作流程：

1. 摄像头读取视频帧；
2. YOLO11 检测 `person`；
3. 构造结构化 `scene_text`；
4. 每隔 `LLM_INTERVAL_SECONDS` 秒请求 llama.cpp；
5. 解析模型 JSON；
6. 执行安全动作；
7. 画面叠加显示 `action` 和 `reason`；
8. 按 `q` 退出。

动作白名单：

```text
stop
gimbal_left
gimbal_right
gimbal_up
gimbal_down
turn_left
turn_right
forward
backward
```

## 常见问题排查

### RoboMaster SDK 连接失败

先确认机器人 SDK 服务已经开放。可以回到上级目录运行：

```powershell
cd D:\codex\dji-s1
python diagnose_sdk_connectivity.py --robot-ip 10.10.10.152 --local-ip 10.10.10.156 --sn 159CG7300406ZU --broadcast-timeout 5 --timeout 2 --sdk-handshake
```

如果 `40923`、`20020` 不可达，说明 PC SDK 还没有真正开放。

### conn_type="sta" 和 conn_type="ap" 如何选择

- `sta`：机器人和电脑连接到同一个路由器，适合你当前 `10.10.10.x` 网络。
- `ap`：电脑直接连接机器人热点，通常机器人 IP 是 `192.168.2.1`。

当前配置默认：

```python
ROBOT_CONN_TYPE = "sta"
```

### llama.cpp 接口无法访问

确认服务在本机地址启动：

```text
http://10.10.10.156:8080
```

浏览器或 curl 访问：

```text
http://10.10.10.156:8080/v1/models
```

然后运行：

```powershell
python test_llm_api.py
```

### YOLO 模型下载失败

`YOLO_MODEL = "yolo11n.pt"` 第一次运行可能需要联网下载。网络差时可以手动把 `yolo11n.pt` 放到工程目录，或改成已有权重路径。

### OpenCV 窗口打不开

确认不是在无 GUI 的终端环境中运行。Windows 桌面环境一般可以正常打开窗口。

### 机器人不动

可能原因：

- 未检测到 `person`；
- LLM 返回 `stop`；
- SDK 连接失败；
- 当前安全速度很低，动作不明显；
- 云台/底盘模块未初始化成功。

先运行 `test_s1_camera_yolo.py`，再运行 `s1_yolo_gimbal_follow.py`。

## 安全注意事项

- 第一次运行时架空底盘或放在开阔区域。
- 保持低速。
- 随时按 `q` 退出。
- 异常时程序会尽量执行 `stop`。
- 不要加入发射器、射击、水弹、红外攻击等功能。

## 目录结构

```text
RoboMaster-S1-YOLO11-llama-demo/
├── README.md
├── requirements.txt
├── config.py
├── test_llm_api.py
├── test_s1_camera_yolo.py
├── s1_yolo_gimbal_follow.py
└── s1_yolo_llm_agent.py
```
