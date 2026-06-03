# RoboMaster S1 + YOLO11 + 本地 llama.cpp 智能控制 Demo

这个工程演示如何使用 PC 端 RoboMaster Python SDK 连接 RoboMaster S1，读取机器人摄像头实时视频流，使用 Ultralytics YOLO11 检测目标，并把检测结果发送给本地 llama.cpp OpenAI 兼容接口，让模型返回安全动作建议。

动作执行只允许白名单 action，速度全部在 `config.py` 中固定配置。模型不能直接指定任意速度，也不会控制发射器、射击、水弹或红外攻击。

## 当前默认配置

主要配置在 [config.py](config.py)：

```python
ROBOT_CONN_TYPE = "sta"

LLM_BASE_URL = "http://10.10.10.156:8080"
LLM_MODEL = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"

YOLO_MODEL = "yolo11n.pt"
YOLO_CONF = 0.45
YOLO_IMGSZ = 640
```

## 运行前必须先安装依赖

如果你直接运行：

```powershell
python test_llm_api.py
```

并看到：

```text
ModuleNotFoundError: No module named 'requests'
```

说明当前 Python 环境没有安装依赖。请先执行下面的安装步骤。

### 方案 A：使用上一级已验证 `.venv`

如果你已经在 `D:\codex\dji-s1\.venv` 里装好了 RoboMaster SDK、OpenCV、YOLO 等依赖，推荐直接使用这个环境运行。

进入项目目录：

```powershell
cd D:\codex\dji-s1\RoboMaster-S1-YOLO11-llama-demo
```

运行 LLM 接口测试：

```powershell
..\.venv\Scripts\python.exe .\test_llm_api.py
```

运行 S1 摄像头 + YOLO 测试：

```powershell
..\.venv\Scripts\python.exe .\test_s1_camera_yolo.py
```

运行云台跟随：

```powershell
..\.venv\Scripts\python.exe .\s1_yolo_gimbal_follow.py
```

运行完整 LLM Agent：

```powershell
..\.venv\Scripts\python.exe .\s1_yolo_llm_agent.py
```

如果缺少依赖，请安装到上一级 `.venv`：

```powershell
..\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 方案 B：使用当前全局 Python 3.8

你当前机器上的 `python` 是：

```text
C:\Users\LGP\AppData\Local\Programs\Python\Python38\python.exe
```

进入项目目录后安装依赖：

```powershell
cd D:\codex\dji-s1\RoboMaster-S1-YOLO11-llama-demo
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果只想先测试本地 LLM 接口，至少需要：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests
```

### 方案 C：使用项目内 `.venv`

如果你希望依赖隔离在当前工程里：

```powershell
cd D:\codex\dji-s1\RoboMaster-S1-YOLO11-llama-demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

之后运行脚本也要使用 `.venv` 里的 Python：

```powershell
.\.venv\Scripts\python.exe test_llm_api.py
```

不要混用环境：如果依赖装在 `.venv`，就用 `.venv` 运行；如果用 `python` 运行，就把依赖装到 `python` 对应的环境。

## 测试 1：本地 llama.cpp 接口

先确认模型服务列表可访问：

```powershell
Invoke-RestMethod -Uri "http://10.10.10.156:8080/v1/models" -Method Get
```

然后测试 chat completions：

```powershell
python test_llm_api.py
```

如果 `/v1/models` 正常，但 `test_llm_api.py` 超时，说明 llama.cpp 服务可访问，但模型生成速度超过了脚本当前 timeout。可以先确认 llama.cpp server 控制台是否正在加载模型或生成文本。

## 测试 2：S1 摄像头 + YOLO11

```powershell
python test_s1_camera_yolo.py
```

功能：

- 初始化 RoboMaster S1；
- 启动摄像头视频流；
- 加载 `yolo11n.pt`；
- 实时检测并显示画面；
- 按 `q` 退出。

第一次运行 YOLO11 可能会下载 `yolo11n.pt`。如果下载失败，可以手动下载权重放到工程目录，或者把 `config.py` 里的 `YOLO_MODEL` 改成已有权重路径。

## 测试 3：YOLO 云台跟随

```powershell
python s1_yolo_gimbal_follow.py
```

功能：

- 只检测 `person`；
- 选择置信度最高的 person；
- 根据目标中心点控制云台低速转动；
- 没有 person 时云台停止；
- 按 `q` 退出。

## 测试 4：完整 LLM Agent

```powershell
python s1_yolo_llm_agent.py
```

工作流程：

1. 读取 S1 摄像头画面；
2. YOLO11 检测 person；
3. 构造结构化 `scene_text`；
4. 每隔 `LLM_INTERVAL_SECONDS` 秒请求 llama.cpp；
5. 解析模型返回 JSON；
6. 执行安全白名单动作；
7. 在画面上显示 action 和 reason；
8. 按 `q` 退出。

允许的 action：

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

## 常见问题

### 1. `ModuleNotFoundError: No module named 'requests'`

当前 Python 没有安装依赖。执行：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests
```

或安装完整依赖：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. RoboMaster SDK 连接失败

回到上级目录运行诊断：

```powershell
cd D:\codex\dji-s1
python diagnose_sdk_connectivity.py --robot-ip 10.10.10.152 --local-ip 10.10.10.156 --sn 159CG7300406ZU --broadcast-timeout 5 --timeout 2 --sdk-handshake
```

如果 `40923`、`20020` 不可达，说明 S1 的 PC SDK 服务还没有打开。

### 3. `conn_type="sta"` 和 `conn_type="ap"` 怎么选

- `sta`：机器人和电脑在同一个路由器或局域网内，适合当前 `10.10.10.x` 网络。
- `ap`：电脑直接连接机器人热点，通常机器人 IP 是 `192.168.2.1`。

当前默认：

```python
ROBOT_CONN_TYPE = "sta"
```

### 4. llama.cpp 接口超时

先测：

```powershell
Invoke-RestMethod -Uri "http://10.10.10.156:8080/v1/models" -Method Get
```

如果这个能返回，但 `test_llm_api.py` 超时，说明 server 在，但模型生成太慢或正在加载。可以检查 llama.cpp server 控制台日志。

### 5. OpenCV 窗口打不开

需要在 Windows 桌面环境运行。远程无 GUI 终端可能无法显示 `cv2.imshow` 窗口。

### 6. 机器人不动

可能原因：

- 没检测到 person；
- LLM 返回 `stop`；
- SDK 连接失败；
- 安全速度设置很低；
- 云台或底盘模块没有初始化成功。

建议先按顺序运行：

```powershell
python test_llm_api.py
python test_s1_camera_yolo.py
python s1_yolo_gimbal_follow.py
python s1_yolo_llm_agent.py
```

## 安全注意事项

- 第一次运行时架空底盘，或放在开阔区域。
- 保持低速。
- 随时按 `q` 退出。
- 异常退出时程序会尽量执行 stop。
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
