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

运行完整手势 + LLM Dashboard Agent：

```powershell
..\.venv\Scripts\python.exe .\s1_yolo_llm_agent.py
```

运行 Web 调试页面：

```powershell
..\.venv\Scripts\python.exe .\s1_web_debug.py
```

然后在浏览器打开：

```text
http://127.0.0.1:5000
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
- 云台速度按低速策略执行，当前配置已将云台控制进一步降低 50%，转向会更柔和；
- 没有 person 时云台停止；
- 按 `q` 退出。

## 测试 4：完整手势 + LLM Dashboard Agent

```powershell
python s1_yolo_llm_agent.py
```

工作流程：

1. 读取 S1 摄像头画面；
2. YOLO11 检测 `person`；
3. MediaPipe Hands 检测手部关键点和标准手势；
4. 构造包含 `detected_gesture` 的结构化 `scene_text`；
5. 每隔 `LLM_INTERVAL_SECONDS` 秒请求 llama.cpp；
6. 解析模型返回 JSON；
7. 手势映射和 LLM 建议都必须经过 `SafetyGuard`；
8. 执行过滤后的安全白名单动作；
9. Dashboard 显示视频、YOLO 框、手势关键点、LLM 决策和安全状态；
10. 按 `q` 退出。

手势映射默认保守：

- `open_palm` / `fist`：建议 `stop`；
- `point_left` / `point_right`：建议云台向左/向右；
- `peace` / `thumbs_up` / `thumbs_down`：建议云台上/下调整；
- 默认不把手势直接映射到底盘前进或后退。

### Safety Guard 避障与防撞说明

YOLO11 不是避障传感器，普通摄像头也无法可靠判断墙、桌子、柜子等物体的真实距离。因此完整 LLM Agent 默认禁止模型直接控制底盘前进：

```python
AUTO_MOVE_ENABLED = False
MAX_FORWARD_DURATION = 0.3
FORWARD_COOLDOWN = 0.8
MAX_PERSON_AREA_RATIO_FOR_FORWARD = 0.12
MIN_PERSON_AREA_RATIO_FOR_FORWARD = 0.02
ALLOW_TURN_IN_PLACE = True
EMERGENCY_STOP = False
```

这意味着：

- LLM 只能给动作建议，最终动作由 `SafetyGuard` 决定。
- 默认情况下，`forward` 和 `backward` 会被过滤成 `stop`。
- 云台调整和原地转向仍可用于观察和对准目标。
- 如果你确实要启用自动前进，必须把 `AUTO_MOVE_ENABLED=True`，并且只在开阔区域测试。
- 第一次运行必须架空底盘，或放在空旷、无障碍区域。

即使开启 `AUTO_MOVE_ENABLED=True`，Safety Guard 也会限制单次前进时长、前进冷却时间和 person 目标框面积范围，避免模型连续输出 `forward` 后让 S1 持续撞墙。

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

## 测试 5：Dashboard 控制台

```powershell
python s1_yolo_llm_agent.py
```

`s1_yolo_llm_agent.py` 会打开窗口 `RoboMaster S1 AI Control Dashboard`。Dashboard 用于把摄像头、YOLO、LLM 建议和安全过滤结果放在同一个窗口里观察。界面按四区组织：

- 视频区：显示 S1 摄像头画面、YOLO 检测框、目标中心点和关键状态叠加。
- 感知区：显示当前检测到的 `person`、置信度、目标框面积占比和结构化 `scene_text`。
- 决策区：显示 LLM 返回的 `raw_action`、`SafetyGuard` 过滤后的 `safe_action`、`safety_reason` 和最近一次请求状态。
- 运行区：显示连接状态、自动移动开关、最近动作、错误信息和运行日志，方便排查机器人为什么停下或不动。

键盘快捷键：

- 按 `q`：安全退出 Dashboard，停止底盘和云台，并释放摄像头与机器人连接。
- 按 `s`：立即发送 `stop`，用于临时急停或在调试时打断当前动作；窗口保持运行，方便继续观察状态。
- 按 `m`：运行时切换 `AUTO_MOVE_ENABLED`。
- 按 `r`：云台回中。

Dashboard 仍遵守 `config.py` 中的安全配置。默认 `AUTO_MOVE_ENABLED = False`，因此 LLM 即使返回 `forward` 或 `backward`，也会被 `SafetyGuard` 过滤为 `stop`。需要自动前进时必须显式改为 `AUTO_MOVE_ENABLED=True`，并先架空底盘或在开阔区域低速测试。

## 测试 6：Web 调试页面

```powershell
..\.venv\Scripts\python.exe .\s1_web_debug.py
```

打开：

```text
http://127.0.0.1:5000
```

页面功能：

- 启动和释放 RoboMaster S1 连接；
- 网页中预览摄像头视频；
- 视频画面叠加 YOLO 检测框；
- 显示最近检测到的类别、置信度、中心点和面积占比；
- 提供手柄式控制区：左侧方向键控制底盘低速前进、后退和原地转向，右侧方向键控制云台；
- 只能锁定 YOLO 类别为 `person` 的目标，将当前识别到的 `person` 拍照并保存到 `media_locked_people/`；
- 人物锁定区域会显示锁定目标状态、`crop_path`，并通过 `/locked_target_image?t=...` 展示后端保存的人物裁剪缩略图；
- 锁定一个人物后，用颜色直方图在后续 YOLO person 框中匹配目标；
- 目标丢失时，页面会展示 `status.last_action` 和 `target.search_mode` 对应的 target/search 状态，例如云台搜索或底盘搜索；
- 自动跟随锁定人物，仍使用低速安全策略：用柔和云台调整方向，并用低速底盘前后移动近似保持 1-2 米距离；
- 不使用发射器。

Web 服务使用 Python 标准库实现，不需要额外安装 Flask。

网页按钮说明：

- `启动`：连接 S1，启动摄像头视频流，加载 YOLO。
- `锁定并跟随`：只会选择当前置信度最高的 `person`，保存整帧图和人物裁剪图，并开启自动跟随。
- `暂停跟随`：保留锁定目标，但停止底盘和云台动作。
- `继续跟随`：继续跟随已经锁定的人物。
- 左侧底盘方向键：按住低速移动或原地转向，松开自动停止底盘；手柄手动控制会暂停自动跟随。
- 云台方向按钮：使用更柔和的半速云台控制，便于近距离调试和观察；手柄手动控制会暂停自动跟随。
- `解锁停止`：清除锁定目标，并停止底盘和云台。
- `停止并释放`：停止机器人、视频流和 Web 侧连接资源。

距离控制说明：

S1 摄像头没有直接提供人物深度，本 Demo 使用目标框面积占比近似估计距离。默认阈值在 `config.py`：

- `TARGET_FAR_AREA_RATIO = 0.055`：小于该值时认为较远，低速前进。
- `TARGET_NEAR_AREA_RATIO = 0.18`：大于该值时认为较近，低速后退。

不同房间、镜头角度、人物身高会影响面积占比，建议先架空底盘观察网页里的 `area_ratio`，再微调这两个阈值。

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
- `AUTO_MOVE_ENABLED = False`，`forward` / `backward` 被 Safety Guard 过滤成 `stop`；
- SDK 连接失败；
- 安全速度设置很低；
- 云台或底盘模块没有初始化成功。

建议先按顺序运行：

```powershell
python test_llm_api.py
python test_s1_camera_yolo.py
python s1_yolo_gimbal_follow.py
python s1_yolo_llm_agent.py
..\.venv\Scripts\python.exe .\s1_web_debug.py
```

### 7. Dashboard 里 LLM 显示 `forward`，但机器人没有前进

这是默认安全行为。完整 Agent 和 Dashboard 默认禁用自动底盘前后移动：

```python
AUTO_MOVE_ENABLED = False
```

此时 `SafetyGuard` 会把 `forward` 和 `backward` 过滤成 `stop`，Dashboard 的决策区会显示原始建议、最终安全动作和过滤原因。只有在明确确认测试区域安全后，才建议把它改成 `True`。

### 8. Dashboard 按键没有反应

先确认 Dashboard 窗口是当前焦点窗口。常用按键：

- `q`：安全退出。
- `s`：立即停止当前动作，但不关闭窗口。

如果按 `q` 后窗口关闭较慢，通常是程序正在释放视频流、SDK 连接或等待最后一次 stop 命令返回。

## 安全注意事项

- 第一次运行时架空底盘，或放在开阔区域。
- 保持低速；当前云台控制会进一步降低 50%，自动跟随仍按低速安全策略执行。
- 随时按 `q` 退出。
- Dashboard 中可随时按 `s` 发送 stop。
- 异常退出时程序会尽量执行 stop。
- 不要加入发射器、射击、水弹、红外攻击等功能。
- Web 人物跟随第一次测试时，先让 S1 离人 1-2 米，并确保前后没有障碍物。

## 目录结构

```text
RoboMaster-S1-YOLO11-llama-demo/
├── README.md
├── requirements.txt
├── config.py
├── test_llm_api.py
├── test_s1_camera_yolo.py
├── s1_yolo_gimbal_follow.py
├── s1_yolo_llm_agent.py
├── gesture_detector.py
├── gesture_action_mapper.py
├── dashboard.py
├── s1_web_debug.py
├── media_locked_people/       # 运行时保存锁定人物截图，默认不提交 Git
└── templates/
    └── index.html
```
