# AGENTS.md — AutoSEM Processor 的安装与运行指令

这个文件是给"替别人把这套东西装起来"的 AI agent / 自动化脚本用的：命令照抄即可，
每步都有可判定的成功条件。人看的使用说明在 [README.md](README.md)。

## 0. 硬约束（不要违反）

- **不要**引入 pytesseract / Tesseract / 任何系统级 OCR。比例尺数值只走 OpenAI 格式的视觉模型。
- **不要**引入 SQLite / 数据库 / "实验组"之类的概念。这个应用是单页无状态的。
- opencv 必须是 **`opencv-python-headless`**，不要装 `opencv-python`（后者在裸 Linux / 容器里会因缺 `libGL.so.1` 而 import 失败）。
- **不要**把 `.env` / `.env.local` 提交到 git；配置落到那里，`.gitignore` 已经排除。
- **不要**硬编码 Windows 路径（字体、`os.startfile` 等）。字体探测已经是跨平台的。
- 不要改 `core/sem.py` 里的检测阈值和注释里记的踩坑结论 —— 那些是用真实电镜图回归出来的。

## 1. 环境自检（先跑，把输出贴进结论）

```bash
python --version      # 需要 >= 3.10
python -c "import sys; print(sys.executable)"    # 确认不是 Windows Store 的占位符
```

## 2. 安装（非交互，可直接复制执行）

优先用仓库自带脚本（幂等，可重复跑）：

```bash
install.bat          # Windows
bash install.sh      # Linux / macOS
```

等价的纯命令：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python     -m pip install -r requirements.txt   # Linux / macOS
# 若超时 / 403，追加国内镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple
```

成功判据：命令退出码 0，且 `.venv` 目录存在。

## 3. 冒烟测试（不启服务，30 秒内应通过）

```bash
# 依赖是否齐（Windows 用 .venv/Scripts/python，下同）
.venv/bin/python -c "import cv2, numpy, PIL, httpx, nicegui; print('deps OK')"

# 算法链路是否可用（应打印两行形状）
.venv/bin/python -c "
import numpy as np
from core.sem import compose_image, compose_grid
img = np.zeros((300, 400, 3), np.uint8)
out = compose_image(img, 80, 2.0, 'μm', label_tl='(a)')
print('compose_image OK', out.shape)
"

# 配置读取是否正常（应打印一行 dict）
.venv/bin/python -c "from core import settings; print(settings.status())"

# 确认**没有**依赖 Tesseract（应当报 ModuleNotFoundError，这正是期望结果）
.venv/bin/python -c "import pytesseract" || echo "pytesseract absent (expected)"
```

## 4. 启动 + 探活

```bash
# 后台起服务（不自动开浏览器）
LAB_NO_BROWSER=1 .venv/bin/python app.py &
sleep 8
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/     # 期望 200
# 结束：kill %1
```

Windows（cmd）：

```bat
set LAB_NO_BROWSER=1
start /b .venv\Scripts\python.exe app.py
timeout /t 8 >nul
curl -s -o nul -w "%%{http_code}\n" http://127.0.0.1:8090/
```

端口默认 **8090**（避免与其它常见服务撞 8080）；用 `LAB_PORT` 覆盖。默认只监听 `127.0.0.1`。

## 5. 配置视觉模型（可选，不配也能跑）

配了才能"自动读出比例尺上的数值"；不配时该功能静默降级为手填，不影响其它功能。

```bash
# 方式一：环境变量（回退名 DEEPSEEK_* 也认）
export VISION_API_KEY=sk-xxx
export VISION_BASE_URL=https://api.deepseek.com
export VISION_MODEL=deepseek-v4-flash-vision-exp

# 方式二：项目根目录建 .env（格式同上，见 .env.example）
# 方式三：起服务后在页面上点右上角「API 设置」填（写 .env.local，保存即生效）
```

自检（不启服务）：

```bash
.venv/bin/python -c "
from core.settings import VisionConfig, test_connection
print(test_connection(VisionConfig(api_key='sk-xxx',
                                   base_url='https://api.deepseek.com',
                                   model='deepseek-chat')))"
```

判据：返回 `(True, ...)`，或 `(False, '...')` 里的中文说明能直接对上原因。

## 6. 报错 → 处置对照表

| 症状 | 原因 | 处置 |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | 依赖没装 | 重跑第 2 步 |
| `ImportError: libGL.so.1` | 装了非 headless 的 opencv | `pip uninstall opencv-python && pip install opencv-python-headless` |
| `[Errno 98]` / `[Errno 10048] address already in use` | 端口被占 | `LAB_PORT=8091` 后重跑 |
| 页面一直是空白 | nicegui 版本越界（装了 3.x） | `pip install "nicegui>=2.0,<3.0"` |
| 数值总读不出来 | key / 地址 / 模型不对 | 页面上点「测试连接」看具体 HTTP 码：401=key 无效，404=base_url 可能少 `/v1`，连不上=网络或本地服务没起 |
| 比例尺数字显示成方块 | 系统缺字体 | 装任意 DejaVu / Noto / Arial 系字体 |
| `python -m venv` 报 ensurepip 缺失 | Debian/Ubuntu 缺包 | `sudo apt install python3-venv` |
| 拖入 10 张只上传了几张 | 有人改过 `ui.upload` 的 `reset()` 时机 | `reset()` 只能在 `on_multi_upload` 里调，不能在 `on_upload` 里（Quasar 会 abort 掉同批未完成的上传） |

## 7. 验收清单（Definition of Done）

- [ ] 安装脚本退出码 0，`.venv` 存在
- [ ] 第 3 节四条冒烟测试全部符合预期
- [ ] `GET /` 返回 200
- [ ] 页面能打开、无侧边栏、右上角有「API 设置」按钮
- [ ] 拖入一张 SEM 图 → 预览缩略图出现（16 位 TIF 也不能是空白）
- [ ] 点缩略图 → 图上出现绿色识别线或"没自动认到"的提示，「原比例尺像素长度」有值
- [ ] 未配 API 时全流程不报错（控制台无 traceback），数值手填后能生成图片
- [ ] 「组合成整合图」→「下载图片」能真的下载到文件
- [ ] `git status` 里看不到 `data/`、`.env`、`.env.local`、`.venv/`
