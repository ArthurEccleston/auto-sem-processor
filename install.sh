#!/usr/bin/env bash
# AutoSEM Processor - Linux / macOS 安装脚本
# 用法:bash install.sh
set -e
cd "$(dirname "$0")"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[错误] 找不到 python3。请先安装 Python 3.10+(Ubuntu: sudo apt install python3-venv)"
  exit 1
fi

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[错误] Python 版本太低,需要 3.10 或更新。当前:$("$PY" --version)"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[1/3] 创建虚拟环境 .venv ..."
  if ! "$PY" -m venv .venv; then
    echo "[错误] 创建虚拟环境失败。Debian/Ubuntu 上可能需要先 sudo apt install python3-venv"
    exit 1
  fi
else
  echo "[1/3] 虚拟环境已存在,跳过。"
fi

echo "[2/3] 安装依赖(首次会慢一点)..."
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python -m pip install -r requirements.txt; then
  echo "  直连失败,改用清华镜像重试..."
  .venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
fi

echo "[3/3] 自检 ..."
.venv/bin/python -c "import cv2, numpy, PIL, httpx, nicegui; print('依赖 OK')"
[ -f .env ] || cp .env.example .env

echo
echo "  装好了。启动:bash run.sh"
echo "  API 不用在这里配 —— 打开页面后点右上角「API 设置」填就行。"
