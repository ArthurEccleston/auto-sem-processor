#!/usr/bin/env bash
# AutoSEM Processor - Linux / macOS 启动脚本
# 用法:bash run.sh   (缺环境会自动先装)
set -e
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || bash install.sh

echo
echo "  ============================================"
echo "    AutoSEM Processor  正在启动..."
echo "    地址: http://127.0.0.1:8090"
echo "    按 Ctrl+C 停止"
echo "  ============================================"
echo

exec .venv/bin/python app.py
