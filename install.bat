@echo off
chcp 65001 >nul
title AutoSEM Processor - 安装
cd /d "%~dp0"

rem 传 /q 参数可以跳过结尾的 pause(run.bat 会这样调它)
set QUIET=%1

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没找到 python。请先安装 Python 3.10 或更新版本:
  echo        https://www.python.org/downloads/
  echo        安装时记得勾上 "Add python.exe to PATH"。
  if not "%QUIET%"=="/q" pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
  echo [错误] Python 版本太低,需要 3.10 或更新。
  python --version
  if not "%QUIET%"=="/q" pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [错误] 创建虚拟环境失败。
    if not "%QUIET%"=="/q" pause
    exit /b 1
  )
) else (
  echo [1/3] 虚拟环境已存在,跳过。
)

echo [2/3] 安装依赖(首次会慢一点)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
if errorlevel 1 (
  echo   镜像没成功,改用官方源重试...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [错误] 依赖安装失败,请检查网络后重跑本脚本。
    if not "%QUIET%"=="/q" pause
    exit /b 1
  )
)

echo [3/3] 自检 ...
".venv\Scripts\python.exe" -c "import cv2, numpy, PIL, httpx, nicegui; print('依赖 OK')"
if errorlevel 1 (
  echo [错误] 依赖导入失败。
  if not "%QUIET%"=="/q" pause
  exit /b 1
)

if not exist ".env" copy /y .env.example .env >nul

echo.
echo   装好了。启动:双击 run.bat(或 .venv\Scripts\python.exe app.py)
echo   API 不用在这里配 —— 打开页面后点右上角「⚙ API 设置」填就行。
echo.
if not "%QUIET%"=="/q" pause
exit /b 0
