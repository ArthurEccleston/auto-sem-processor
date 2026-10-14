@echo off
chcp 65001 >nul
title SEM 比例尺分析工具
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [首次运行] 还没装环境,先跑一遍安装...
  call install.bat /q
  if errorlevel 1 (
    echo 安装没成功,请手动双击 install.bat 看看报什么错。
    pause
    exit /b 1
  )
)

rem 已经在跑?那就别再起一个,直接开浏览器就行
netstat -ano | findstr ":8090 " | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo 服务已经在运行,正在打开浏览器...
  start "" http://127.0.0.1:8090
  timeout /t 2 >nul
  exit /b
)

echo.
echo   ============================================
echo     SEM 比例尺分析工具  正在启动...
echo     地址: http://127.0.0.1:8090
echo     浏览器会自动打开
echo.
echo     关闭这个窗口 = 停止服务
echo   ============================================
echo.

".venv\Scripts\python.exe" app.py

rem 服务异常退出时留个窗口看报错
echo.
echo 服务已停止。
pause
