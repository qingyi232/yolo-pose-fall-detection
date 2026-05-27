@echo off
chcp 65001 >nul
title 摔倒检测系统 - 一键安装

echo ============================================
echo    摔倒检测与区域闯入系统 - 一键安装
echo ============================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8以上版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 检测到Python已安装
python --version

echo.
echo [2/3] 正在安装依赖包，请稍候...
echo （首次安装可能需要5-10分钟，取决于网络速度）
echo.

pip install ultralytics opencv-python flask flask-cors flask-socketio pyyaml loguru psutil requests -i https://pypi.tuna.tsinghua.edu.cn/simple

if errorlevel 1 (
    echo.
    echo [警告] 部分依赖安装失败，尝试使用默认源重新安装...
    pip install ultralytics opencv-python flask flask-cors flask-socketio pyyaml loguru psutil requests
)

echo.
echo [3/3] 创建必要目录...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "snapshots" mkdir snapshots

echo.
echo ============================================
echo    安装完成！
echo ============================================
echo.
echo 运行方式:
echo   1. 双击 "启动系统.bat" 启动完整系统
echo   2. 双击 "摄像头测试.bat" 测试摄像头
echo.
pause
