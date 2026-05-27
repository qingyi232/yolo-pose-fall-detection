@echo off
chcp 65001 >nul
title 摔倒检测系统 - 运行中

echo ============================================
echo    摔倒检测与区域闯入系统
echo ============================================
echo.
echo 系统启动中，请稍候...
echo.
echo 启动后请打开浏览器访问: http://localhost:5000
echo.
echo 按 Ctrl+C 可停止系统
echo ============================================
echo.

python main.py

pause
