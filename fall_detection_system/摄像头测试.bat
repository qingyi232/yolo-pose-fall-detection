@echo off
chcp 65001 >nul
title 摔像头测试

echo ============================================
echo    摄像头测试
echo ============================================
echo.
echo 操作说明:
echo   按 Q 键 - 退出
echo   按 F 键 - 全屏
echo.
echo 正在启动摄像头...
echo.

python test_with_video.py 0

pause
