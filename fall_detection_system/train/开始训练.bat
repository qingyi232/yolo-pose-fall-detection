@echo off
chcp 65001 >nul
title 模型训练

echo ============================================
echo    YOLOv8 摔倒检测模型训练
echo ============================================
echo.
echo 请确保已准备好数据集！
echo 数据集应放在 train/datasets/ 目录下
echo.
echo 按任意键开始训练...
pause >nul

cd train
python train.py

echo.
echo 训练完成！
pause
