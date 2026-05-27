# 基于YOLO与YOLO-Pose姿态估计的摔倒检测与区域闯入系统

## 系统概述

本系统是一个基于深度学习的智能视频监控系统，主要功能包括：
- 实时人体检测与姿态估计
- 摔倒行为检测与预警
- 区域闯入检测与预警
- 多摄像头视频流管理
- Web可视化界面
- RESTful API接口

## 系统架构

```
fall_detection_system/
├── config/
│   └── config.yaml          # 系统配置文件
├── core/
│   ├── config_manager.py    # 模块3: 系统配置与管理
│   ├── detector.py          # 模块1: 多目标检测与人体姿态估计
│   ├── video_stream.py      # 模块2: 视频流处理与数据采集
│   ├── fall_detector.py     # 模块4: 摔倒检测算法
│   ├── intrusion_detector.py# 模块5: 区域闯入检测
│   ├── alert_manager.py     # 模块6: 事件分析与预警
│   ├── monitor.py           # 模块7: 机器运维与监控
│   └── database.py          # 模块9: 数据存储
├── api/
│   └── routes.py            # 模块8: API服务与消息推送
├── web/
│   └── index.html           # 模块9: 可视化界面
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖列表
└── README.md
```

## 九大功能模块

1. **多目标检测与人体姿态估计模块** - 使用YOLOv8进行人体检测，YOLOv8-Pose进行17关键点姿态估计
2. **视频流处理与数据采集模块** - 支持本地摄像头、RTSP流等多种视频源
3. **系统配置与管理模块** - YAML配置文件管理，支持动态更新
4. **摔倒检测算法模块** - 基于姿态角度、身体比例、关键点分布的多维度摔倒判断
5. **区域闯入检测模块** - 支持多边形区域绘制，实时闯入检测
6. **事件分析与预警模块** - 事件记录、声音/邮件/Webhook多渠道预警
7. **机器运维与监控模块** - CPU/内存/GPU监控，服务健康检查
8. **API服务与消息推送模块** - RESTful API + WebSocket实时推送
9. **可视化与存储模块** - Web界面 + SQLite数据库存储

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

首次运行时会自动下载YOLOv8模型，也可手动下载：
- yolov8n.pt (目标检测)
- yolov8n-pose.pt (姿态估计)

### 3. 配置系统

编辑 `config/config.yaml` 配置视频源、检测区域等参数。

### 4. 运行系统

```bash
python main.py --host 0.0.0.0 --port 5000
```

### 5. 访问界面

打开浏览器访问: http://localhost:5000

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| /api/status | GET | 获取系统状态 |
| /api/start | POST | 启动系统 |
| /api/stop | POST | 停止系统 |
| /api/config | GET/PUT | 获取/更新配置 |
| /api/zones | GET/POST | 获取/添加检测区域 |
| /api/zones/{id} | PUT/DELETE | 更新/删除区域 |
| /api/events | GET | 获取事件列表 |
| /api/statistics | GET | 获取统计数据 |
| /api/video/{camera_id} | GET | 视频流(MJPEG) |
| /api/system/metrics | GET | 系统指标 |

## 摔倒检测算法

系统采用多维度融合的摔倒检测算法：

1. **身体高宽比检测** - 正常站立时身高>身宽，摔倒时相反
2. **躯干角度检测** - 计算肩部-髋部连线与垂直方向的夹角
3. **头部位置检测** - 检测头部是否异常低于髋部
4. **关键点分布检测** - 分析17个关键点的水平/垂直分布比例

## 技术栈

- **深度学习**: PyTorch, Ultralytics YOLOv8
- **计算机视觉**: OpenCV
- **后端框架**: Flask, Flask-SocketIO
- **数据库**: SQLite
- **前端**: HTML5, TailwindCSS, Chart.js

## 注意事项

- 建议使用GPU加速推理
- 首次运行需要下载模型文件
- 确保摄像头权限已开启
