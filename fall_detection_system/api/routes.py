"""
模块8: API服务与消息推送模块
"""
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
from pathlib import Path
from loguru import logger

# 获取web目录路径
WEB_DIR = Path(__file__).parent.parent / 'web'

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# ==================== 静态文件服务 ====================

@app.route('/')
def index():
    """首页"""
    return send_from_directory(WEB_DIR, 'index.html')

# 全局引用（由main.py注入）
detection_system = None


def init_api(system):
    """初始化API，注入检测系统实例"""
    global detection_system
    detection_system = system


# ==================== 系统状态 ====================

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    return jsonify({
        'running': detection_system.running,
        'cameras': detection_system.video_manager.get_status(),
        'config': {
            'fall_detection': detection_system.config.get('fall_detection.enabled'),
            'intrusion_detection': detection_system.config.get('intrusion.enabled')
        }
    })


@app.route('/api/start', methods=['POST'])
def start_system():
    """启动系统"""
    if detection_system:
        detection_system.start()
        return jsonify({'status': 'started'})
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """停止系统"""
    if detection_system:
        detection_system.stop()
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'System not initialized'}), 500


# ==================== 配置管理 ====================

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    if detection_system:
        return jsonify(detection_system.config.get_all())
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/config', methods=['PUT'])
def update_config():
    """更新配置"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.json
    for key, value in data.items():
        detection_system.config.set(key, value)
    
    return jsonify({'status': 'updated'})


# ==================== 区域管理 ====================

@app.route('/api/zones', methods=['GET'])
def get_zones():
    """获取所有检测区域"""
    if detection_system:
        return jsonify(detection_system.intrusion_detector.get_zones_info())
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/zones', methods=['POST'])
def add_zone():
    """添加检测区域"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.json
    zone_id = detection_system.intrusion_detector.add_zone(
        name=data['name'],
        points=data['points'],
        color=data.get('color', [255, 0, 0])
    )
    return jsonify({'id': zone_id, 'status': 'created'})


@app.route('/api/zones/<int:zone_id>', methods=['PUT'])
def update_zone(zone_id):
    """更新检测区域"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.json
    detection_system.intrusion_detector.update_zone(zone_id, data['points'])
    return jsonify({'status': 'updated'})


@app.route('/api/zones/<int:zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """删除检测区域"""
    if detection_system:
        detection_system.intrusion_detector.remove_zone(zone_id)
        return jsonify({'status': 'deleted'})
    return jsonify({'error': 'System not initialized'}), 500


# ==================== 事件管理 ====================

@app.route('/api/events', methods=['GET'])
def get_events():
    """获取事件列表"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    event_type = request.args.get('type')
    camera_id = request.args.get('camera_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', 100, type=int)
    
    events = detection_system.alert_manager.db.get_events(
        event_type=event_type,
        camera_id=camera_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return jsonify(events)


@app.route('/api/events/<int:event_id>/handle', methods=['POST'])
def handle_event(event_id):
    """标记事件已处理"""
    if detection_system:
        detection_system.alert_manager.mark_handled(event_id)
        return jsonify({'status': 'handled'})
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取统计数据"""
    if detection_system is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    stats = detection_system.alert_manager.get_statistics(start_date, end_date)
    return jsonify(stats)


# ==================== 视频流 ====================

@app.route('/api/video/<int:camera_id>')
def video_feed(camera_id):
    """视频流（MJPEG）"""
    def generate():
        while True:
            if detection_system and detection_system.running:
                frame = detection_system.get_processed_frame(camera_id)
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30fps，让出CPU时间给其他请求
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/snapshot/<int:camera_id>')
def get_snapshot(camera_id):
    """获取快照"""
    if detection_system:
        frame = detection_system.get_processed_frame(camera_id)
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return jsonify({'error': 'No frame available'}), 404


# ==================== 系统监控 ====================

@app.route('/api/system/metrics', methods=['GET'])
def get_system_metrics():
    """获取系统指标"""
    if detection_system:
        return jsonify(detection_system.system_monitor.get_system_metrics())
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/system/metrics/history', methods=['GET'])
def get_metrics_history():
    """获取历史指标"""
    if detection_system:
        limit = request.args.get('limit', 100, type=int)
        return jsonify(detection_system.system_monitor.get_metrics_history(limit))
    return jsonify({'error': 'System not initialized'}), 500


@app.route('/api/system/health', methods=['GET'])
def health_check():
    """健康检查"""
    if detection_system:
        return jsonify(detection_system.service_health.check_all())
    return jsonify({'status': 'unhealthy', 'error': 'System not initialized'}), 500


# ==================== WebSocket ====================

@socketio.on('connect')
def handle_connect():
    """WebSocket连接"""
    logger.info("WebSocket客户端连接")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开"""
    logger.info("WebSocket客户端断开")


@socketio.on('subscribe_events')
def handle_subscribe():
    """订阅事件推送"""
    logger.info("客户端订阅事件推送")


def broadcast_event(event):
    """广播事件到所有WebSocket客户端"""
    socketio.emit('event', event.to_dict())


def broadcast_frame(camera_id: int, frame):
    """广播视频帧"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('frame', {
        'camera_id': camera_id,
        'frame': frame_base64
    })


def run_api(host: str = '0.0.0.0', port: int = 5000):
    """运行API服务"""
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
