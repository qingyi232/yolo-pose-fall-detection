"""
模块6: 事件分析与预警模块
"""
import time
import threading
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import deque
from loguru import logger
from .config_manager import config
from .database import DatabaseManager

class Event:
    """事件类"""
    
    EVENT_FALL = "fall"
    EVENT_INTRUSION = "intrusion"
    
    def __init__(self, event_type: str, camera_id: int, details: dict = None):
        self.id = None
        self.type = event_type
        self.camera_id = camera_id
        self.timestamp = datetime.now()
        self.details = details or {}
        self.handled = False
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'handled': self.handled
        }


class AlertManager:
    """预警管理器"""
    
    def __init__(self):
        self.enabled = config.get('alert.enabled', True)
        self.cooldown = config.get('alert.cooldown', 30)
        self.methods = config.get('alert.methods', {})
        
        self.last_alert_time = {}  # {event_key: timestamp}
        self.event_queue = deque(maxlen=1000)
        self.db = DatabaseManager()
        
        # 回调函数列表
        self.callbacks = []
    
    def register_callback(self, callback):
        """注册事件回调"""
        self.callbacks.append(callback)
    
    def trigger_event(self, event: Event):
        """触发事件"""
        if not self.enabled:
            return
        
        # 检查冷却时间
        event_key = f"{event.type}_{event.camera_id}"
        current_time = time.time()
        
        if event_key in self.last_alert_time:
            if current_time - self.last_alert_time[event_key] < self.cooldown:
                logger.debug(f"事件冷却中: {event_key}")
                return
        
        self.last_alert_time[event_key] = current_time
        
        # 保存到数据库
        event.id = self.db.save_event(event)
        
        # 添加到队列
        self.event_queue.append(event)
        
        # 执行回调
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
        
        # 发送预警
        self._send_alerts(event)
        
        logger.warning(f"触发事件: {event.type} - 摄像头{event.camera_id}")
    
    def _send_alerts(self, event: Event):
        """发送预警通知"""
        message = self._format_message(event)
        
        # 声音预警
        if self.methods.get('sound', False):
            self._play_sound()
        
        # Webhook推送
        if self.methods.get('webhook', False):
            threading.Thread(target=self._send_webhook, args=(event,)).start()
        
        # 邮件通知
        if self.methods.get('email', False):
            threading.Thread(target=self._send_email, args=(message,)).start()
    
    def _format_message(self, event: Event) -> str:
        """格式化预警消息"""
        type_names = {
            Event.EVENT_FALL: "摔倒检测",
            Event.EVENT_INTRUSION: "区域闯入"
        }
        
        msg = f"""
【预警通知】
类型: {type_names.get(event.type, event.type)}
时间: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
摄像头: {event.camera_id}
详情: {event.details}
        """
        return msg.strip()
    
    def _play_sound(self):
        """播放预警声音"""
        try:
            import winsound
            winsound.Beep(1000, 500)  # Windows
        except:
            print('\a')  # 终端响铃
    
    def _send_webhook(self, event: Event):
        """发送Webhook通知"""
        webhook_url = config.get('alert.webhook_url')
        if not webhook_url:
            return
        
        try:
            payload = event.to_dict()
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Webhook发送成功")
            else:
                logger.error(f"Webhook发送失败: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook发送异常: {e}")
    
    def _send_email(self, message: str):
        """发送邮件通知"""
        email_config = config.get('alert.email', {})
        
        smtp_server = email_config.get('smtp_server')
        smtp_port = email_config.get('smtp_port', 587)
        sender = email_config.get('sender')
        password = email_config.get('password')
        receivers = email_config.get('receivers', [])
        
        if not all([smtp_server, sender, password, receivers]):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(receivers)
            msg['Subject'] = "【摔倒检测系统】预警通知"
            msg.attach(MIMEText(message, 'plain', 'utf-8'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, receivers, msg.as_string())
            
            logger.info("邮件发送成功")
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
    
    def get_recent_events(self, limit: int = 50) -> list:
        """获取最近的事件"""
        return self.db.get_events(limit=limit)
    
    def get_statistics(self, start_date: str = None, end_date: str = None) -> dict:
        """获取事件统计"""
        return self.db.get_event_statistics(start_date, end_date)
    
    def mark_handled(self, event_id: int):
        """标记事件已处理"""
        self.db.update_event(event_id, {'handled': True})
