"""
模块7: 机器运维与监控模块
"""
import psutil
import threading
import time
from datetime import datetime
from loguru import logger
from .config_manager import config

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history = 1000
        
        # 阈值配置
        self.cpu_threshold = 90
        self.memory_threshold = 90
        self.disk_threshold = 90
        
        # 回调
        self.alert_callback = None
    
    def start(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            metrics = self.get_system_metrics()
            self.metrics_history.append(metrics)
            
            # 限制历史记录数量
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            # 检查阈值
            self._check_thresholds(metrics)
            
            time.sleep(5)  # 每5秒采集一次
    
    def get_system_metrics(self) -> dict:
        """获取系统指标"""
        cpu_percent = psutil.cpu_percent(interval=0)  # 非阻塞
        memory = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage('C:/')  # Windows用C盘
        except:
            disk = psutil.disk_usage('/')
        
        # GPU信息（如果可用）
        gpu_info = self._get_gpu_info()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'gpu': gpu_info
        }
    
    def _get_gpu_info(self) -> dict:
        """获取GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'available': True,
                    'name': torch.cuda.get_device_name(0),
                    'memory_total': torch.cuda.get_device_properties(0).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(0),
                    'memory_cached': torch.cuda.memory_reserved(0)
                }
        except:
            pass
        return {'available': False}
    
    def _check_thresholds(self, metrics: dict):
        """检查阈值并触发告警"""
        alerts = []
        
        if metrics['cpu']['percent'] > self.cpu_threshold:
            alerts.append(f"CPU使用率过高: {metrics['cpu']['percent']}%")
        
        if metrics['memory']['percent'] > self.memory_threshold:
            alerts.append(f"内存使用率过高: {metrics['memory']['percent']}%")
        
        if metrics['disk']['percent'] > self.disk_threshold:
            alerts.append(f"磁盘使用率过高: {metrics['disk']['percent']}%")
        
        if alerts and self.alert_callback:
            for alert in alerts:
                logger.warning(alert)
                self.alert_callback(alert)
    
    def get_process_info(self) -> dict:
        """获取当前进程信息"""
        process = psutil.Process()
        return {
            'pid': process.pid,
            'name': process.name(),
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_info': {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms
            },
            'threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
        }
    
    def get_metrics_history(self, limit: int = 100) -> list:
        """获取历史指标"""
        return self.metrics_history[-limit:]
    
    def set_alert_callback(self, callback):
        """设置告警回调"""
        self.alert_callback = callback


class ServiceHealth:
    """服务健康检查"""
    
    def __init__(self):
        self.services = {}
    
    def register_service(self, name: str, check_func):
        """注册服务健康检查函数"""
        self.services[name] = check_func
    
    def check_all(self) -> dict:
        """检查所有服务"""
        results = {}
        for name, check_func in self.services.items():
            try:
                status = check_func()
                results[name] = {
                    'status': 'healthy' if status else 'unhealthy',
                    'checked_at': datetime.now().isoformat()
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'checked_at': datetime.now().isoformat()
                }
        return results
    
    def check_service(self, name: str) -> dict:
        """检查单个服务"""
        if name not in self.services:
            return {'status': 'unknown', 'error': 'Service not registered'}
        
        try:
            status = self.services[name]()
            return {
                'status': 'healthy' if status else 'unhealthy',
                'checked_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'checked_at': datetime.now().isoformat()
            }
