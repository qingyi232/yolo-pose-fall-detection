"""
模块3: 系统配置与管理模块
"""
import yaml
import os
from pathlib import Path
from loguru import logger

class ConfigManager:
    """配置管理器"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self._config = {}
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
            logger.info("配置文件保存成功")
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
    
    def get(self, key: str, default=None):
        """获取配置项，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value):
        """设置配置项"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config()
    
    def get_all(self):
        """获取所有配置"""
        return self._config.copy()
    
    def update_zone(self, zone_id: int, points: list):
        """更新区域配置"""
        zones = self.get('intrusion.zones', [])
        for zone in zones:
            if zone['id'] == zone_id:
                zone['points'] = points
                break
        self.set('intrusion.zones', zones)
    
    def add_zone(self, name: str, points: list, color: list = [255, 0, 0]):
        """添加新区域"""
        zones = self.get('intrusion.zones', [])
        new_id = max([z['id'] for z in zones], default=0) + 1
        zones.append({
            'id': new_id,
            'name': name,
            'points': points,
            'color': color
        })
        self.set('intrusion.zones', zones)
        return new_id
    
    def delete_zone(self, zone_id: int):
        """删除区域"""
        zones = self.get('intrusion.zones', [])
        zones = [z for z in zones if z['id'] != zone_id]
        self.set('intrusion.zones', zones)

# 全局配置实例
config = ConfigManager()
