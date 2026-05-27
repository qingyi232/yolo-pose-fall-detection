"""
模块9: 可视化与存储模块 - 数据库部分
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
from .config_manager import config

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        db_path = config.get('database.path', 'data/fall_detection.db')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. 事件表 - 存储检测到的摔倒和闯入事件
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                camera_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                handled INTEGER DEFAULT 0,
                snapshot_path TEXT,
                confidence REAL DEFAULT 0.0,
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
        ''')
        
        # 2. 系统日志表 - 记录系统运行日志
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                message TEXT,
                module TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 3. 统计数据表 - 按日期统计检测数据
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                camera_id INTEGER,
                fall_count INTEGER DEFAULT 0,
                intrusion_count INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
        ''')
        
        # 4. 摄像头配置表 - 存储摄像头信息
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                location TEXT,
                resolution TEXT,
                fps INTEGER DEFAULT 30,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 5. 检测区域表 - 存储闯入检测区域配置
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                camera_id INTEGER,
                points TEXT NOT NULL,
                color TEXT DEFAULT '[255,0,0]',
                enabled INTEGER DEFAULT 1,
                alert_level INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
        ''')
        
        # 6. 用户表 - 系统用户管理
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'operator',
                email TEXT,
                phone TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        ''')
        
        # 7. 预警配置表 - 预警通知配置
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                notify_email INTEGER DEFAULT 0,
                notify_sound INTEGER DEFAULT 1,
                notify_webhook INTEGER DEFAULT 0,
                cooldown_seconds INTEGER DEFAULT 30,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def _get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(str(self.db_path))
    
    def save_event(self, event) -> int:
        """保存事件"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (type, camera_id, timestamp, details, handled)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            event.type,
            event.camera_id,
            event.timestamp.isoformat(),
            json.dumps(event.details),
            0
        ))
        
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 更新统计
        self._update_statistics(event)
        
        return event_id
    
    def get_events(self, event_type: str = None, camera_id: int = None, 
                   start_date: str = None, end_date: str = None, 
                   limit: int = 100) -> list:
        """查询事件"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append({
                'id': row[0],
                'type': row[1],
                'camera_id': row[2],
                'timestamp': row[3],
                'details': json.loads(row[4]) if row[4] else {},
                'handled': bool(row[5]),
                'snapshot_path': row[6]
            })
        
        return events
    
    def update_event(self, event_id: int, updates: dict):
        """更新事件"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [event_id]
        
        cursor.execute(f"UPDATE events SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()
    
    def _update_statistics(self, event):
        """更新统计数据"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        # 检查今天的记录是否存在
        cursor.execute('''
            SELECT id FROM statistics WHERE date = ? AND camera_id = ?
        ''', (today, event.camera_id))
        
        row = cursor.fetchone()
        
        if row:
            # 更新现有记录
            if event.type == 'fall':
                cursor.execute('''
                    UPDATE statistics SET fall_count = fall_count + 1, 
                    total_detections = total_detections + 1 WHERE id = ?
                ''', (row[0],))
            elif event.type == 'intrusion':
                cursor.execute('''
                    UPDATE statistics SET intrusion_count = intrusion_count + 1,
                    total_detections = total_detections + 1 WHERE id = ?
                ''', (row[0],))
        else:
            # 创建新记录
            fall_count = 1 if event.type == 'fall' else 0
            intrusion_count = 1 if event.type == 'intrusion' else 0
            cursor.execute('''
                INSERT INTO statistics (date, camera_id, fall_count, intrusion_count, total_detections)
                VALUES (?, ?, ?, ?, 1)
            ''', (today, event.camera_id, fall_count, intrusion_count))
        
        conn.commit()
        conn.close()
    
    def get_event_statistics(self, start_date: str = None, end_date: str = None) -> dict:
        """获取事件统计"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT date, camera_id, fall_count, intrusion_count, total_detections FROM statistics WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # 汇总统计
        total_falls = sum(row[2] for row in rows)
        total_intrusions = sum(row[3] for row in rows)
        total_detections = sum(row[4] for row in rows)
        
        daily_stats = [{
            'date': row[0],
            'camera_id': row[1],
            'fall_count': row[2],
            'intrusion_count': row[3],
            'total': row[4]
        } for row in rows]
        
        return {
            'total_falls': total_falls,
            'total_intrusions': total_intrusions,
            'total_detections': total_detections,
            'daily': daily_stats
        }
    
    def save_log(self, level: str, message: str):
        """保存系统日志"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_logs (level, message) VALUES (?, ?)
        ''', (level, message))
        
        conn.commit()
        conn.close()
    
    def get_logs(self, level: str = None, limit: int = 100) -> list:
        """获取系统日志"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if level:
            cursor.execute('''
                SELECT * FROM system_logs WHERE level = ? 
                ORDER BY timestamp DESC LIMIT ?
            ''', (level, limit))
        else:
            cursor.execute('''
                SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'level': row[1],
            'message': row[2],
            'timestamp': row[3]
        } for row in rows]
