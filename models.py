from app import db
from datetime import datetime
from sqlalchemy import Text, JSON
import json

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hostname = db.Column(db.String(255), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)
    status = db.Column(db.String(20), default='offline')  # online, offline, testing
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    system_info = db.Column(Text)  # JSON string for system information
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    test_results = db.relationship('TestResult', backref='client', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'status': self.status,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'system_info': json.loads(self.system_info) if self.system_info else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(Text)
    destination = db.Column(db.String(255), nullable=False)  # Target host/IP to test
    scheduled_time = db.Column(db.DateTime)
    duration = db.Column(db.Integer, default=300)  # Test duration in seconds
    interval = db.Column(db.Integer, default=5)  # Measurement interval in seconds
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    test_results = db.relationship('TestResult', backref='test', lazy=True)
    test_clients = db.relationship('TestClient', backref='test', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'destination': self.destination,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'duration': self.duration,
            'interval': self.interval,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class TestClient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)
    status = db.Column(db.String(20), default='assigned')  # assigned, running, completed, failed
    
    __table_args__ = (db.UniqueConstraint('test_id', 'client_id'),)

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # System metrics
    cpu_percent = db.Column(db.Float)
    memory_percent = db.Column(db.Float)
    memory_used = db.Column(db.BigInteger)  # bytes
    memory_total = db.Column(db.BigInteger)  # bytes
    disk_percent = db.Column(db.Float)
    disk_used = db.Column(db.BigInteger)  # bytes
    disk_total = db.Column(db.BigInteger)  # bytes
    
    # Network metrics
    ping_latency = db.Column(db.Float)  # milliseconds
    ping_packet_loss = db.Column(db.Float)  # percentage
    traceroute_hops = db.Column(db.Integer)
    traceroute_data = db.Column(Text)  # JSON string
    bandwidth_upload = db.Column(db.Float)  # Mbps
    bandwidth_download = db.Column(db.Float)  # Mbps
    
    def to_dict(self):
        return {
            'id': self.id,
            'test_id': self.test_id,
            'client_id': self.client_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_total': self.memory_total,
            'disk_percent': self.disk_percent,
            'disk_used': self.disk_used,
            'disk_total': self.disk_total,
            'ping_latency': self.ping_latency,
            'ping_packet_loss': self.ping_packet_loss,
            'traceroute_hops': self.traceroute_hops,
            'traceroute_data': json.loads(self.traceroute_data) if self.traceroute_data else {},
            'bandwidth_upload': self.bandwidth_upload,
            'bandwidth_download': self.bandwidth_download
        }
