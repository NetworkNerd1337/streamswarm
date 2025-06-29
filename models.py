from app import db
from datetime import datetime, timezone
import zoneinfo
from sqlalchemy import Text, JSON
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json
import secrets
import string

class SystemConfig(db.Model):
    """System configuration settings"""
    __tablename__ = 'system_config'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None), onupdate=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    
    @classmethod
    def get_setting(cls, key, default=None):
        """Get a configuration setting value"""
        config = cls.query.filter_by(key=key).first()
        return config.value if config else default
    
    @classmethod
    def set_setting(cls, key, value, description=None):
        """Set a configuration setting value"""
        config = cls.query.filter_by(key=key).first()
        if config:
            config.value = str(value)
            config.updated_at = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
            if description:
                config.description = description
        else:
            config = cls(key=key, value=str(value), description=description)
            db.session.add(config)
        db.session.commit()
        return config
    
    @classmethod
    def is_development_mode(cls):
        """Check if development mode is enabled"""
        setting_value = cls.get_setting('development_mode', 'false')
        if setting_value is None:
            return False
        return str(setting_value).lower() == 'true'
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class User(db.Model):
    """Web GUI user authentication model - separate from client API tokens"""
    __tablename__ = 'web_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    last_login = db.Column(db.DateTime, nullable=True)
    
    def __init__(self, username=None, email=None, role='user'):
        if username:
            self.username = username
        if email:
            self.email = email
        self.role = role
        self.is_active = True
    
    # Flask-Login integration methods
    def is_authenticated(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user has admin role"""
        return self.role == 'admin'
    

    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
        db.session.commit()
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': bool(self.is_active),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hostname = db.Column(db.String(255), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)
    status = db.Column(db.String(20), default='offline')  # online, offline, testing
    last_seen = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    system_info = db.Column(Text)  # JSON string for system information
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    
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
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
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

class ApiToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)  # Human-readable name for the token
    description = db.Column(Text)  # Optional description
    status = db.Column(db.String(20), default='available')  # available, consumed, revoked
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=True)  # Which client consumed it
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    consumed_at = db.Column(db.DateTime, nullable=True)  # When token was consumed
    last_used = db.Column(db.DateTime, nullable=True)  # Last API request with this token
    
    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.token = self.generate_token()
    
    @staticmethod
    def generate_token():
        """Generate a secure random token"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(32))
    
    def consume(self, client_id):
        """Mark token as consumed by a client"""
        self.status = 'consumed'
        self.client_id = client_id
        self.consumed_at = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    
    def to_dict(self):
        return {
            'id': self.id,
            'token': self.token,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'client_id': self.client_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'consumed_at': self.consumed_at.isoformat() if self.consumed_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None))
    
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
    
    # Advanced network metrics
    dns_resolution_time = db.Column(db.Float)  # milliseconds
    tcp_connect_time = db.Column(db.Float)  # milliseconds
    ssl_handshake_time = db.Column(db.Float)  # milliseconds
    ttfb = db.Column(db.Float)  # time to first byte in milliseconds
    jitter = db.Column(db.Float)  # network jitter in milliseconds
    
    # Network interface metrics
    network_bytes_sent = db.Column(db.BigInteger)  # bytes
    network_bytes_recv = db.Column(db.BigInteger)  # bytes
    network_packets_sent = db.Column(db.BigInteger)
    network_packets_recv = db.Column(db.BigInteger)
    network_errors_in = db.Column(db.Integer)
    network_errors_out = db.Column(db.Integer)
    network_drops_in = db.Column(db.Integer)
    network_drops_out = db.Column(db.Integer)
    
    # Enhanced CPU metrics
    cpu_load_1min = db.Column(db.Float)  # 1-minute load average
    cpu_load_5min = db.Column(db.Float)  # 5-minute load average
    cpu_load_15min = db.Column(db.Float)  # 15-minute load average
    cpu_cores = db.Column(db.Integer)  # number of CPU cores
    cpu_freq_current = db.Column(db.Float)  # current CPU frequency MHz
    cpu_context_switches = db.Column(db.BigInteger)  # context switches per second
    cpu_interrupts = db.Column(db.BigInteger)  # interrupts per second
    
    # Enhanced memory metrics
    memory_available = db.Column(db.BigInteger)  # available memory bytes
    memory_cached = db.Column(db.BigInteger)  # cached memory bytes
    memory_buffers = db.Column(db.BigInteger)  # buffer memory bytes
    memory_shared = db.Column(db.BigInteger)  # shared memory bytes
    swap_total = db.Column(db.BigInteger)  # total swap bytes
    swap_used = db.Column(db.BigInteger)  # used swap bytes
    swap_percent = db.Column(db.Float)  # swap usage percentage
    
    # Enhanced disk metrics
    disk_read_iops = db.Column(db.Float)  # read operations per second
    disk_write_iops = db.Column(db.Float)  # write operations per second
    disk_read_bytes_sec = db.Column(db.BigInteger)  # bytes read per second
    disk_write_bytes_sec = db.Column(db.BigInteger)  # bytes written per second
    disk_io_util = db.Column(db.Float)  # disk utilization percentage
    
    # Process and system metrics
    process_count = db.Column(db.Integer)  # total running processes
    top_processes_cpu = db.Column(Text)  # JSON string of top processes by CPU usage
    top_processes_memory = db.Column(Text)  # JSON string of top processes by memory usage
    tcp_connections = db.Column(db.Integer)  # active TCP connections
    open_files = db.Column(db.Integer)  # open file descriptors
    
    # Temperature metrics (where available)
    cpu_temperature = db.Column(db.Float)  # CPU temperature in Celsius
    disk_temperature = db.Column(db.Float)  # disk temperature in Celsius
    
    # QoS metrics
    dscp_value = db.Column(db.Integer)  # DSCP value from IP header
    cos_value = db.Column(db.Integer)  # CoS value from Ethernet frame
    traffic_class = db.Column(db.String(50))  # Classified traffic type
    qos_policy_compliant = db.Column(db.Boolean)  # Whether QoS marking is correct
    bandwidth_per_class = db.Column(Text)  # JSON string of per-class bandwidth usage
    
    # Advanced Network-Level Metrics
    mtu_size = db.Column(db.Integer)  # Maximum Transmission Unit discovered
    tcp_window_size = db.Column(db.Integer)  # TCP window size in bytes
    tcp_window_scaling = db.Column(db.Boolean)  # TCP window scaling enabled
    tcp_congestion_window = db.Column(db.Integer)  # TCP congestion window size
    tcp_retransmission_rate = db.Column(db.Float)  # Percentage of retransmitted packets
    tcp_out_of_order_packets = db.Column(db.Integer)  # Count of out-of-order packets
    tcp_duplicate_acks = db.Column(db.Integer)  # Count of duplicate ACKs
    
    # TCP Window Analysis Metrics
    tcp_rtt_min = db.Column(db.Float)  # Minimum RTT during test (ms)
    tcp_rtt_max = db.Column(db.Float)  # Maximum RTT during test (ms)
    tcp_rtt_avg = db.Column(db.Float)  # Average RTT during test (ms)
    tcp_rtt_variation = db.Column(db.Float)  # RTT variation (max - min) (ms)
    tcp_cwnd_min = db.Column(db.Integer)  # Minimum congestion window size
    tcp_cwnd_max = db.Column(db.Integer)  # Maximum congestion window size
    tcp_cwnd_avg = db.Column(db.Float)  # Average congestion window size
    tcp_ssthresh_avg = db.Column(db.Float)  # Average slow start threshold
    tcp_congestion_events = db.Column(db.Integer)  # Number of congestion events detected
    tcp_retransmissions = db.Column(db.Integer)  # Total retransmissions during test
    tcp_window_efficiency = db.Column(db.Float)  # TCP window efficiency score (0-100)
    tcp_bottleneck_type = db.Column(db.String(50))  # Bottleneck attribution (network_congestion, server_limited, etc)
    tcp_window_timeline = db.Column(Text)  # JSON string of window behavior timeline
    
    # TCP Handshake Timing Diagnostics
    tcp_handshake_syn_time = db.Column(db.Float)  # Time to send SYN packet (ms)
    tcp_handshake_synack_time = db.Column(db.Float)  # Time to receive SYN-ACK (ms)
    tcp_handshake_ack_time = db.Column(db.Float)  # Time to send ACK packet (ms)
    tcp_handshake_total_time = db.Column(db.Float)  # Total handshake completion time (ms)
    tcp_handshake_network_delay = db.Column(db.Float)  # Estimated one-way network delay (ms)
    tcp_handshake_server_processing = db.Column(db.Float)  # Estimated server processing time (ms)
    tcp_handshake_analysis = db.Column(Text)  # Diagnostic analysis text
    
    # Advanced QoS Metrics
    per_dscp_latency = db.Column(Text)  # JSON string of latency per DSCP class
    traffic_policing_detected = db.Column(db.Boolean)  # Rate limiting/shaping detected
    queue_depth = db.Column(db.Integer)  # Network buffer/queue depth
    ecn_capable = db.Column(db.Boolean)  # ECN (Explicit Congestion Notification) support
    ecn_congestion_experienced = db.Column(db.Boolean)  # ECN congestion signaling
    flow_control_events = db.Column(db.Integer)  # TCP/UDP flow control events
    
    # Application-Layer Metrics
    http_response_codes = db.Column(Text)  # JSON string of response code counts (2xx, 3xx, 4xx, 5xx)
    content_download_time = db.Column(db.Float)  # Time to download full page/resource (ms)
    connection_reuse_ratio = db.Column(db.Float)  # HTTP keep-alive effectiveness (percentage)
    compression_ratio = db.Column(db.Float)  # gzip/deflate compression effectiveness
    certificate_validation_time = db.Column(db.Float)  # SSL/TLS certificate chain validation time (ms)
    
    # Performance Profiling Metrics
    dns_cache_hit_ratio = db.Column(db.Float)  # DNS cache effectiveness (percentage)
    http_cache_hit_ratio = db.Column(db.Float)  # HTTP cache effectiveness (percentage)
    cdn_performance_score = db.Column(db.Float)  # CDN performance rating (0-100)
    multipath_detected = db.Column(db.Boolean)  # MPTCP or ECMP path diversity detected
    application_response_time = db.Column(db.Float)  # End-to-end transaction timing (ms)
    database_query_time = db.Column(db.Float)  # Backend database performance if detectable (ms)
    
    # Infrastructure Monitoring Metrics
    power_consumption_watts = db.Column(db.Float)  # Energy usage in watts (if accessible)
    fan_speeds_rpm = db.Column(Text)  # JSON string of fan speeds by device
    smart_drive_health = db.Column(Text)  # JSON string of SMART health data
    memory_error_rate = db.Column(db.Float)  # ECC memory error rate (errors per hour)
    network_interface_errors = db.Column(Text)  # JSON string of physical layer error rates
    
    # Network interface detection
    network_interface_info = db.Column(Text)  # JSON string of network interface details
    
    # Signal strength monitoring for wireless connections
    signal_strength_min = db.Column(db.Float)  # Minimum signal strength during test (dBm)
    signal_strength_max = db.Column(db.Float)  # Maximum signal strength during test (dBm)
    signal_strength_avg = db.Column(db.Float)  # Average signal strength during test (dBm)
    signal_strength_samples = db.Column(db.Integer)  # Number of signal strength samples collected
    signal_strength_data = db.Column(Text)  # Comma-delimited raw signal strength readings
    
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
            'bandwidth_download': self.bandwidth_download,
            # Advanced network metrics
            'dns_resolution_time': self.dns_resolution_time,
            'tcp_connect_time': self.tcp_connect_time,
            'ssl_handshake_time': self.ssl_handshake_time,
            'ttfb': self.ttfb,
            'jitter': self.jitter,
            # Network interface metrics
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'network_packets_sent': self.network_packets_sent,
            'network_packets_recv': self.network_packets_recv,
            'network_errors_in': self.network_errors_in,
            'network_errors_out': self.network_errors_out,
            'network_drops_in': self.network_drops_in,
            'network_drops_out': self.network_drops_out,
            # Enhanced CPU metrics
            'cpu_load_1min': self.cpu_load_1min,
            'cpu_load_5min': self.cpu_load_5min,
            'cpu_load_15min': self.cpu_load_15min,
            'cpu_cores': self.cpu_cores,
            'cpu_freq_current': self.cpu_freq_current,
            'cpu_context_switches': self.cpu_context_switches,
            'cpu_interrupts': self.cpu_interrupts,
            # Enhanced memory metrics
            'memory_available': self.memory_available,
            'memory_cached': self.memory_cached,
            'memory_buffers': self.memory_buffers,
            'memory_shared': self.memory_shared,
            'swap_total': self.swap_total,
            'swap_used': self.swap_used,
            'swap_percent': self.swap_percent,
            # Enhanced disk metrics
            'disk_read_iops': self.disk_read_iops,
            'disk_write_iops': self.disk_write_iops,
            'disk_read_bytes_sec': self.disk_read_bytes_sec,
            'disk_write_bytes_sec': self.disk_write_bytes_sec,
            'disk_io_util': self.disk_io_util,
            # Process and system metrics
            'process_count': self.process_count,
            'tcp_connections': self.tcp_connections,
            'open_files': self.open_files,
            # Temperature metrics
            'cpu_temperature': self.cpu_temperature,
            'disk_temperature': self.disk_temperature,
            # QoS metrics
            'dscp_value': self.dscp_value,
            'cos_value': self.cos_value,
            'traffic_class': self.traffic_class,
            'qos_policy_compliant': self.qos_policy_compliant,
            'bandwidth_per_class': json.loads(self.bandwidth_per_class) if self.bandwidth_per_class else {},
            # Signal strength monitoring  
            'signal_strength_min': self.signal_strength_min,
            'signal_strength_max': self.signal_strength_max,
            'signal_strength_avg': self.signal_strength_avg,
            'signal_strength_samples': self.signal_strength_samples,
            'signal_strength_data': self.signal_strength_data
        }
