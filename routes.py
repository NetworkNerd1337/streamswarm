from flask import render_template, request, jsonify, redirect, url_for, flash, send_file, Blueprint
from functools import wraps
from flask_login import login_user, logout_user, login_required, current_user
from app import app, db, login_required_with_dev_bypass, require_admin_or_dev_mode
from models import Client, Test, TestResult, TestClient, ApiToken, User, SystemConfig, GnmiDevice, GnmiCertificate
from datetime import datetime, timezone, timedelta
import zoneinfo
import json
import logging
import math
import os
import threading
import time
from pdf_generator import generate_test_report_pdf
from ml_diagnostics import diagnostic_engine
import bleach
import re
import ipaddress

# Import server-side geolocation processor for enhanced path analysis
try:
    from server_geo_processor import geo_processor
    GEO_PROCESSOR_AVAILABLE = True
except ImportError:
    GEO_PROCESSOR_AVAILABLE = False

# ================================
# AUTHENTICATION DECORATORS
# ================================

def admin_required(f):
    """Decorator to require admin role for access with development mode bypass"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if development mode is enabled
        if SystemConfig.is_development_mode():
            # In development mode, bypass admin checks
            return f(*args, **kwargs)
            
        # Normal admin authentication check
        from flask_login import login_required
        @login_required
        @wraps(f)
        def inner(*args, **kwargs):
            if not current_user.is_admin():
                flash('Admin access required.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return inner(*args, **kwargs)
    return decorated_function

def web_auth_required(f):
    """Decorator to require web GUI authentication with development mode bypass"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if development mode is enabled
        if SystemConfig.is_development_mode():
            # In development mode, bypass authentication
            return f(*args, **kwargs)
        
        # Normal authentication check - use Flask-Login's built-in decorator
        if not current_user.is_authenticated:
            # For API endpoints, return JSON error instead of redirect
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            # Store the current URL for redirect after login
            from flask import session
            session['next'] = request.url
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    
    return decorated_function

# Input validation functions
def sanitize_string(text, max_length=255):
    """Sanitize string input to prevent injection attacks"""
    if not text:
        return text
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(text))
    
    # Strip dangerous characters for SQL injection prevention
    text = re.sub(r'[<>&"\'\\;]', '', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()

def validate_hostname(hostname):
    """Validate hostname format"""
    if not hostname:
        return False, "Hostname is required"
    
    hostname = sanitize_string(hostname, 253)
    
    # Check format using regex
    hostname_pattern = re.compile(
        r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    )
    
    if not hostname_pattern.match(hostname):
        return False, "Invalid hostname format"
    
    return True, hostname

def validate_url_destination(destination):
    """Validate URL destination with hostname/IP validation and safe path/parameter handling"""
    if not destination:
        return False, "Destination is required"
    
    # Remove dangerous characters but preserve URL structure
    destination = re.sub(r'[<>&"\'\\;`]', '', destination)
    destination = destination.strip()
    
    # Extract hostname/IP from URL for validation
    url_parts = destination.split('/')
    hostname_part = url_parts[0]
    
    # Handle query parameters in hostname part
    if '?' in hostname_part:
        hostname_part = hostname_part.split('?')[0]
    
    # Validate the hostname/IP part
    hostname_valid, validated_hostname = validate_hostname(hostname_part)
    ip_valid, validated_ip = validate_ip_address(hostname_part)
    
    if not hostname_valid and not ip_valid:
        return False, f"Invalid hostname or IP address: {hostname_part}"
    
    # Reconstruct the destination with validated hostname/IP
    if hostname_valid:
        validated_base = validated_hostname
    else:
        validated_base = validated_ip
    
    # Rebuild the full destination with path and parameters
    if len(url_parts) > 1:
        path_and_params = '/'.join(url_parts[1:])
        # Additional sanitization for path and parameters
        path_and_params = re.sub(r'[<>&"\'\\;`]', '', path_and_params)
        validated_destination = f"{validated_base}/{path_and_params}"
    else:
        validated_destination = validated_base
    
    return True, validated_destination

def validate_ip_address(ip):
    """Validate IP address format"""
    if not ip:
        return False, "IP address is required"
    
    try:
        ipaddress.ip_address(ip)
        return True, ip
    except ValueError:
        return False, "Invalid IP address format"

def validate_json_field(json_str):
    """Validate JSON string field"""
    if not json_str:
        return True, json_str
    
    try:
        parsed = json.loads(json_str)
        return True, json.dumps(parsed)
    except (json.JSONDecodeError, TypeError):
        return False, "Invalid JSON format"

def validate_positive_number(value, field_name="Value"):
    """Validate positive number"""
    try:
        num = float(value)
        if num < 0:
            return False, f"{field_name} must be positive"
        return True, num
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number"

# Rate limiting storage
_rate_limits = {}

def check_rate_limit(client_ip, endpoint, max_requests=100, window_seconds=3600):
    """Check rate limiting"""
    import time
    current_time = time.time()
    key = f"{client_ip}:{endpoint}"
    
    if key not in _rate_limits:
        _rate_limits[key] = []
    
    # Clean old requests
    _rate_limits[key] = [
        req_time for req_time in _rate_limits[key]
        if current_time - req_time < window_seconds
    ]
    
    # Check limit
    if len(_rate_limits[key]) >= max_requests:
        return False
    
    # Add current request
    _rate_limits[key].append(current_time)
    return True

# Authentication decorator for API endpoints
def require_api_token(f):
    """Decorator to require valid API token for client endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication for registration endpoint
        if request.endpoint == 'register_client':
            return f(*args, **kwargs)
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Validate token - must be consumed and active
        api_token = ApiToken.query.filter_by(token=token, status='consumed').first()
        if not api_token:
            return jsonify({'error': 'Invalid or inactive API token'}), 401
        
        # Update last used timestamp
        api_token.update_last_used()
        db.session.commit()
        
        # Add token info to request for use in endpoint
        request.api_token = api_token
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@web_auth_required
def dashboard():
    """Main dashboard view"""
    # Get summary statistics
    total_clients = Client.query.count()
    online_clients = Client.query.filter_by(status='online').count()
    total_tests = Test.query.count()
    active_tests = Test.query.filter_by(status='running').count()
    
    # Get recent test results for charts
    recent_results = TestResult.query.order_by(TestResult.timestamp.desc()).limit(100).all()
    
    return render_template('dashboard.html', 
                         total_clients=total_clients,
                         online_clients=online_clients,
                         total_tests=total_tests,
                         active_tests=active_tests,
                         recent_results=recent_results)

@app.route('/clients')
@web_auth_required
def clients():
    """Client management view"""
    # Load initial batch of clients (most recent 20)
    clients = Client.query.order_by(Client.last_seen.desc()).limit(20).all()
    
    # Mark clients as busy and parse system info
    for client in clients:
        busy_tests = db.session.query(Test).join(TestClient).filter(
            TestClient.client_id == client.id,
            Test.status.in_(['running', 'pending'])
        ).all()
        client.is_busy = len(busy_tests) > 0
        client.active_tests = [test.name for test in busy_tests]
        
        # Parse system info if available
        if client.system_info:
            try:
                import json
                # Handle double-escaped JSON by parsing twice if needed
                parsed_data = json.loads(client.system_info)
                if isinstance(parsed_data, str):
                    # If result is still a string, parse again
                    parsed_data = json.loads(parsed_data)
                client.parsed_system_info = parsed_data
            except (json.JSONDecodeError, TypeError):
                client.parsed_system_info = {}
        else:
            client.parsed_system_info = {}
    
    # Get system configuration for version comparison
    from models import SystemConfig
    config = {
        'expected_client_version': SystemConfig.get_setting('expected_client_version', '1.0.0')
    }
    
    return render_template('clients.html', clients=clients, config=config)

@app.route('/tests')
@web_auth_required
def tests():
    """Test management view"""
    # Load initial batch of tests (most recent 20)
    tests = Test.query.order_by(Test.created_at.desc()).limit(20).all()
    clients = Client.query.filter_by(status='online').all()
    
    # Mark clients as available/busy for test assignment
    for client in clients:
        busy_tests = db.session.query(Test).join(TestClient).filter(
            TestClient.client_id == client.id,
            Test.status.in_(['running', 'pending'])
        ).all()
        client.is_available = len(busy_tests) == 0
        client.active_tests = [test.name for test in busy_tests]
    
    return render_template('tests.html', tests=tests, clients=clients)

@app.route('/test/<int:test_id>')
@web_auth_required
def test_results(test_id):
    """Test results view"""
    test = Test.query.get_or_404(test_id)
    results = TestResult.query.filter_by(test_id=test_id).order_by(TestResult.timestamp.asc()).all()
    clients = db.session.query(Client).join(TestResult).filter(TestResult.test_id == test_id).distinct().all()
    
    # Get results with client information joined for handshake analysis
    handshake_analysis_results = db.session.query(TestResult).join(Client).filter(
        TestResult.test_id == test_id,
        TestResult.tcp_handshake_analysis.isnot(None)
    ).order_by(TestResult.timestamp.desc()).all()
    
    return render_template('test_results.html', 
                         test=test, 
                         results=results, 
                         clients=clients,
                         handshake_analysis_results=handshake_analysis_results)

@app.route('/tutorial')
@web_auth_required
def tutorial():
    """Tutorial and documentation view"""
    return render_template('tutorial.html')

@app.route('/tokens')
@admin_required
def tokens():
    """API Token management view"""
    tokens = ApiToken.query.order_by(ApiToken.created_at.desc()).all()
    return render_template('tokens.html', tokens=tokens)

# Removed standalone metrics dashboard - all metrics now integrated into test results pages

# API Routes

@app.route('/api/client/register', methods=['POST'])
def register_client():
    """Register a new client using an available API token"""
    # Rate limiting
    client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
    if not check_rate_limit(client_ip, 'register', max_requests=10, window_seconds=3600):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    # Get and validate input data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate hostname
    hostname_valid, hostname = validate_hostname(data.get('hostname'))
    if not hostname_valid:
        return jsonify({'error': hostname}), 400
    
    # Validate IP address
    ip_valid, ip_address = validate_ip_address(data.get('ip_address'))
    if not ip_valid:
        return jsonify({'error': ip_address}), 400
    
    # Validate token
    token = sanitize_string(data.get('token'), 64)
    if not token:
        return jsonify({'error': 'API token is required'}), 400
    
    # Validate system info JSON if provided
    system_info = data.get('system_info')
    if system_info:
        json_valid, system_info = validate_json_field(json.dumps(system_info))
        if not json_valid:
            return jsonify({'error': 'Invalid system_info format'}), 400
    
    # Get client version (optional)
    client_version = sanitize_string(data.get('client_version'), 20)
    
    # Check if client already exists
    existing_client = Client.query.filter_by(hostname=hostname, ip_address=ip_address).first()
    
    # Validate token - allow available tokens or consumed tokens belonging to this client
    api_token = ApiToken.query.filter_by(token=token).first()
    if not api_token:
        return jsonify({'error': 'Invalid token'}), 401
    
    # If token is consumed, verify it belongs to the same client
    if api_token.status == 'consumed':
        if not existing_client or api_token.client_id != existing_client.id:
            return jsonify({'error': 'Token already consumed by different client'}), 401
        # Token belongs to this client - allow reconnection
    elif api_token.status != 'available':
        return jsonify({'error': 'Token is not available'}), 401
    
    if existing_client:
        # Update existing client
        existing_client.last_seen = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
        existing_client.status = 'online'
        existing_client.system_info = system_info
        existing_client.client_version = client_version
        client = existing_client
    else:
        # Create new client
        client = Client(
            hostname=hostname,
            ip_address=ip_address,
            status='online',
            last_seen=datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None),
            system_info=system_info,
            client_version=client_version
        )
        db.session.add(client)
        db.session.flush()  # Get client ID
    
    # Consume the token if it's not already consumed
    if api_token.status == 'available':
        api_token.consume(client.id)
    
    db.session.commit()
    
    return jsonify({
        'client_id': client.id,
        'token': token,  # Return token for client to use in future requests
        'status': 'registered' if api_token.status == 'available' else 'reconnected',
        'message': 'Client registered successfully' if existing_client is None else 'Client reconnected successfully'
    })

@app.route('/api/client/<int:client_id>/heartbeat', methods=['POST'])
@require_api_token
def client_heartbeat(client_id):
    """Update client last seen timestamp"""
    client = Client.query.get_or_404(client_id)
    client.last_seen = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    client.status = 'online'
    
    # Update client version if provided in request
    data = request.get_json()
    if data and 'client_version' in data:
        client_version = sanitize_string(data.get('client_version'), 20)
        if client_version:
            client.client_version = client_version
    
    db.session.commit()
    
    return jsonify({'status': 'ok'})

@app.route('/api/client/<int:client_id>/tests', methods=['GET'])
@require_api_token
def get_client_tests(client_id):
    """Get pending tests for a client"""
    # Find tests assigned to this client that are ready to run
    assigned_tests = db.session.query(Test).join(TestClient).filter(
        TestClient.client_id == client_id,
        TestClient.status == 'assigned',
        Test.status.in_(['pending', 'running'])
    ).all()
    
    # Check if any tests should be started
    now = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    ready_tests = []
    
    for test in assigned_tests:
        if test.scheduled_time is None or test.scheduled_time <= now:
            ready_tests.append(test.to_dict())
            
            # Update test status if it's the first time it's being started
            if test.status == 'pending':
                test.status = 'running'
                test.started_at = now
                
            # Update test client status
            test_client = TestClient.query.filter_by(test_id=test.id, client_id=client_id).first()
            if test_client and test_client.status == 'assigned':
                test_client.status = 'running'
    
    db.session.commit()
    
    return jsonify({'tests': ready_tests})

@app.route('/api/test/results', methods=['POST'])
@require_api_token
def submit_test_results():
    """Submit test results from client"""
    # Rate limiting
    client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
    if not check_rate_limit(client_ip, 'test_results', max_requests=1000, window_seconds=3600):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    try:
        client_id = int(data.get('client_id', 0))
        test_id = int(data.get('test_id', 0))
        if client_id <= 0 or test_id <= 0:
            raise ValueError("Invalid ID")
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid client_id or test_id'}), 400
    
    # Validate and sanitize all numeric fields
    def safe_float(value, field_name, min_val=None, max_val=None):
        if value is None:
            return None
        try:
            val = float(value)
            if min_val is not None and val < min_val:
                raise ValueError(f"{field_name} cannot be negative")
            if max_val is not None and val > max_val:
                raise ValueError(f"{field_name} exceeds maximum value")
            return val
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {field_name}")
    
    def safe_int(value, field_name, min_val=None):
        if value is None:
            return None
        try:
            val = int(value)
            if min_val is not None and val < min_val:
                raise ValueError(f"{field_name} cannot be negative")
            return val
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {field_name}")
    
    def safe_json_string(value, field_name):
        if value is None:
            return None
        try:
            if isinstance(value, dict):
                return json.dumps(value)
            elif isinstance(value, str):
                # Try to parse as JSON first
                try:
                    json.loads(value)
                    return value
                except json.JSONDecodeError:
                    # If it's malformed JSON (like Python dict repr), fix it properly
                    try:
                        import ast
                        import re
                        
                        # Fix common Python dict notation issues
                        fixed_value = value
                        
                        # Replace Python literals with JSON equivalents
                        fixed_value = fixed_value.replace('null', 'None')
                        fixed_value = fixed_value.replace('true', 'True') 
                        fixed_value = fixed_value.replace('false', 'False')
                        
                        # Parse as Python literal
                        parsed = ast.literal_eval(fixed_value)
                        
                        # Convert to proper JSON
                        return json.dumps(parsed)
                        
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Invalid format for {field_name}: unable to parse as JSON or Python literal")
            else:
                return json.dumps(value)
        except (TypeError, json.JSONDecodeError):
            raise ValueError(f"Invalid JSON format for {field_name}")
    
    try:
        # Validate all fields
        cpu_percent = safe_float(data.get('cpu_percent'), 'cpu_percent', 0, 100)
        memory_percent = safe_float(data.get('memory_percent'), 'memory_percent', 0, 100)
        memory_used = safe_int(data.get('memory_used'), 'memory_used', 0)
        memory_total = safe_int(data.get('memory_total'), 'memory_total', 0)
        disk_percent = safe_float(data.get('disk_percent'), 'disk_percent', 0, 100)
        disk_used = safe_int(data.get('disk_used'), 'disk_used', 0)
        disk_total = safe_int(data.get('disk_total'), 'disk_total', 0)
        ping_latency = safe_float(data.get('ping_latency'), 'ping_latency', 0)
        ping_packet_loss = safe_float(data.get('ping_packet_loss'), 'ping_packet_loss', 0, 100)
        bandwidth_upload = safe_float(data.get('bandwidth_upload'), 'bandwidth_upload', 0)
        bandwidth_download = safe_float(data.get('bandwidth_download'), 'bandwidth_download', 0)
        
        # Validate JSON fields
        traceroute_data = safe_json_string(data.get('traceroute_data'), 'traceroute_data')
        network_interface_info = safe_json_string(data.get('network_interface_info'), 'network_interface_info')
        top_processes_cpu = safe_json_string(data.get('top_processes_cpu'), 'top_processes_cpu')
        top_processes_memory = safe_json_string(data.get('top_processes_memory'), 'top_processes_memory')
        
        # Validate signal strength fields
        signal_strength_min = safe_float(data.get('signal_strength_min'), 'signal_strength_min')
        signal_strength_max = safe_float(data.get('signal_strength_max'), 'signal_strength_max')
        signal_strength_avg = safe_float(data.get('signal_strength_avg'), 'signal_strength_avg')
        signal_strength_samples = safe_int(data.get('signal_strength_samples'), 'signal_strength_samples', 0)
        signal_strength_data = data.get('signal_strength_data') if data.get('signal_strength_data') else None
        
        # Validate geolocation path analysis fields
        path_geolocation_data = safe_json_string(data.get('path_geolocation_data'), 'path_geolocation_data')
        path_map_html = data.get('path_map_html') if data.get('path_map_html') else None
        path_total_distance_km = safe_float(data.get('path_total_distance_km'), 'path_total_distance_km')
        path_geographic_efficiency = safe_float(data.get('path_geographic_efficiency'), 'path_geographic_efficiency')
        
        result = TestResult(
            test_id=test_id,
            client_id=client_id,
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat())).replace(tzinfo=None),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_percent=disk_percent,
            disk_used=disk_used,
            disk_total=disk_total,
            ping_latency=ping_latency,
            ping_packet_loss=ping_packet_loss,
            traceroute_hops=safe_int(data.get('traceroute_hops'), 'traceroute_hops', 0),
            traceroute_data=traceroute_data,
            bandwidth_upload=bandwidth_upload,
            bandwidth_download=bandwidth_download,
            # Advanced network metrics
            dns_resolution_time=safe_float(data.get('dns_resolution_time'), 'dns_resolution_time', 0),
            tcp_connect_time=safe_float(data.get('tcp_connect_time'), 'tcp_connect_time', 0),
            ssl_handshake_time=safe_float(data.get('ssl_handshake_time'), 'ssl_handshake_time', 0),
            ttfb=safe_float(data.get('ttfb'), 'ttfb', 0),
            jitter=safe_float(data.get('jitter'), 'jitter', 0),
            # Network interface information
            network_interface_info=network_interface_info,
            # Network interface metrics
            network_bytes_sent=safe_int(data.get('network_bytes_sent'), 'network_bytes_sent', 0),
            network_bytes_recv=safe_int(data.get('network_bytes_recv'), 'network_bytes_recv', 0),
            network_packets_sent=safe_int(data.get('network_packets_sent'), 'network_packets_sent', 0),
            network_packets_recv=safe_int(data.get('network_packets_recv'), 'network_packets_recv', 0),
            network_errors_in=safe_int(data.get('network_errors_in'), 'network_errors_in', 0),
            network_errors_out=safe_int(data.get('network_errors_out'), 'network_errors_out', 0),
            network_drops_in=safe_int(data.get('network_drops_in'), 'network_drops_in', 0),
            network_drops_out=safe_int(data.get('network_drops_out'), 'network_drops_out', 0),
            # Enhanced CPU metrics
            cpu_load_1min=safe_float(data.get('cpu_load_1min'), 'cpu_load_1min', 0),
            cpu_load_5min=safe_float(data.get('cpu_load_5min'), 'cpu_load_5min', 0),
            cpu_load_15min=safe_float(data.get('cpu_load_15min'), 'cpu_load_15min', 0),
            cpu_cores=safe_int(data.get('cpu_cores'), 'cpu_cores', 1),
            cpu_freq_current=safe_float(data.get('cpu_freq_current'), 'cpu_freq_current', 0),
            cpu_context_switches=safe_int(data.get('cpu_context_switches'), 'cpu_context_switches', 0),
            cpu_interrupts=safe_int(data.get('cpu_interrupts'), 'cpu_interrupts', 0),
            # Enhanced memory metrics
            memory_available=safe_int(data.get('memory_available'), 'memory_available', 0),
            memory_cached=safe_int(data.get('memory_cached'), 'memory_cached', 0),
            memory_buffers=safe_int(data.get('memory_buffers'), 'memory_buffers', 0),
            memory_shared=safe_int(data.get('memory_shared'), 'memory_shared', 0),
            swap_total=safe_int(data.get('swap_total'), 'swap_total', 0),
            swap_used=safe_int(data.get('swap_used'), 'swap_used', 0),
            swap_percent=safe_float(data.get('swap_percent'), 'swap_percent', 0, 100),
            # Enhanced disk metrics
            disk_read_iops=safe_int(data.get('disk_read_iops'), 'disk_read_iops', 0),
            disk_write_iops=safe_int(data.get('disk_write_iops'), 'disk_write_iops', 0),
            disk_read_bytes_sec=safe_int(data.get('disk_read_bytes_sec'), 'disk_read_bytes_sec', 0),
            disk_write_bytes_sec=safe_int(data.get('disk_write_bytes_sec'), 'disk_write_bytes_sec', 0),
            disk_io_util=safe_float(data.get('disk_io_util'), 'disk_io_util', 0, 100),
            # Process and system metrics
            process_count=safe_int(data.get('process_count'), 'process_count', 0),
            top_processes_cpu=top_processes_cpu,
            top_processes_memory=top_processes_memory,
            tcp_connections=safe_int(data.get('tcp_connections'), 'tcp_connections', 0),
            open_files=safe_int(data.get('open_files'), 'open_files', 0),
            # Temperature metrics
            cpu_temperature=safe_float(data.get('cpu_temperature'), 'cpu_temperature'),
            disk_temperature=safe_float(data.get('disk_temperature'), 'disk_temperature'),
            # QoS metrics
            dscp_value=safe_int(data.get('dscp_value'), 'dscp_value', 0),
            cos_value=safe_int(data.get('cos_value'), 'cos_value', 0),
            traffic_class=sanitize_string(data.get('traffic_class'), 50) if data.get('traffic_class') else None,
            qos_policy_compliant=bool(data.get('qos_policy_compliant')) if data.get('qos_policy_compliant') is not None else None,
            bandwidth_per_class=safe_json_string(data.get('bandwidth_per_class'), 'bandwidth_per_class'),
            # Advanced Network-Level Metrics
            mtu_size=data.get('mtu_size'),
            tcp_window_size=data.get('tcp_window_size'),
            tcp_window_scaling=data.get('tcp_window_scaling'),
            tcp_congestion_window=data.get('tcp_congestion_window'),
            tcp_retransmission_rate=data.get('tcp_retransmission_rate'),
            tcp_out_of_order_packets=data.get('tcp_out_of_order_packets'),
            tcp_duplicate_acks=data.get('tcp_duplicate_acks'),
            # TCP Window Analysis Metrics
            tcp_rtt_min=safe_float(data.get('tcp_rtt_min'), 'tcp_rtt_min'),
            tcp_rtt_max=safe_float(data.get('tcp_rtt_max'), 'tcp_rtt_max'),
            tcp_rtt_avg=safe_float(data.get('tcp_rtt_avg'), 'tcp_rtt_avg'),
            tcp_rtt_variation=safe_float(data.get('tcp_rtt_variation'), 'tcp_rtt_variation'),
            tcp_cwnd_min=safe_int(data.get('tcp_cwnd_min'), 'tcp_cwnd_min'),
            tcp_cwnd_max=safe_int(data.get('tcp_cwnd_max'), 'tcp_cwnd_max'),
            tcp_cwnd_avg=safe_float(data.get('tcp_cwnd_avg'), 'tcp_cwnd_avg'),
            tcp_ssthresh_avg=safe_float(data.get('tcp_ssthresh_avg'), 'tcp_ssthresh_avg'),
            tcp_congestion_events=safe_int(data.get('tcp_congestion_events'), 'tcp_congestion_events'),
            tcp_retransmissions=safe_int(data.get('tcp_retransmissions'), 'tcp_retransmissions'),
            tcp_window_efficiency=safe_float(data.get('tcp_window_efficiency'), 'tcp_window_efficiency'),
            tcp_bottleneck_type=sanitize_string(data.get('tcp_bottleneck_type'), 50) if data.get('tcp_bottleneck_type') else None,
            tcp_window_timeline=safe_json_string(data.get('tcp_window_timeline'), 'tcp_window_timeline'),
            # Advanced QoS Metrics
            per_dscp_latency=data.get('per_dscp_latency'),
            traffic_policing_detected=data.get('traffic_policing_detected'),
            queue_depth=data.get('queue_depth'),
            ecn_capable=data.get('ecn_capable'),
            ecn_congestion_experienced=data.get('ecn_congestion_experienced'),
            flow_control_events=data.get('flow_control_events'),
            # Application-Layer Metrics
            http_response_codes=data.get('http_response_codes'),
            content_download_time=data.get('content_download_time'),
            connection_reuse_ratio=data.get('connection_reuse_ratio'),
            compression_ratio=data.get('compression_ratio'),
            certificate_validation_time=data.get('certificate_validation_time'),
            # Performance Profiling Metrics
            dns_cache_hit_ratio=data.get('dns_cache_hit_ratio'),
            http_cache_hit_ratio=data.get('http_cache_hit_ratio'),
            cdn_performance_score=data.get('cdn_performance_score'),
            multipath_detected=data.get('multipath_detected'),
            application_response_time=data.get('application_response_time'),
            database_query_time=data.get('database_query_time'),
            # Infrastructure Monitoring Metrics
            power_consumption_watts=data.get('power_consumption_watts'),
            fan_speeds_rpm=data.get('fan_speeds_rpm'),
            smart_drive_health=data.get('smart_drive_health'),
            memory_error_rate=data.get('memory_error_rate'),
            network_interface_errors=data.get('network_interface_errors'),
            # Signal strength monitoring
            signal_strength_min=signal_strength_min,
            signal_strength_max=signal_strength_max,
            signal_strength_avg=signal_strength_avg,
            signal_strength_samples=signal_strength_samples,
            signal_strength_data=signal_strength_data,
            # TCP Handshake Timing Analysis
            tcp_handshake_total_time=safe_float(data.get('tcp_handshake_total_time'), 'tcp_handshake_total_time'),
            tcp_handshake_syn_time=safe_float(data.get('tcp_handshake_syn_time'), 'tcp_handshake_syn_time'),
            tcp_handshake_synack_time=safe_float(data.get('tcp_handshake_synack_time'), 'tcp_handshake_synack_time'),
            tcp_handshake_ack_time=safe_float(data.get('tcp_handshake_ack_time'), 'tcp_handshake_ack_time'),
            tcp_handshake_network_delay=safe_float(data.get('tcp_handshake_network_delay'), 'tcp_handshake_network_delay'),
            tcp_handshake_server_processing=safe_float(data.get('tcp_handshake_server_processing'), 'tcp_handshake_server_processing'),
            tcp_handshake_analysis=sanitize_string(data.get('tcp_handshake_analysis'), 500) if data.get('tcp_handshake_analysis') else None,
            # Geolocation path analysis for enhanced traceroute visualization
            path_geolocation_data=path_geolocation_data,
            path_map_html=path_map_html,
            path_total_distance_km=path_total_distance_km,
            path_geographic_efficiency=path_geographic_efficiency
        )
        
    except ValueError as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    
    try:
        db.session.add(result)
        db.session.commit()
        
        # Check if test should be marked as completed
        test = Test.query.get(result.test_id)
        if test and test.started_at:
            # Ensure both datetimes are timezone-naive for comparison (database stores naive datetimes in Eastern Time)
            if test.started_at.tzinfo is None:
                test_start_time = test.started_at
            else:
                test_start_time = test.started_at.replace(tzinfo=None)
            
            elapsed_time = (datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - test_start_time).total_seconds()
            if elapsed_time >= test.duration:
                test.status = 'completed'
                test.completed_at = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
                
                # Mark test client as completed
                test_client = TestClient.query.filter_by(test_id=test.id, client_id=result.client_id).first()
                if test_client:
                    test_client.status = 'completed'
                
                # Handle recurring test logic for "Create New Tests" mode
                _handle_recurring_test_completion(test)
        
        return jsonify({'status': 'success', 'message': 'Results submitted successfully'})
        
    except Exception as e:
        logging.error(f"Error submitting test results: {str(e)}")
        return jsonify({'error': 'Failed to submit results'}), 500

@app.route('/api/test/create', methods=['POST'])
def create_test():
    """Create a new test"""
    # Rate limiting
    client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
    if not check_rate_limit(client_ip, 'create_test', max_requests=50, window_seconds=3600):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    name = sanitize_string(data.get('name'), 255)
    if not name:
        return jsonify({'error': 'Test name is required'}), 400
    
    destination = sanitize_string(data.get('destination'), 255)
    if not destination:
        return jsonify({'error': 'Test destination is required'}), 400
    
    # Validate optional fields
    description = sanitize_string(data.get('description', ''), 1000)
    
    try:
        duration = int(data.get('duration', 300))
        if duration <= 0 or duration > 86400:  # Max 24 hours
            return jsonify({'error': 'Duration must be between 1 and 86400 seconds'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid duration format'}), 400
    
    try:
        interval = int(data.get('interval', 5))
        if interval <= 0 or interval > 3600:  # Max 1 hour
            return jsonify({'error': 'Interval must be between 1 and 3600 seconds'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid interval format'}), 400
    
    try:
        packet_size = int(data.get('packet_size', 64))
        if packet_size <= 0 or packet_size > 65535:  # Max UDP packet size
            return jsonify({'error': 'Packet size must be between 1 and 65535 bytes'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid packet size format'}), 400
    
    # Validate client IDs
    client_ids = data.get('client_ids', [])
    if not isinstance(client_ids, list):
        return jsonify({'error': 'client_ids must be a list'}), 400
    
    validated_client_ids = []
    for client_id in client_ids:
        try:
            validated_id = int(client_id)
            if validated_id <= 0:
                raise ValueError("Invalid client ID")
            validated_client_ids.append(validated_id)
        except (ValueError, TypeError):
            return jsonify({'error': f'Invalid client ID: {client_id}'}), 400
    
    try:
        # Clean up destination by removing protocol prefixes
        destination = destination.strip()
        if destination.startswith('http://'):
            destination = destination[7:]
        elif destination.startswith('https://'):
            destination = destination[8:]
        
        # Remove trailing slash if present
        if destination.endswith('/'):
            destination = destination[:-1]
        
        # Validate destination format (URL, hostname, or IP)
        dest_valid, validated_destination = validate_url_destination(destination)
        if not dest_valid:
            return jsonify({'error': f'Invalid destination: {validated_destination}'}), 400
        
        destination = validated_destination
        
        # Handle recurrence settings
        is_recurring = data.get('is_recurring', False)
        recurrence_interval = data.get('recurrence_interval') if is_recurring else None
        recurrence_type = data.get('recurrence_type', 'continue') if is_recurring else 'continue'
        
        # Validate recurrence if enabled
        if is_recurring:
            if not recurrence_interval:
                return jsonify({'error': 'Recurrence interval is required when recurring is enabled'}), 400
            
            # Validate recurrence type
            if recurrence_type not in ['continue', 'new']:
                return jsonify({'error': 'Recurrence type must be either "continue" or "new"'}), 400
            
            # Minimum recurrence interval: test duration + 10 minute buffer
            min_interval = duration + (10 * 60)  # 10 minutes buffer
            if recurrence_interval < min_interval:
                min_minutes = math.ceil(min_interval / 60)
                return jsonify({'error': f'Recurrence interval must be at least {min_minutes} minutes (test duration + 10 minute buffer)'}), 400
        
        # Calculate next execution time if recurring
        scheduled_time = None
        next_execution = None
        if data.get('scheduled_time'):
            scheduled_time = datetime.fromisoformat(data['scheduled_time']).replace(tzinfo=None)
            if is_recurring:
                next_execution = scheduled_time + timedelta(seconds=recurrence_interval)
        elif is_recurring:
            # If no scheduled time but recurring, set next execution for the interval from now
            next_execution = datetime.now() + timedelta(seconds=recurrence_interval)
        
        test = Test(
            name=name,
            description=description,
            destination=destination,
            scheduled_time=scheduled_time,
            duration=duration,
            interval=interval,
            packet_size=packet_size,
            is_recurring=is_recurring,
            recurrence_interval=recurrence_interval,
            recurrence_type=recurrence_type,
            next_execution=next_execution
        )
        
        db.session.add(test)
        db.session.flush()  # Get the test ID
        
        # Assign clients to test
        for client_id in validated_client_ids:
            test_client = TestClient(test_id=test.id, client_id=client_id)
            db.session.add(test_client)
        
        db.session.commit()
        
        return jsonify({'status': 'success', 'test_id': test.id, 'message': 'Test created successfully'})
        
    except Exception as e:
        logging.error(f"Error creating test: {str(e)}")
        return jsonify({'error': 'Failed to create test'}), 500

@app.route('/api/tests/status', methods=['POST'])
def get_tests_status():
    """Get current status for multiple tests"""
    data = request.get_json()
    if not data or 'test_ids' not in data:
        return jsonify({'error': 'test_ids required'}), 400
    
    test_ids = data['test_ids']
    if not isinstance(test_ids, list):
        return jsonify({'error': 'test_ids must be a list'}), 400
    
    try:
        tests = Test.query.filter(Test.id.in_(test_ids)).all()
        test_statuses = []
        
        for test in tests:
            test_statuses.append({
                'id': test.id,
                'status': test.status,
                'progress': calculate_test_progress(test)
            })
        
        return jsonify({
            'status': 'success',
            'tests': test_statuses
        })
        
    except Exception as e:
        logging.error(f"Error fetching test statuses: {str(e)}")
        return jsonify({'error': 'Failed to fetch test statuses'}), 500

def calculate_test_progress(test):
    """Calculate test progress percentage"""
    if test.status == 'completed':
        return 100
    elif test.status == 'failed':
        return 0
    elif test.status == 'running':
        if test.started_at and test.duration:
            from datetime import datetime
            elapsed = (datetime.now() - test.started_at).total_seconds()
            progress = min(100, int((elapsed / test.duration) * 100))
            return progress
        return 0
    else:  # pending
        return 0

@app.route('/api/tests/bulk-delete', methods=['DELETE'])
@web_auth_required
def bulk_delete_tests():
    """Delete multiple tests at once"""
    data = request.get_json()
    if not data or 'test_ids' not in data:
        return jsonify({'error': 'test_ids required'}), 400
    
    test_ids = data['test_ids']
    if not isinstance(test_ids, list) or not test_ids:
        return jsonify({'error': 'test_ids must be a non-empty list'}), 400
    
    # Validate that all test_ids are integers
    try:
        test_ids = [int(test_id) for test_id in test_ids]
    except (ValueError, TypeError):
        return jsonify({'error': 'All test_ids must be valid integers'}), 400
    
    try:
        deleted_count = 0
        errors = []
        
        for test_id in test_ids:
            test = Test.query.get(test_id)
            if not test:
                errors.append(f"Test {test_id} not found")
                continue
            
            try:
                # Delete all test results
                TestResult.query.filter_by(test_id=test_id).delete()
                
                # Delete all test client assignments
                TestClient.query.filter_by(test_id=test_id).delete()
                
                # Delete the test itself
                db.session.delete(test)
                db.session.commit()
                deleted_count += 1
                
            except Exception as e:
                db.session.rollback()
                errors.append(f"Failed to delete test {test_id}: {str(e)}")
                continue
        
        if deleted_count > 0:
            message = f"Successfully deleted {deleted_count} test{'s' if deleted_count > 1 else ''}"
            if errors:
                message += f". {len(errors)} error{'s' if len(errors) > 1 else ''} occurred."
            
            return jsonify({
                'status': 'success',
                'message': message,
                'deleted_count': deleted_count,
                'errors': errors
            })
        else:
            return jsonify({
                'error': 'No tests were deleted',
                'errors': errors
            }), 400
            
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error in bulk delete: {str(e)}")
        return jsonify({'error': 'Failed to delete tests'}), 500

@app.route('/api/test/<int:test_id>/restart', methods=['POST'])
@web_auth_required
def restart_test(test_id):
    """Restart a test with its original parameters"""
    test = Test.query.get_or_404(test_id)
    
    # Check if test is currently running
    if test.status == 'running':
        return jsonify({'error': 'Cannot restart a test that is currently running'}), 400
    
    try:
        # Reset test status and timing for immediate execution using consistent timezone
        test.status = 'pending'
        # Set scheduled time to 5 seconds ago using the same timezone as client polling
        now = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
        test.scheduled_time = now - timedelta(seconds=5)
        test.started_at = None
        test.completed_at = None
        test.recurrence_interval = None  # Clear recurring schedule so it runs once immediately
        
        # Reset all client assignments to 'assigned' status
        test_clients = TestClient.query.filter_by(test_id=test_id).all()
        for test_client in test_clients:
            test_client.status = 'assigned'
        
        # Clear any previous test results to start fresh
        TestResult.query.filter_by(test_id=test_id).delete()
        
        db.session.commit()
        
        logging.info(f"Test {test_id} restarted by user")
        
        return jsonify({
            'status': 'success',
            'message': 'Test restarted successfully',
            'test_id': test_id
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error restarting test {test_id}: {str(e)}")
        return jsonify({'error': 'Failed to restart test'}), 500

@app.route('/api/test/<int:test_id>/data', methods=['GET'])
def get_test_data(test_id):
    """Get test data for charts"""
    results = TestResult.query.filter_by(test_id=test_id).order_by(TestResult.timestamp.asc()).all()
    
    data = {
        'timestamps': [],
        'clients': {},
        'metrics': {
            'cpu': {},
            'memory': {},
            'disk': {},
            'ping_latency': {},
            'ping_packet_loss': {},
            'jitter': {},
            'dns_resolution_time': {},
            'tcp_connect_time': {},
            'ssl_handshake_time': {},
            'ttfb': {},
            'cpu_load_1min': {},
            'cpu_load_5min': {},
            'cpu_load_15min': {},
            'network_bytes_sent': {},
            'network_bytes_recv': {},
            'network_errors_in': {},
            'network_errors_out': {},
            'network_drops_in': {},
            'network_drops_out': {},
            'network_packets_sent': {},
            'network_packets_recv': {},
            'disk_read_iops': {},
            'disk_write_iops': {},
            'disk_read_bytes_sec': {},
            'disk_write_bytes_sec': {},
            'bandwidth_upload': {},
            'bandwidth_download': {},
            'cpu_cores': {},
            'cpu_freq_current': {},
            'cpu_temperature': {},
            'memory_available': {},
            'memory_cached': {},
            'memory_buffers': {},
            'swap_used': {},
            'swap_percent': {},
            'process_count': {},
            'tcp_connections': {},
            'traceroute_hops': {},
            'dscp_value': {},
            'cos_value': {},
            'qos_policy_compliant': {},
            'traceroute_data': {},
            'path_map_html': {},
            'path_total_distance_km': {},
            'path_geographic_efficiency': {}
        }
    }
    
    for result in results:
        timestamp = result.timestamp.isoformat()
        client_id = str(result.client_id)
        
        if timestamp not in data['timestamps']:
            data['timestamps'].append(timestamp)
        
        if client_id not in data['clients']:
            client = Client.query.get(result.client_id)
            data['clients'][client_id] = client.hostname if client else f'Client {client_id}'
        
        # Initialize client data in metrics if not exists
        for metric in data['metrics']:
            if client_id not in data['metrics'][metric]:
                data['metrics'][metric][client_id] = []
        
        # Add metric values
        data['metrics']['cpu'][client_id].append({
            'x': timestamp,
            'y': result.cpu_percent
        })
        data['metrics']['memory'][client_id].append({
            'x': timestamp,
            'y': result.memory_percent
        })
        data['metrics']['disk'][client_id].append({
            'x': timestamp,
            'y': result.disk_percent
        })
        data['metrics']['ping_latency'][client_id].append({
            'x': timestamp,
            'y': result.ping_latency
        })
        data['metrics']['ping_packet_loss'][client_id].append({
            'x': timestamp,
            'y': result.ping_packet_loss
        })
        data['metrics']['jitter'][client_id].append({
            'x': timestamp,
            'y': result.jitter
        })
        data['metrics']['dns_resolution_time'][client_id].append({
            'x': timestamp,
            'y': result.dns_resolution_time
        })
        data['metrics']['tcp_connect_time'][client_id].append({
            'x': timestamp,
            'y': result.tcp_connect_time
        })
        data['metrics']['ssl_handshake_time'][client_id].append({
            'x': timestamp,
            'y': result.ssl_handshake_time
        })
        data['metrics']['ttfb'][client_id].append({
            'x': timestamp,
            'y': result.ttfb
        })
        data['metrics']['cpu_load_1min'][client_id].append({
            'x': timestamp,
            'y': result.cpu_load_1min
        })
        data['metrics']['cpu_load_5min'][client_id].append({
            'x': timestamp,
            'y': result.cpu_load_5min
        })
        data['metrics']['cpu_load_15min'][client_id].append({
            'x': timestamp,
            'y': result.cpu_load_15min
        })
        data['metrics']['network_bytes_sent'][client_id].append({
            'x': timestamp,
            'y': result.network_bytes_sent
        })
        data['metrics']['network_bytes_recv'][client_id].append({
            'x': timestamp,
            'y': result.network_bytes_recv
        })
        data['metrics']['network_errors_in'][client_id].append({
            'x': timestamp,
            'y': result.network_errors_in
        })
        data['metrics']['network_errors_out'][client_id].append({
            'x': timestamp,
            'y': result.network_errors_out
        })
        data['metrics']['network_drops_in'][client_id].append({
            'x': timestamp,
            'y': result.network_drops_in
        })
        data['metrics']['network_drops_out'][client_id].append({
            'x': timestamp,
            'y': result.network_drops_out
        })
        data['metrics']['network_packets_sent'][client_id].append({
            'x': timestamp,
            'y': result.network_packets_sent
        })
        data['metrics']['network_packets_recv'][client_id].append({
            'x': timestamp,
            'y': result.network_packets_recv
        })
        data['metrics']['disk_read_iops'][client_id].append({
            'x': timestamp,
            'y': result.disk_read_iops
        })
        data['metrics']['disk_write_iops'][client_id].append({
            'x': timestamp,
            'y': result.disk_write_iops
        })
        data['metrics']['disk_read_bytes_sec'][client_id].append({
            'x': timestamp,
            'y': result.disk_read_bytes_sec
        })
        data['metrics']['disk_write_bytes_sec'][client_id].append({
            'x': timestamp,
            'y': result.disk_write_bytes_sec
        })
        data['metrics']['dscp_value'][client_id].append({
            'x': timestamp,
            'y': result.dscp_value
        })
        data['metrics']['cos_value'][client_id].append({
            'x': timestamp,
            'y': result.cos_value
        })
        data['metrics']['qos_policy_compliant'][client_id].append({
            'x': timestamp,
            'y': 1 if result.qos_policy_compliant else 0 if result.qos_policy_compliant is False else None
        })
        data['metrics']['traceroute_data'][client_id].append({
            'x': timestamp,
            'y': result.traceroute_data
        })
        data['metrics']['path_map_html'][client_id].append({
            'x': timestamp,
            'y': result.path_map_html
        })
        data['metrics']['path_total_distance_km'][client_id].append({
            'x': timestamp,
            'y': result.path_total_distance_km
        })
        data['metrics']['path_geographic_efficiency'][client_id].append({
            'x': timestamp,
            'y': result.path_geographic_efficiency
        })
        data['metrics']['bandwidth_upload'][client_id].append({
            'x': timestamp,
            'y': result.bandwidth_upload
        })
        data['metrics']['bandwidth_download'][client_id].append({
            'x': timestamp,
            'y': result.bandwidth_download
        })
        data['metrics']['cpu_cores'][client_id].append({
            'x': timestamp,
            'y': result.cpu_cores
        })
        data['metrics']['cpu_freq_current'][client_id].append({
            'x': timestamp,
            'y': result.cpu_freq_current
        })
        data['metrics']['cpu_temperature'][client_id].append({
            'x': timestamp,
            'y': result.cpu_temperature
        })
        data['metrics']['memory_available'][client_id].append({
            'x': timestamp,
            'y': result.memory_available / (1024**3) if result.memory_available else None  # Convert bytes to GB
        })
        data['metrics']['memory_cached'][client_id].append({
            'x': timestamp,
            'y': result.memory_cached / (1024**3) if result.memory_cached else None  # Convert bytes to GB
        })
        data['metrics']['memory_buffers'][client_id].append({
            'x': timestamp,
            'y': result.memory_buffers / (1024**3) if result.memory_buffers else None  # Convert bytes to GB
        })
        data['metrics']['swap_used'][client_id].append({
            'x': timestamp,
            'y': result.swap_used / (1024**3) if result.swap_used else None  # Convert bytes to GB
        })
        data['metrics']['swap_percent'][client_id].append({
            'x': timestamp,
            'y': result.swap_percent
        })
        data['metrics']['process_count'][client_id].append({
            'x': timestamp,
            'y': result.process_count
        })
        data['metrics']['tcp_connections'][client_id].append({
            'x': timestamp,
            'y': result.tcp_connections
        })
        data['metrics']['traceroute_hops'][client_id].append({
            'x': timestamp,
            'y': result.traceroute_hops
        })
    
    return jsonify(data)

@app.route('/api/client/<int:client_id>/details', methods=['GET'])
def get_client_details(client_id):
    """Get detailed client information"""
    client = Client.query.get_or_404(client_id)
    
    # Get recent test results for this client
    recent_results = TestResult.query.filter_by(client_id=client_id).order_by(TestResult.timestamp.desc()).limit(10).all()
    
    # Calculate uptime (time since client was created)
    if client.created_at:
        # Ensure both datetimes are timezone-naive for comparison (database stores naive datetimes in Eastern Time)
        if client.created_at.tzinfo is None:
            client_start_time = client.created_at
        else:
            client_start_time = client.created_at.replace(tzinfo=None)
        
        uptime_seconds = (datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - client_start_time).total_seconds()
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime = f"{uptime_hours}h {uptime_minutes}m"
    else:
        uptime = "Unknown"
    
    # Get latest metrics
    latest_result = TestResult.query.filter_by(client_id=client_id).order_by(TestResult.timestamp.desc()).first()
    
    metrics = {
        'cpu_percent': latest_result.cpu_percent if latest_result else None,
        'memory_percent': latest_result.memory_percent if latest_result else None,
        'disk_percent': latest_result.disk_percent if latest_result else None
    }
    
    # Get test history
    test_history = []
    for result in recent_results:
        test = Test.query.get(result.test_id)
        test_history.append({
            'test_name': test.name if test else 'Unknown',
            'timestamp': result.timestamp.isoformat(),
            'ping_latency': result.ping_latency,
            'ping_packet_loss': result.ping_packet_loss
        })
    
    return jsonify({
        'client': client.to_dict(),
        'uptime': uptime,
        'metrics': metrics,
        'test_history': test_history
    })

@app.route('/api/test/<int:test_id>/status', methods=['GET'])
def get_test_status(test_id):
    """Get current test status"""
    test = Test.query.get_or_404(test_id)
    return jsonify({
        'id': test.id,
        'status': test.status,
        'started_at': test.started_at.isoformat() if test.started_at else None,
        'completed_at': test.completed_at.isoformat() if test.completed_at else None
    })

@app.route('/api/test/<int:test_id>/stop', methods=['POST'])
def stop_test(test_id):
    """Stop a running test"""
    test = Test.query.get_or_404(test_id)
    
    if test.status not in ['running', 'pending']:
        return jsonify({'error': 'Test is not running'}), 400
    
    test.status = 'completed'
    test.completed_at = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    
    # Update all test clients to completed status
    TestClient.query.filter_by(test_id=test_id, status='running').update({'status': 'completed'})
    
    # Handle recurring test logic for "Create New Tests" mode
    _handle_recurring_test_completion(test)
    
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Test stopped successfully'})

@app.route('/api/test/<int:test_id>/delete', methods=['DELETE'])
def delete_test(test_id):
    """Delete a test and all its associated data"""
    test = Test.query.get_or_404(test_id)
    
    try:
        # Delete all test results
        TestResult.query.filter_by(test_id=test_id).delete()
        
        # Delete all test client assignments
        TestClient.query.filter_by(test_id=test_id).delete()
        
        # Delete the test itself
        db.session.delete(test)
        db.session.commit()
        
        return jsonify({'status': 'success', 'message': 'Test deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting test: {str(e)}")
        return jsonify({'error': 'Failed to delete test'}), 500

@app.route('/api/test/<int:test_id>/progress', methods=['GET'])
def get_test_progress(test_id):
    """Get real-time test progress"""
    test = Test.query.get_or_404(test_id)
    
    if test.status not in ['running', 'completed']:
        return jsonify({'progress': 0, 'status': test.status})
    
    if test.status == 'completed':
        return jsonify({'progress': 100, 'status': 'completed'})
    
    if test.started_at:
        # Ensure both datetimes are timezone-naive for comparison (database stores naive datetimes in Eastern Time)
        if test.started_at.tzinfo is None:
            test_start_time = test.started_at
        else:
            test_start_time = test.started_at.replace(tzinfo=None)
        
        elapsed_time = (datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - test_start_time).total_seconds()
        progress = min((elapsed_time / test.duration) * 100, 100)
        
        # Auto-complete if duration exceeded
        if progress >= 100 and test.status == 'running':
            test.status = 'completed'
            test.completed_at = datetime.now(zoneinfo.ZoneInfo('America/New_York'))
            TestClient.query.filter_by(test_id=test_id, status='running').update({'status': 'completed'})
            
            # Handle recurring test logic for "Create New Tests" mode
            _handle_recurring_test_completion(test)
            
            db.session.commit()
            
        return jsonify({
            'progress': round(progress, 1),
            'status': test.status,
            'elapsed_time': int(elapsed_time),
            'remaining_time': max(0, test.duration - int(elapsed_time))
        })
    
    return jsonify({'progress': 0, 'status': test.status})

@app.route('/api/download/linux-startup-script', methods=['GET'])
def download_linux_startup_script():
    """Download the Linux startup script for StreamSwarm clients"""
    try:
        # Path to the startup script in scripts folder
        script_path = os.path.join(os.getcwd(), 'scripts', 'start_streamswarm_client.sh')
        
        # Check if file exists
        if not os.path.exists(script_path):
            return jsonify({'error': 'Startup script not found'}), 404
        
        # Send file with appropriate headers
        return send_file(
            script_path,
            as_attachment=True,
            download_name='start_streamswarm_client.sh',
            mimetype='application/x-sh'
        )
        
    except Exception as e:
        logging.error(f"Error downloading startup script: {str(e)}")
        return jsonify({'error': 'Failed to download startup script'}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics"""
    # Mark clients as offline if they haven't been seen in 5 minutes
    offline_threshold = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - timedelta(minutes=5)
    Client.query.filter(Client.last_seen < offline_threshold).update({'status': 'offline'})
    db.session.commit()
    
    total_clients = Client.query.count()
    online_clients = Client.query.filter_by(status='online').count()
    total_tests = Test.query.count()
    active_tests = Test.query.filter_by(status='running').count()
    
    # Get recent activity
    recent_results = TestResult.query.order_by(TestResult.timestamp.desc()).limit(10).all()
    recent_activity = []
    
    for result in recent_results:
        client = Client.query.get(result.client_id)
        test = Test.query.get(result.test_id)
        recent_activity.append({
            'timestamp': result.timestamp.isoformat(),
            'client_name': client.hostname if client else 'Unknown',
            'test_name': test.name if test else 'Unknown',
            'ping_latency': result.ping_latency
        })
    
    return jsonify({
        'total_clients': total_clients,
        'online_clients': online_clients,
        'total_tests': total_tests,
        'active_tests': active_tests,
        'recent_activity': recent_activity
    })

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get aggregated dashboard metrics for charts"""
    from sqlalchemy import func
    
    # Get recent results from last 24 hours
    cutoff_time = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - timedelta(hours=24)
    recent_results = TestResult.query.filter(TestResult.timestamp >= cutoff_time).all()
    
    if not recent_results:
        return jsonify({
            'latency_data': [],
            'avg_cpu': 0,
            'avg_memory': 0,
            'avg_disk': 0,
            'metrics_count': 0
        })
    
    # Calculate aggregated metrics
    latencies = [r.ping_latency for r in recent_results if r.ping_latency is not None]
    cpu_values = [r.cpu_percent for r in recent_results if r.cpu_percent is not None]
    memory_values = [r.memory_percent for r in recent_results if r.memory_percent is not None]
    disk_values = [r.disk_percent for r in recent_results if r.disk_percent is not None]
    
    # Prepare latency time series data (last 50 points)
    latency_data = []
    for result in recent_results[-50:]:
        if result.ping_latency is not None:
            latency_data.append({
                'x': result.timestamp.isoformat(),
                'y': result.ping_latency
            })
    
    # Calculate averages
    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
    avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
    avg_disk = sum(disk_values) / len(disk_values) if disk_values else 0
    
    return jsonify({
        'latency_data': latency_data,
        'avg_cpu': round(avg_cpu, 1),
        'avg_memory': round(avg_memory, 1),
        'avg_disk': round(avg_disk, 1),
        'metrics_count': len(recent_results)
    })

@app.route('/api/test/<int:test_id>/export/pdf')
def export_test_pdf(test_id):
    """Export test results as executive PDF report"""
    try:
        test = Test.query.get_or_404(test_id)
        
        # Generate PDF report
        pdf_filename = f"test_{test_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join('static', 'reports', pdf_filename)
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        # Generate the PDF
        generated_path = generate_test_report_pdf(test_id, pdf_path)
        
        # Send the file for download
        return send_file(
            generated_path,
            as_attachment=True,
            download_name=f"{test.name.replace(' ', '_')}_Performance_Report.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logging.error(f"PDF generation failed for test {test_id}: {str(e)}")
        return jsonify({'error': f'Failed to generate PDF report: {str(e)}'}), 500

@app.route('/api/client/<int:client_id>', methods=['DELETE'])
def delete_client(client_id):
    """Delete a client (only if offline) while preserving historical data"""
    try:
        client = Client.query.get_or_404(client_id)
        
        # Security check: only allow deletion of offline clients
        if client.status == 'online':
            return jsonify({'error': 'Cannot delete online client. Client must be offline first.'}), 400
        
        # Check for API token references
        api_token_count = ApiToken.query.filter_by(client_id=client_id).count()
        
        # Check if client has test data
        result_count = TestResult.query.filter_by(client_id=client_id).count()
        test_client_count = TestClient.query.filter_by(client_id=client_id).count()
        
        # Store client info for logging
        client_hostname = client.hostname
        
        if api_token_count > 0:
            # API tokens must be deleted first
            return jsonify({
                'error': f'Cannot delete client "{client_hostname}" because it has {api_token_count} API token(s) assigned. Please delete the API token(s) first in the Token Management section, then try deleting the client again.'
            }), 400
        
        if result_count > 0 or test_client_count > 0:
            # Cannot delete client with existing test data due to foreign key constraints
            return jsonify({
                'error': f'Cannot delete client "{client_hostname}" because it has {result_count} test results and {test_client_count} test assignments. Historical data must be preserved.'
            }), 400
        
        # Only delete clients with no test data to maintain referential integrity
        db.session.delete(client)
        db.session.commit()
        
        logging.info(f"Client {client_hostname} (ID: {client_id}) deleted successfully. No test data was associated with this client.")
        
        return jsonify({
            'status': 'success',
            'message': f'Client "{client_hostname}" deleted successfully'
        })
        
    except Exception as e:
        logging.error(f"Error deleting client {client_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete client'}), 500

# API Token Management Endpoints
@app.route('/api/tokens', methods=['GET'])
@admin_required
def get_tokens():
    """Get all API tokens with optional filtering"""
    # Validate query parameters
    status_filter = sanitize_string(request.args.get('status'), 20)
    
    query = ApiToken.query
    if status_filter and status_filter in ['available', 'consumed', 'revoked']:
        query = query.filter_by(status=status_filter)
    
    tokens = query.order_by(ApiToken.created_at.desc()).all()
    
    return jsonify({
        'tokens': [token.to_dict() for token in tokens],
        'total': len(tokens)
    })

@app.route('/api/tokens', methods=['POST'])
@admin_required
def create_token():
    """Create a new API token"""
    # Rate limiting
    client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
    if not check_rate_limit(client_ip, 'create_token', max_requests=20, window_seconds=3600):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate token name
    name = sanitize_string(data.get('name'), 255)
    if not name:
        return jsonify({'error': 'Token name is required'}), 400
    
    # Validate description
    description = sanitize_string(data.get('description', ''), 1000)
    
    # Check for duplicate names
    existing = ApiToken.query.filter_by(name=name).first()
    if existing:
        return jsonify({'error': 'Token name already exists'}), 400
    
    try:
        token = ApiToken(
            name=name,
            description=description
        )
        
        db.session.add(token)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Token created successfully',
            'token': token.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating token: {str(e)}")
        return jsonify({'error': 'Failed to create token'}), 500

@app.route('/api/tokens/<int:token_id>', methods=['PUT'])
@admin_required
def update_token(token_id):
    """Update an API token"""
    # Rate limiting
    client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
    if not check_rate_limit(client_ip, 'update_token', max_requests=50, window_seconds=3600):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    token = ApiToken.query.get_or_404(token_id)
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    try:
        if 'name' in data:
            # Validate and sanitize name
            name = sanitize_string(data['name'], 255)
            if not name:
                return jsonify({'error': 'Token name cannot be empty'}), 400
            
            # Check for duplicate names (excluding current token)
            existing = ApiToken.query.filter(
                ApiToken.name == name,
                ApiToken.id != token_id
            ).first()
            if existing:
                return jsonify({'error': 'Token name already exists'}), 400
            token.name = name
        
        if 'description' in data:
            # Validate and sanitize description
            description = sanitize_string(data['description'], 1000)
            token.description = description
        
        if 'status' in data:
            # Validate status value
            status = sanitize_string(data['status'], 20)
            if status and status in ['available', 'consumed', 'revoked']:
                token.status = status
            else:
                return jsonify({'error': 'Invalid status value'}), 400
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Token updated successfully',
            'token': token.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating token: {str(e)}")
        return jsonify({'error': 'Failed to update token'}), 500

@app.route('/api/tokens/<int:token_id>', methods=['DELETE'])
@admin_required
def delete_token(token_id):
    """Delete an API token"""
    token = ApiToken.query.get_or_404(token_id)
    
    try:
        db.session.delete(token)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Token deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting token: {str(e)}")
        return jsonify({'error': 'Failed to delete token'}), 500

@app.route('/api/tokens/<int:token_id>/regenerate', methods=['POST'])
@admin_required
def regenerate_token(token_id):
    """Regenerate an API token"""
    token = ApiToken.query.get_or_404(token_id)
    
    if token.status == 'consumed':
        return jsonify({'error': 'Cannot regenerate consumed token'}), 400
    
    try:
        token.token = ApiToken.generate_token()
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Token regenerated successfully',
            'token': token.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error regenerating token: {str(e)}")
        return jsonify({'error': 'Failed to regenerate token'}), 500

# ML Diagnostics Routes

@app.route('/diagnose/<int:test_id>')
@web_auth_required
def diagnose_test(test_id):
    """ML Diagnostic analysis page for test results"""
    test = Test.query.get_or_404(test_id)
    
    if test.status not in ['completed', 'failed']:
        flash('Can only diagnose completed or failed tests', 'warning')
        return redirect(url_for('test_results', test_id=test_id))
    
    # Get basic test info for the page
    results_count = TestResult.query.filter_by(test_id=test_id).count()
    
    return render_template('ml_diagnosis.html', 
                         test=test, 
                         results_count=results_count)

def clean_diagnosis_data(data):
    """Clean diagnosis data for JSON serialization by handling NaN values"""
    import math
    
    if isinstance(data, dict):
        return {k: clean_diagnosis_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_diagnosis_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data

@app.route('/api/test/<int:test_id>/diagnose', methods=['POST'])
@web_auth_required
def api_diagnose_test(test_id):
    """API endpoint to run ML diagnosis on test results"""
    try:
        diagnosis = diagnostic_engine.diagnose_test(test_id)
        # Clean the data to handle NaN values that can't be JSON serialized
        clean_diagnosis = clean_diagnosis_data(diagnosis)
        return jsonify(clean_diagnosis)
    except Exception as e:
        logging.error(f"Error running diagnosis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ml-models')
@admin_required
def ml_models():
    """ML Model management page"""
    model_status = diagnostic_engine.get_model_status()
    return render_template('ml_models.html', model_status=model_status)

@app.route('/api/ml-models/train', methods=['POST'])
@admin_required
def train_ml_models():
    """Train ML models with available data"""
    try:
        success = diagnostic_engine.train_models()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Models trained successfully'
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Not enough data for training. Need at least 50 samples.'
            }), 400
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml-models/status')
@admin_required
def ml_models_status():
    """Get ML model status"""
    try:
        status = diagnostic_engine.get_model_status()
        return jsonify(status)
    except Exception as e:
        logging.error(f"Error getting model status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ================================
# PREDICTIVE ANALYTICS API
# ================================

@app.route('/api/predict-performance', methods=['POST'])
@web_auth_required
def predict_performance():
    """Predict network performance before running tests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No configuration provided'}), 400
        
        test_config = data.get('test_config', {})
        current_conditions = data.get('current_conditions', {})
        
        prediction = diagnostic_engine.predict_performance(test_config, current_conditions)
        return jsonify(prediction)
        
    except Exception as e:
        logging.error(f"Error predicting performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-failure', methods=['POST'])
@web_auth_required
def predict_network_failure():
    """Predict network failure probability based on destination and conditions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No configuration provided'}), 400
        
        destination = data.get('destination')
        prediction_horizon = data.get('prediction_horizon', 24)
        current_conditions = data.get('current_conditions', 'normal')
        
        if not destination:
            return jsonify({'error': 'Destination is required'}), 400
        
        prediction = diagnostic_engine.predict_network_failure(destination, prediction_horizon, current_conditions)
        return jsonify(prediction)
        
    except Exception as e:
        logging.error(f"Error predicting network failure: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/capacity-trends')
@web_auth_required
def capacity_trends():
    """Get network capacity trends for planning"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        if days_back < 1 or days_back > 365:
            return jsonify({'error': 'Days must be between 1 and 365'}), 400
        
        trends = diagnostic_engine.analyze_capacity_trends(days_back)
        return jsonify(trends)
        
    except Exception as e:
        logging.error(f"Error analyzing capacity trends: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predictive-analytics')
@web_auth_required
def predictive_analytics():
    """Predictive Analytics dashboard page"""
    return render_template('predictive_analytics.html')

@app.route('/api/qos_compliance_analysis')
@login_required_with_dev_bypass
def api_qos_compliance_analysis():
    """API endpoint for QoS compliance analysis"""
    try:
        test_id = request.args.get('test_id', type=int)
        destination = request.args.get('destination', type=str)
        
        # Perform QoS compliance analysis
        analysis_result = diagnostic_engine.analyze_qos_compliance(test_id=test_id, destination=destination)
        
        return jsonify(analysis_result)
    except Exception as e:
        logging.error(f"Error in QoS compliance analysis: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/analyze-client-infrastructure', methods=['POST'])
@web_auth_required
def analyze_client_infrastructure():
    """Analyze client infrastructure correlation and provide improvement recommendations"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        client_id = data.get('client_id')
        days_back = data.get('days_back', 30)  # Default to 30 days
        
        if not client_id:
            return jsonify({'error': 'Client ID is required'}), 400
        
        # Validate client_id is integer
        try:
            client_id = int(client_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid client ID format'}), 400
        
        # Validate days_back
        try:
            days_back = int(days_back)
            if days_back < 1 or days_back > 365:
                days_back = 30  # Default fallback
        except (ValueError, TypeError):
            days_back = 30
        
        # Check if client exists
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': f'Client {client_id} not found'}), 404
        
        # Perform analysis
        analysis_result = diagnostic_engine.analyze_client_infrastructure_correlation(client_id, days_back)
        
        # Debug logging
        import logging
        logging.info(f"Route handler - analysis_result type: {type(analysis_result)}")
        logging.info(f"Route handler - analysis_result value: {str(analysis_result)[:200]}...")
        
        # Add client information to response
        logging.info(f"About to check analysis_result.get('status'): {type(analysis_result)}")
        try:
            status = analysis_result.get('status')
            logging.info(f"Successfully got status: {status}")
        except Exception as status_error:
            logging.error(f"Error getting status: {status_error}")
            logging.error(f"analysis_result type at error: {type(analysis_result)}")
            raise status_error
        
        if status == 'success':
            logging.info("Entering success block")
            # Parse system_info JSON to get platform information - use fresh import to avoid namespace conflicts
            import json as json_module
            parsed_system_data = {}
            if client.system_info:
                logging.info(f"Raw client.system_info type: {type(client.system_info)}")
                logging.info(f"Raw client.system_info value: {client.system_info}")
                try:
                    if isinstance(client.system_info, str):
                        logging.info("About to call json_module.loads()")
                        # Create a test dict to validate JSON parsing works
                        test_dict = json_module.loads('{"test": "value"}')
                        logging.info(f"Test JSON parsing result: {type(test_dict)} - {test_dict}")
                        
                        # Now try with actual data using fresh import
                        first_parse = json_module.loads(client.system_info)
                        logging.info(f"First parse result type: {type(first_parse)}")
                        logging.info(f"First parse result: {first_parse}")
                        
                        # Check if first parse gives us a dict or string
                        if isinstance(first_parse, dict):
                            logging.info("First parse successfully returned dictionary")
                            parsed_system_data = first_parse
                        elif isinstance(first_parse, str):
                            logging.info("First parse returned string - attempting double decode")
                            # This might be double-encoded JSON
                            second_parse = json_module.loads(first_parse)
                            logging.info(f"Second parse result type: {type(second_parse)}")
                            logging.info(f"Second parse result: {second_parse}")
                            parsed_system_data = second_parse
                        else:
                            logging.error(f"Unexpected first parse type: {type(first_parse)}")
                            parsed_system_data = {}
                    else:
                        logging.info("Using system_info as-is (not a string)")
                        parsed_system_data = client.system_info
                except (json_module.JSONDecodeError, TypeError) as json_error:
                    logging.error(f"Failed to parse system_info JSON: {json_error}")
                    logging.error(f"client.system_info type: {type(client.system_info)}")
                    logging.error(f"client.system_info value: {client.system_info}")
                    parsed_system_data = {}
            
            logging.info(f"Final parsed_system_data type: {type(parsed_system_data)}")
            logging.info(f"Final parsed_system_data value: {parsed_system_data}")
            
            logging.info("About to modify analysis_result dict")
            logging.info(f"analysis_result type before modification: {type(analysis_result)}")
            
            try:
                logging.info(f"client.hostname: {client.hostname}")
                logging.info(f"parsed_system_data: {parsed_system_data}")
                logging.info(f"client.client_version: {client.client_version}")
                
                client_info_dict = {
                    'hostname': client.hostname,
                    'platform': parsed_system_data.get('platform', 'Unknown'),
                    'client_version': client.client_version
                }
                logging.info(f"Created client_info_dict: {client_info_dict}")
                
                analysis_result['client_info'] = client_info_dict
                logging.info("Successfully added client_info to analysis_result")
            except Exception as modify_error:
                logging.error(f"Error modifying analysis_result: {modify_error}")
                logging.error(f"analysis_result type at modification error: {type(analysis_result)}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                raise modify_error
        
        return jsonify(analysis_result)
    except Exception as e:
        logging.error(f"Error in client infrastructure analysis: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'status': 'error'
        }), 500

# ================================
# AUTHENTICATION SYSTEM
# ================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for web GUI access with robust error handling"""
    # Handle malformed URL parameters gracefully
    try:
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.warning(f"Login page access error: {str(e)}")
        # Clear any problematic session data and redirect to clean login
        from flask import session
        session.clear()
        return redirect(url_for('login'))
    
    # Check for session timeout message
    if request.args.get('timeout') == '1':
        flash('Your session has expired due to inactivity. Please log in again.', 'warning')
    
    try:
        if request.method == 'POST':
            username = sanitize_string(request.form.get('username', '').strip(), 80)
            password = request.form.get('password', '')
            
            if not username or not password:
                flash('Username and password are required.', 'danger')
                return render_template('login.html')
            
            # Case-insensitive username lookup
            user = User.query.filter(User.username.ilike(username)).first()
            
            if user and user.check_password(password) and user.is_active:
                login_user(user, remember=True)
                user.update_last_login()
                
                # Initialize session activity tracking
                from datetime import datetime
                import zoneinfo
                from flask import session
                session['last_activity'] = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None).isoformat()
                
                # Safely handle next parameter
                next_page = request.args.get('next', '').strip()
                flash(f'Welcome back, {user.username}!', 'success')
                
                # Validate and sanitize next URL
                if next_page and next_page.startswith('/') and len(next_page) < 200:
                    try:
                        # Test if the URL is valid by creating a response (but don't send it)
                        from urllib.parse import unquote
                        clean_next = unquote(next_page)
                        if clean_next.startswith('/') and not clean_next.startswith('//'):
                            return redirect(clean_next)
                    except:
                        pass
                
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
        
        return render_template('login.html')
    
    except Exception as e:
        logging.error(f"Login function error: {str(e)}")
        # For any unexpected errors, show a graceful message and render clean login page
        flash('An error occurred while processing your request. Please try again.', 'warning')
        return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout from web GUI"""
    try:
        username = current_user.username if hasattr(current_user, 'username') else 'unknown'
        logout_user()
        session.clear()  # Clear all session data including last_activity
        flash('You have been logged out.', 'info')
        logging.info(f"User {username} logged out successfully")
    except Exception as e:
        logging.error(f"Error during logout: {str(e)}")
        flash('Logout completed.', 'info')
    
    return redirect(url_for('login'))

# Session timeout debugging endpoint (admin only)
@app.route('/api/session-debug')
@admin_required
def session_debug():
    """Debug session timeout functionality - admin only"""
    try:
        from models import SystemConfig
        from datetime import datetime
        import zoneinfo
        
        current_time = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
        timeout_minutes = SystemConfig.get_session_timeout_minutes()
        last_activity = session.get('last_activity')
        
        response_data = {
            'current_time': current_time.isoformat(),
            'timeout_minutes': timeout_minutes,
            'timeout_enabled': timeout_minutes > 0,
            'last_activity': last_activity,
            'session_keys': list(session.keys()),
            'user_authenticated': current_user.is_authenticated,
            'user_id': current_user.id if hasattr(current_user, 'id') else None,
            'username': current_user.username if hasattr(current_user, 'username') else None
        }
        
        if last_activity:
            try:
                last_activity_time = datetime.fromisoformat(last_activity)
                time_diff = current_time - last_activity_time
                response_data.update({
                    'last_activity_parsed': last_activity_time.isoformat(),
                    'time_since_activity_seconds': int(time_diff.total_seconds()),
                    'time_since_activity_minutes': time_diff.total_seconds() / 60,
                    'should_timeout': timeout_minutes > 0 and time_diff.total_seconds() > (timeout_minutes * 60)
                })
            except (ValueError, TypeError) as e:
                response_data['last_activity_error'] = str(e)
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Session debug error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/user-management')
@admin_required
def user_management():
    """Admin-only user management page"""
    users = User.query.all()
    return render_template('user_management.html', users=users)

@app.route('/api/users', methods=['POST'])
@admin_required
def create_user():
    """Create a new user - admin only"""
    try:
        data = request.get_json()
        
        username = sanitize_string(data.get('username', '').strip(), 80)
        email = sanitize_string(data.get('email', '').strip(), 120)
        password = data.get('password', '')
        role = sanitize_string(data.get('role', 'user').strip(), 20)
        
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        if role not in ['user', 'admin']:
            role = 'user'
        
        # Validate email format
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if username or email already exists (case-insensitive username check)
        existing_user = User.query.filter((User.username.ilike(username)) | (User.email == email)).first()
        if existing_user:
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Create new user
        user = User(
            username=username,
            email=email,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating user: {str(e)}")
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    """Update user - admin only"""
    try:
        user = User.query.get_or_404(user_id)
        data = request.get_json()
        
        # Don't allow admin to modify themselves
        if user.id == current_user.id:
            return jsonify({'error': 'Cannot modify your own account'}), 403
        
        if 'username' in data:
            username = sanitize_string(data['username'].strip(), 80)
            if username.lower() != user.username.lower():
                # Case-insensitive username uniqueness check
                existing = User.query.filter(User.username.ilike(username)).first()
                if existing:
                    return jsonify({'error': 'Username already exists'}), 400
                user.username = username
        
        if 'email' in data:
            email = sanitize_string(data['email'].strip(), 120)
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return jsonify({'error': 'Invalid email format'}), 400
            if email != user.email:
                existing = User.query.filter_by(email=email).first()
                if existing:
                    return jsonify({'error': 'Email already exists'}), 400
                user.email = email
        
        if 'role' in data:
            role = sanitize_string(data['role'].strip(), 20)
            if role in ['user', 'admin']:
                user.role = role
        
        if 'active' in data:
            user.is_active = bool(data['active'])
        
        if 'password' in data and data['password']:
            user.set_password(data['password'])
        
        db.session.commit()
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating user: {str(e)}")
        return jsonify({'error': 'Failed to update user'}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete user - admin only"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Don't allow admin to delete themselves
        if user.id == current_user.id:
            return jsonify({'error': 'Cannot delete your own account'}), 403
        
        # Don't delete the last admin
        if user.role == 'admin':
            admin_count = User.query.filter_by(role='admin').count()
            if admin_count <= 1:
                return jsonify({'error': 'Cannot delete the last admin user'}), 400
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'User deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting user: {str(e)}")
        return jsonify({'error': 'Failed to delete user'}), 500

@app.route('/api/users')
@admin_required
def list_users():
    """List all users - admin only"""
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
    except Exception as e:
        logging.error(f"Error listing users: {str(e)}")
        return jsonify({'error': 'Failed to retrieve users'}), 500

@app.route('/profile')
@web_auth_required
def user_profile():
    """User profile page for changing own password"""
    return render_template('user_profile.html')

@app.route('/api/profile/change-password', methods=['POST'])
@web_auth_required
def change_own_password():
    """Allow user to change their own password"""
    try:
        data = request.get_json()
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        if not current_password or not new_password or not confirm_password:
            return jsonify({'error': 'All password fields are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'New passwords do not match'}), 400
        
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters long'}), 400
        
        # Verify current password
        if not current_user.check_password(current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Update password
        current_user.set_password(new_password)
        db.session.commit()
        
        return jsonify({'message': 'Password changed successfully'})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error changing password: {str(e)}")
        return jsonify({'error': 'Failed to change password'}), 500

# ================================
# SYSTEM CONFIGURATION
# ================================

@app.route('/system-configuration')
@admin_required
def system_configuration():
    """System configuration dashboard - admin only"""
    from models import SystemConfig
    
    # Get all system configuration settings and management options
    configurations = {
        'development_mode': {
            'current_value': SystemConfig.is_development_mode(),
            'description': 'Temporarily disable authentication for development',
            'type': 'boolean'
        },
        'expected_client_version': {
            'current_value': SystemConfig.get_setting('expected_client_version', '1.0.0'),
            'description': 'Version that clients should be running for deployment tracking',
            'type': 'text'
        },
        'session_timeout': {
            'current_value': SystemConfig.get_session_timeout_minutes(),
            'description': 'Automatically log out users after inactive period (minutes)',
            'type': 'slider',
            'min': 0,
            'max': 60,
            'options': ['Disabled', '10min', '15min', '30min', '45min', '60min']
        },
        'user_management': {
            'description': 'Manage web GUI user accounts and permissions',
            'type': 'management'
        },
        'api_token_management': {
            'description': 'Manage API tokens for client authentication',
            'type': 'management'
        },
        'ai_model_configuration': {
            'description': 'Configure and train machine learning models',
            'type': 'management'
        }
    }
    
    return render_template('system_configuration.html', configurations=configurations)

# ================================
# DEVELOPMENT MODE CONTROLS
# ================================

@app.route('/dev-mode')
@admin_required
def dev_mode_settings():
    """Development mode settings page - admin only"""
    from models import SystemConfig
    current_mode = SystemConfig.is_development_mode()
    return render_template('dev_mode.html', dev_mode_enabled=current_mode)

@app.route('/api/dev-mode/toggle', methods=['POST'])
@admin_required
def toggle_dev_mode():
    """Toggle development mode - admin only"""
    try:
        from models import SystemConfig
        current_mode = SystemConfig.is_development_mode()
        new_mode = not current_mode
        
        SystemConfig.set_setting(
            'development_mode', 
            'true' if new_mode else 'false',
            'Temporarily disable authentication for development purposes'
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Development mode {"enabled" if new_mode else "disabled"}',
            'dev_mode_enabled': new_mode
        })
        
    except Exception as e:
        logging.error(f"Error toggling development mode: {str(e)}")
        return jsonify({'error': 'Failed to toggle development mode'}), 500

@app.route('/api/dev-mode/status')
def dev_mode_status():
    """Get development mode status - available to all"""
    try:
        from models import SystemConfig
        is_enabled = SystemConfig.is_development_mode()
        return jsonify({
            'dev_mode_enabled': is_enabled,
            'message': 'Development mode is ' + ('enabled' if is_enabled else 'disabled')
        })
    except Exception as e:
        logging.error(f"Error getting dev mode status: {str(e)}")
        return jsonify({'error': 'Failed to get development mode status'}), 500

@app.route('/api/update-system-config', methods=['POST'])
@admin_required
def update_system_config():
    """Update system configuration setting - admin only"""
    try:
        from models import SystemConfig
        data = request.get_json()
        
        key = data.get('key')
        value = data.get('value')
        
        if not key or value is None:
            return jsonify({'error': 'Key and value are required'}), 400
        
        # Validate specific configuration keys
        if key == 'expected_client_version':
            # Validate version format
            import re
            if not re.match(r'^[0-9]+\.[0-9]+\.[0-9]+$', value):
                return jsonify({'error': 'Version must be in format x.y.z (e.g., 1.0.0)'}), 400
            
            SystemConfig.set_setting(
                key, 
                value,
                'Version that clients should be running for deployment tracking'
            )
            
            return jsonify({
                'success': True,
                'message': f'Expected client version updated to {value}',
                'key': key,
                'value': value
            })
        
        else:
            return jsonify({'error': f'Unknown configuration key: {key}'}), 400
        
    except Exception as e:
        logging.error(f"Error updating system configuration: {str(e)}")
        return jsonify({'error': 'Failed to update system configuration'}), 500

@app.route('/api/session-timeout', methods=['POST'])
@admin_required
def update_session_timeout():
    """Update session timeout setting - admin only"""
    try:
        from models import SystemConfig
        data = request.get_json()
        
        timeout_minutes = data.get('timeout_minutes')
        if timeout_minutes is None:
            return jsonify({'error': 'Timeout value is required'}), 400
        
        try:
            timeout_value = int(timeout_minutes)
        except (ValueError, TypeError):
            return jsonify({'error': 'Timeout must be a valid number'}), 400
        
        if timeout_value < 0 or (timeout_value > 0 and timeout_value < 5) or timeout_value > 480:
            return jsonify({'error': 'Timeout must be 0 (disabled) or between 5 and 480 minutes'}), 400
        
        SystemConfig.set_setting(
            'session_timeout_minutes', 
            str(timeout_value),
            'Automatically log out users after inactive period (minutes)'
        )
        
        message = 'Session timeout disabled' if timeout_value == 0 else f'Session timeout updated to {timeout_value} minutes'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'timeout_minutes': timeout_value
        })
        
    except Exception as e:
        logging.error(f"Error updating session timeout: {str(e)}")
        return jsonify({'error': 'Failed to update session timeout'}), 500

# ================================
# Tests API Routes for Infinite Scroll
# ================================

@app.route('/api/tests', methods=['GET'])
@web_auth_required
def api_tests():
    """Get paginated tests for infinite scroll with optional search"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    per_page = 20
    offset = (page - 1) * per_page
    
    # Build query with optional search filter
    query = Test.query
    
    if search:
        # Search across multiple fields using case-insensitive LIKE
        search_filter = f"%{search}%"
        query = query.filter(
            Test.name.ilike(search_filter) |
            Test.description.ilike(search_filter) |
            Test.destination.ilike(search_filter) |
            Test.status.ilike(search_filter)
        )
    
    tests = query.order_by(Test.created_at.desc()).offset(offset).limit(per_page).all()
    
    test_data = []
    for test in tests:
        # Calculate progress for running tests
        progress = 0
        if test.status == 'running' and test.started_at:
            if test.started_at.tzinfo is None:
                test_start_time = test.started_at
            else:
                test_start_time = test.started_at.replace(tzinfo=None)
            
            elapsed_time = (datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None) - test_start_time).total_seconds()
            progress = min((elapsed_time / test.duration) * 100, 100)
        elif test.status == 'completed':
            progress = 100
            
        test_data.append({
            'id': test.id,
            'name': test.name,
            'description': test.description,
            'destination': test.destination,
            'duration': test.duration,
            'interval': test.interval,
            'status': test.status,
            'progress': round(progress, 1),
            'scheduled_time': test.scheduled_time.strftime('%Y-%m-%d %H:%M') if test.scheduled_time else None,
            'created_at': test.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    has_more = len(tests) == per_page
    
    return jsonify({
        'tests': test_data,
        'has_more': has_more,
        'page': page,
        'search': search
    })

@app.route('/api/clients', methods=['GET'])
@web_auth_required
def api_clients():
    """Get paginated clients for infinite scroll with optional search"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    per_page = 20
    offset = (page - 1) * per_page
    
    # Build query with optional search filter
    query = Client.query
    
    if search:
        # Search across multiple fields using case-insensitive LIKE
        search_filter = f"%{search}%"
        query = query.filter(
            Client.hostname.ilike(search_filter) |
            Client.ip_address.ilike(search_filter) |
            Client.status.ilike(search_filter) |
            Client.system_info.ilike(search_filter)
        )
    
    clients = query.order_by(Client.last_seen.desc()).offset(offset).limit(per_page).all()
    
    client_data = []
    for client in clients:
        # Mark clients as busy if they're assigned to running tests
        busy_tests = db.session.query(Test).join(TestClient).filter(
            TestClient.client_id == client.id,
            Test.status.in_(['running', 'pending'])
        ).all()
        
        # Parse system info if available
        system_info = {}
        if client.system_info:
            try:
                import json
                # Handle double-escaped JSON by parsing twice if needed
                parsed_data = json.loads(client.system_info)
                if isinstance(parsed_data, str):
                    # If result is still a string, parse again
                    parsed_data = json.loads(parsed_data)
                system_info = parsed_data
            except Exception as e:
                system_info = {}
        
        client_data.append({
            'id': client.id,
            'hostname': client.hostname,
            'ip_address': client.ip_address,
            'status': client.status,
            'system_info': system_info,
            'client_version': client.client_version,
            'last_seen': client.last_seen.strftime('%Y-%m-%d %H:%M:%S') if client.last_seen else None,
            'is_busy': len(busy_tests) > 0,
            'active_tests': [test.name for test in busy_tests],
            'created_at': client.created_at.strftime('%Y-%m-%d %H:%M:%S') if client.created_at else None
        })
    
    has_more = len(clients) == per_page
    
    return jsonify({
        'clients': client_data,
        'has_more': has_more,
        'page': page,
        'search': search
    })

@app.route('/api/test/<int:test_id>/process-geolocation', methods=['POST'])
@web_auth_required
def process_test_geolocation(test_id):
    """Process geolocation data for test results that don't have map data yet"""
    if not GEO_PROCESSOR_AVAILABLE:
        return jsonify({'error': 'Server-side geolocation processor not available'}), 503
    
    try:
        # Check if test exists
        test = Test.query.get_or_404(test_id)
        
        # Find results with traceroute data but no geolocation data
        pending_results = TestResult.query.filter(
            TestResult.test_id == test_id,
            TestResult.traceroute_data.isnot(None),
            TestResult.path_geolocation_data.is_(None)
        ).all()
        
        if not pending_results:
            return jsonify({
                'status': 'success',
                'message': 'No results require geolocation processing',
                'processed_count': 0
            })
        
        # Process results in a background thread to avoid timeout
        def process_async():
            with app.app_context():
                processed_count = 0
                for result in pending_results:
                    try:
                        # Refresh the result object in the current session
                        result = db.session.merge(result)
                        if geo_processor._process_single_result(result):
                            processed_count += 1
                            # Commit after each successful result to ensure data is saved
                            db.session.commit()
                    except Exception as e:
                        logging.error(f"Error processing result {result.id}: {str(e)}")
                        db.session.rollback()
                        continue
                
                logging.info(f"Processed geolocation data for {processed_count} results in test {test_id}")
        
        # Start background processing
        thread = threading.Thread(target=process_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'message': f'Started processing geolocation data for {len(pending_results)} results',
            'pending_count': len(pending_results)
        })
        
    except Exception as e:
        logging.error(f"Error processing geolocation for test {test_id}: {str(e)}")
        return jsonify({'error': 'Failed to process geolocation data'}), 500

@app.route('/api/process-all-geolocation', methods=['POST'])
@admin_required
def process_all_geolocation():
    """Process geolocation data for all test results that don't have map data yet"""
    if not GEO_PROCESSOR_AVAILABLE:
        return jsonify({'error': 'Server-side geolocation processor not available'}), 503
    
    try:
        # Start background processing
        def process_async():
            with app.app_context():
                processed_count = geo_processor.process_pending_results()
                logging.info(f"Completed batch geolocation processing: {processed_count} results processed")
        
        thread = threading.Thread(target=process_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'message': 'Started batch geolocation processing for all pending results'
        })
        
    except Exception as e:
        logging.error(f"Error starting batch geolocation processing: {str(e)}")
        return jsonify({'error': 'Failed to start geolocation processing'}), 500

def _handle_recurring_test_completion(completed_test):
    """Handle the completion of a test that might be part of a recurring series"""
    
    # SCENARIO 1: Original recurring test completes - create first child test
    if completed_test.is_recurring and completed_test.recurrence_type == 'new' and not completed_test.parent_test_id:
        # This is the original recurring test completing - create first child
        _create_next_recurring_test(completed_test)
    
    # SCENARIO 2: Child test completes - create next child test
    elif completed_test.parent_test_id:
        # Find the original recurring test
        original_test = Test.query.get(completed_test.parent_test_id)
        if original_test and original_test.is_recurring and original_test.recurrence_type == 'new':
            # Create the next test in the series
            _create_next_recurring_test(original_test)

def _create_next_recurring_test(original_test):
    """Create the next test in a recurring series immediately"""
    from datetime import datetime
    import zoneinfo
    
    def get_eastern_time():
        """Get current time in Eastern timezone"""
        return datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    
    # Schedule new test 2 minutes in the future to allow previous test cleanup
    scheduled_time = get_eastern_time() + timedelta(minutes=2)
    
    # Create new test with same settings
    new_test = Test(
        name=original_test.name,
        description=original_test.description,
        destination=original_test.destination,
        scheduled_time=scheduled_time,  # Schedule 2 minutes in future
        duration=original_test.duration,
        interval=original_test.interval,
        packet_size=original_test.packet_size,
        test_config=original_test.test_config,
        status='pending',
        is_recurring=False,  # New test is not recurring itself
        recurrence_interval=None,
        recurrence_type='continue',
        parent_test_id=original_test.id,  # Link to original recurring test
        next_execution=None
    )
    
    db.session.add(new_test)
    db.session.flush()  # Get the new test ID
    
    # Copy client assignments from original test
    original_clients = db.session.query(TestClient).filter_by(test_id=original_test.id).all()
    for original_client in original_clients:
        new_test_client = TestClient(
            test_id=new_test.id,
            client_id=original_client.client_id,
            status='assigned'
        )
        db.session.add(new_test_client)
    
    db.session.commit()
    
    logging.info(f"Created next recurring test {new_test.id} from completed test with parent {original_test.id}")
    
    return new_test


# ================================
# GNMI DEVICE MANAGEMENT ROUTES
# ================================

@app.route('/gnmi-manager')
@admin_required
def gnmi_manager():
    """GNMI Device Manager page"""
    devices = GnmiDevice.query.order_by(GnmiDevice.created_at.desc()).all()
    return render_template('gnmi_manager.html', devices=devices)

@app.route('/api/gnmi/devices', methods=['GET'])
@admin_required
def get_gnmi_devices():
    """Get all GNMI devices for client synchronization"""
    try:
        devices = GnmiDevice.query.filter_by(enabled=True).all()
        return jsonify({
            'success': True,
            'devices': [device.to_dict() for device in devices]
        })
    except Exception as e:
        logging.error(f"Error getting GNMI devices: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to fetch devices'}), 500

@app.route('/api/gnmi/devices', methods=['POST'])
@admin_required
def create_gnmi_device():
    """Create a new GNMI device"""
    try:
        # Get form data
        name = request.form.get('name')
        ip_address = request.form.get('ip_address')
        port = int(request.form.get('port', 830))
        auth_method = request.form.get('auth_method', 'password')
        username = request.form.get('username')
        password = request.form.get('password')
        description = request.form.get('description')
        enabled = request.form.get('enabled') == 'on'
        
        logging.info(f"Creating GNMI device: name={name}, ip={ip_address}, port={port}, auth_method={auth_method}, enabled={enabled}")
        
        # Validate required fields
        if not name or not ip_address:
            return jsonify({'success': False, 'message': 'Name and IP address are required'}), 400
        
        # Check for duplicate device names
        existing_device = GnmiDevice.query.filter_by(name=name).first()
        if existing_device:
            return jsonify({'success': False, 'message': f'Device name "{name}" already exists'}), 400
        
        # Validate IP address
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid IP address format'}), 400
        
        # Validate authentication based on method
        if auth_method == 'password' and (not username or not password):
            return jsonify({'success': False, 'message': 'Username and password required for password authentication'}), 400
        
        # For certificate authentication, validate that required certificates are provided
        if auth_method in ['certificate', 'cert_username']:
            if 'client_cert' not in request.files or not request.files['client_cert'].filename:
                return jsonify({'success': False, 'message': 'Client certificate file is required for certificate authentication'}), 400
            if 'client_key' not in request.files or not request.files['client_key'].filename:
                return jsonify({'success': False, 'message': 'Client private key file is required for certificate authentication'}), 400
        
        # Create device
        device = GnmiDevice(
            name=name,
            ip_address=ip_address,
            port=port,
            auth_method=auth_method,
            username=username if auth_method != 'certificate' else None,
            password=password if auth_method == 'password' else None,
            description=description,
            enabled=enabled,
            created_by=current_user.id if current_user.is_authenticated else None
        )
        
        db.session.add(device)
        db.session.flush()  # Get the device ID
        
        # Handle certificate uploads
        if auth_method in ['certificate', 'cert_username']:
            cert_files = ['client_cert', 'client_key', 'ca_cert']
            cert_types = ['client_cert', 'client_key', 'ca_cert']
            
            for cert_file, cert_type in zip(cert_files, cert_types):
                if cert_file in request.files and request.files[cert_file].filename:
                    file = request.files[cert_file]
                    certificate = GnmiCertificate(
                        device_id=device.id,
                        cert_type=cert_type,
                        filename=file.filename,
                        content=file.read(),
                        uploaded_by=current_user.id if current_user.is_authenticated else None
                    )
                    db.session.add(certificate)
        
        db.session.commit()
        
        logging.info(f"GNMI device '{name}' created by user {current_user.username if current_user.is_authenticated else 'system'}")
        
        return jsonify({
            'success': True,
            'message': 'Device created successfully',
            'device': device.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating GNMI device: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to create device'}), 500

@app.route('/api/gnmi/devices/<int:device_id>', methods=['GET'])
@admin_required
def get_gnmi_device(device_id):
    """Get a specific GNMI device"""
    try:
        device = GnmiDevice.query.get_or_404(device_id)
        return jsonify({
            'success': True,
            'device': device.to_dict()
        })
    except Exception as e:
        logging.error(f"Error getting GNMI device {device_id}: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to fetch device'}), 500

@app.route('/api/gnmi/devices/<int:device_id>', methods=['PUT'])
@admin_required
def update_gnmi_device(device_id):
    """Update a GNMI device"""
    try:
        device = GnmiDevice.query.get_or_404(device_id)
        
        # Update basic fields
        device.name = request.form.get('name', device.name)
        device.ip_address = request.form.get('ip_address', device.ip_address)
        device.port = int(request.form.get('port', device.port))
        device.auth_method = request.form.get('auth_method', device.auth_method)
        device.username = request.form.get('username') if device.auth_method != 'certificate' else None
        device.description = request.form.get('description', device.description)
        device.enabled = request.form.get('enabled') == 'on'
        
        # Update password only if provided
        password = request.form.get('password')
        if password and device.auth_method == 'password':
            device.password = password
        
        device.updated_at = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
        
        db.session.commit()
        
        logging.info(f"GNMI device '{device.name}' updated by user {current_user.username if current_user.is_authenticated else 'system'}")
        
        return jsonify({
            'success': True,
            'message': 'Device updated successfully',
            'device': device.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating GNMI device {device_id}: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to update device'}), 500

@app.route('/api/gnmi/devices/<int:device_id>', methods=['DELETE'])
@admin_required
def delete_gnmi_device(device_id):
    """Delete a GNMI device and its certificates"""
    try:
        device = GnmiDevice.query.get_or_404(device_id)
        device_name = device.name
        
        # Delete device (certificates will be deleted automatically due to cascade)
        db.session.delete(device)
        db.session.commit()
        
        logging.info(f"GNMI device '{device_name}' deleted by user {current_user.username if current_user.is_authenticated else 'system'}")
        
        return jsonify({
            'success': True,
            'message': f"Device '{device_name}' deleted successfully"
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting GNMI device {device_id}: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to delete device'}), 500

@app.route('/api/gnmi/devices/<int:device_id>/certificates', methods=['GET'])
@admin_required
def get_gnmi_device_certificates(device_id):
    """Get certificates for a GNMI device"""
    try:
        device = GnmiDevice.query.get_or_404(device_id)
        certificates = GnmiCertificate.query.filter_by(device_id=device_id).all()
        
        return jsonify({
            'success': True,
            'certificates': [cert.to_dict() for cert in certificates]
        })
        
    except Exception as e:
        logging.error(f"Error getting certificates for device {device_id}: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to fetch certificates'}), 500

@app.route('/api/gnmi/certificates/<int:cert_id>/download', methods=['GET'])
@admin_required
def download_gnmi_certificate(cert_id):
    """Download a GNMI certificate file"""
    try:
        certificate = GnmiCertificate.query.get_or_404(cert_id)
        
        # Create a temporary file-like object
        from io import BytesIO
        file_obj = BytesIO(certificate.content)
        file_obj.seek(0)
        
        return send_file(
            file_obj,
            as_attachment=True,
            download_name=certificate.filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logging.error(f"Error downloading certificate {cert_id}: {str(e)}")
        return jsonify({'error': 'Failed to download certificate'}), 500

# Client API endpoints for GNMI configuration synchronization
@app.route('/api/client/gnmi/devices', methods=['GET'])
def get_client_gnmi_devices():
    """Get GNMI devices for client synchronization (no auth required - used by clients)"""
    try:
        devices = GnmiDevice.query.filter_by(enabled=True).all()
        
        # Return devices without sensitive password data
        device_list = []
        for device in devices:
            device_data = {
                'id': device.id,
                'name': device.name,
                'ip_address': device.ip_address,
                'port': device.port,
                'auth_method': device.auth_method,
                'username': device.username if device.auth_method != 'certificate' else None,
                'description': device.description
            }
            device_list.append(device_data)
        
        return jsonify({
            'success': True,
            'devices': device_list
        })
        
    except Exception as e:
        logging.error(f"Error getting GNMI devices for client: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to fetch devices'}), 500

@app.route('/api/client/gnmi/devices/<int:device_id>/certificates/<cert_type>', methods=['GET'])
def get_client_gnmi_certificate(device_id, cert_type):
    """Get a specific certificate for client download"""
    try:
        certificate = GnmiCertificate.query.filter_by(
            device_id=device_id, 
            cert_type=cert_type
        ).first()
        
        if not certificate:
            return jsonify({'error': 'Certificate not found'}), 404
        
        # Create a temporary file-like object
        from io import BytesIO
        file_obj = BytesIO(certificate.content)
        file_obj.seek(0)
        
        return send_file(
            file_obj,
            as_attachment=True,
            download_name=certificate.filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logging.error(f"Error getting certificate for client: {str(e)}")
        return jsonify({'error': 'Failed to get certificate'}), 500
