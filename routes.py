from flask import render_template, request, jsonify, redirect, url_for, flash, send_file
from functools import wraps
from app import app, db
from models import Client, Test, TestResult, TestClient, ApiToken
from datetime import datetime, timezone, timedelta
import zoneinfo
import json
import logging
import os
import threading
import time
from pdf_generator import generate_test_report_pdf

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
def clients():
    """Client management view"""
    clients = Client.query.order_by(Client.last_seen.desc()).all()
    
    # Mark clients as busy if they're assigned to running tests
    for client in clients:
        busy_tests = db.session.query(Test).join(TestClient).filter(
            TestClient.client_id == client.id,
            Test.status.in_(['running', 'pending'])
        ).all()
        client.is_busy = len(busy_tests) > 0
        client.active_tests = [test.name for test in busy_tests]
    
    return render_template('clients.html', clients=clients)

@app.route('/tests')
def tests():
    """Test management view"""
    tests = Test.query.order_by(Test.created_at.desc()).all()
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
def test_results(test_id):
    """Test results view"""
    test = Test.query.get_or_404(test_id)
    results = TestResult.query.filter_by(test_id=test_id).order_by(TestResult.timestamp.asc()).all()
    clients = db.session.query(Client).join(TestResult).filter(TestResult.test_id == test_id).distinct().all()
    
    return render_template('test_results.html', test=test, results=results, clients=clients)

@app.route('/tutorial')
def tutorial():
    """Tutorial and documentation view"""
    return render_template('tutorial.html')

@app.route('/tokens')
def tokens():
    """API Token management view"""
    tokens = ApiToken.query.order_by(ApiToken.created_at.desc()).all()
    return render_template('tokens.html', tokens=tokens)

# Removed standalone metrics dashboard - all metrics now integrated into test results pages

# API Routes

@app.route('/api/client/register', methods=['POST'])
def register_client():
    """Register a new client using an available API token"""
    data = request.get_json()
    
    hostname = data.get('hostname')
    ip_address = data.get('ip_address')
    token = data.get('token')
    
    if not hostname or not ip_address or not token:
        return jsonify({'error': 'Missing hostname, ip_address, or token'}), 400
    
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
        existing_client.system_info = json.dumps(data.get('system_info', {}))
        client = existing_client
    else:
        # Create new client
        client = Client(
            hostname=hostname,
            ip_address=ip_address,
            status='online',
            last_seen=datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None),
            system_info=json.dumps(data.get('system_info', {}))
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
    data = request.get_json()
    
    if not data or 'client_id' not in data or 'test_id' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        result = TestResult(
            test_id=data['test_id'],
            client_id=data['client_id'],
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat())).replace(tzinfo=None),
            cpu_percent=data.get('cpu_percent'),
            memory_percent=data.get('memory_percent'),
            memory_used=data.get('memory_used'),
            memory_total=data.get('memory_total'),
            disk_percent=data.get('disk_percent'),
            disk_used=data.get('disk_used'),
            disk_total=data.get('disk_total'),
            ping_latency=data.get('ping_latency'),
            ping_packet_loss=data.get('ping_packet_loss'),
            traceroute_hops=data.get('traceroute_hops'),
            traceroute_data=json.dumps(data.get('traceroute_data', {})),
            bandwidth_upload=data.get('bandwidth_upload'),
            bandwidth_download=data.get('bandwidth_download'),
            # Advanced network metrics
            dns_resolution_time=data.get('dns_resolution_time'),
            tcp_connect_time=data.get('tcp_connect_time'),
            ssl_handshake_time=data.get('ssl_handshake_time'),
            ttfb=data.get('ttfb'),
            jitter=data.get('jitter'),
            # Network interface metrics
            network_bytes_sent=data.get('network_bytes_sent'),
            network_bytes_recv=data.get('network_bytes_recv'),
            network_packets_sent=data.get('network_packets_sent'),
            network_packets_recv=data.get('network_packets_recv'),
            network_errors_in=data.get('network_errors_in'),
            network_errors_out=data.get('network_errors_out'),
            network_drops_in=data.get('network_drops_in'),
            network_drops_out=data.get('network_drops_out'),
            # Enhanced CPU metrics
            cpu_load_1min=data.get('cpu_load_1min'),
            cpu_load_5min=data.get('cpu_load_5min'),
            cpu_load_15min=data.get('cpu_load_15min'),
            cpu_cores=data.get('cpu_cores'),
            cpu_freq_current=data.get('cpu_freq_current'),
            cpu_context_switches=data.get('cpu_context_switches'),
            cpu_interrupts=data.get('cpu_interrupts'),
            # Enhanced memory metrics
            memory_available=data.get('memory_available'),
            memory_cached=data.get('memory_cached'),
            memory_buffers=data.get('memory_buffers'),
            memory_shared=data.get('memory_shared'),
            swap_total=data.get('swap_total'),
            swap_used=data.get('swap_used'),
            swap_percent=data.get('swap_percent'),
            # Enhanced disk metrics
            disk_read_iops=data.get('disk_read_iops'),
            disk_write_iops=data.get('disk_write_iops'),
            disk_read_bytes_sec=data.get('disk_read_bytes_sec'),
            disk_write_bytes_sec=data.get('disk_write_bytes_sec'),
            disk_io_util=data.get('disk_io_util'),
            # Process and system metrics
            process_count=data.get('process_count'),
            top_processes_cpu=data.get('top_processes_cpu'),
            top_processes_memory=data.get('top_processes_memory'),
            tcp_connections=data.get('tcp_connections'),
            open_files=data.get('open_files'),
            # Temperature metrics
            cpu_temperature=data.get('cpu_temperature'),
            disk_temperature=data.get('disk_temperature'),
            # QoS metrics
            dscp_value=data.get('dscp_value'),
            cos_value=data.get('cos_value'),
            traffic_class=data.get('traffic_class'),
            qos_policy_compliant=data.get('qos_policy_compliant'),
            bandwidth_per_class=json.dumps(data.get('bandwidth_per_class', {})),
            # Advanced Network-Level Metrics
            mtu_size=data.get('mtu_size'),
            tcp_window_size=data.get('tcp_window_size'),
            tcp_window_scaling=data.get('tcp_window_scaling'),
            tcp_congestion_window=data.get('tcp_congestion_window'),
            tcp_retransmission_rate=data.get('tcp_retransmission_rate'),
            tcp_out_of_order_packets=data.get('tcp_out_of_order_packets'),
            tcp_duplicate_acks=data.get('tcp_duplicate_acks'),
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
            network_interface_errors=data.get('network_interface_errors')
        )
        
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
        
        return jsonify({'status': 'success', 'message': 'Results submitted successfully'})
        
    except Exception as e:
        logging.error(f"Error submitting test results: {str(e)}")
        return jsonify({'error': 'Failed to submit results'}), 500

@app.route('/api/test/create', methods=['POST'])
def create_test():
    """Create a new test"""
    data = request.get_json()
    
    if not data or 'name' not in data or 'destination' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Clean up destination by removing protocol prefixes
        destination = data['destination'].strip()
        if destination.startswith('http://'):
            destination = destination[7:]
        elif destination.startswith('https://'):
            destination = destination[8:]
        
        # Remove trailing slash if present
        if destination.endswith('/'):
            destination = destination[:-1]
        
        test = Test(
            name=data['name'],
            description=data.get('description', ''),
            destination=destination,
            scheduled_time=datetime.fromisoformat(data['scheduled_time']) if data.get('scheduled_time') else None,
            duration=data.get('duration', 300),
            interval=data.get('interval', 5)
        )
        
        db.session.add(test)
        db.session.flush()  # Get the test ID
        
        # Assign clients to test
        client_ids = data.get('client_ids', [])
        for client_id in client_ids:
            test_client = TestClient(test_id=test.id, client_id=client_id)
            db.session.add(test_client)
        
        db.session.commit()
        
        return jsonify({'status': 'success', 'test_id': test.id, 'message': 'Test created successfully'})
        
    except Exception as e:
        logging.error(f"Error creating test: {str(e)}")
        return jsonify({'error': 'Failed to create test'}), 500

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
            'traceroute_data': {}
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
            db.session.commit()
            
        return jsonify({
            'progress': round(progress, 1),
            'status': test.status,
            'elapsed_time': int(elapsed_time),
            'remaining_time': max(0, test.duration - int(elapsed_time))
        })
    
    return jsonify({'progress': 0, 'status': test.status})

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
        
        # Check if client has test data
        result_count = TestResult.query.filter_by(client_id=client_id).count()
        test_client_count = TestClient.query.filter_by(client_id=client_id).count()
        
        # Store client info for logging
        client_hostname = client.hostname
        
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
def get_tokens():
    """Get all API tokens with optional filtering"""
    status_filter = request.args.get('status')
    
    query = ApiToken.query
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    tokens = query.order_by(ApiToken.created_at.desc()).all()
    
    return jsonify({
        'tokens': [token.to_dict() for token in tokens],
        'total': len(tokens)
    })

@app.route('/api/tokens', methods=['POST'])
def create_token():
    """Create a new API token"""
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({'error': 'Token name is required'}), 400
    
    # Check for duplicate names
    existing = ApiToken.query.filter_by(name=data['name']).first()
    if existing:
        return jsonify({'error': 'Token name already exists'}), 400
    
    try:
        token = ApiToken(
            name=data['name'],
            description=data.get('description', '')
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
def update_token(token_id):
    """Update an API token"""
    token = ApiToken.query.get_or_404(token_id)
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        if 'name' in data:
            # Check for duplicate names (excluding current token)
            existing = ApiToken.query.filter(
                ApiToken.name == data['name'],
                ApiToken.id != token_id
            ).first()
            if existing:
                return jsonify({'error': 'Token name already exists'}), 400
            token.name = data['name']
        
        if 'description' in data:
            token.description = data['description']
        
        if 'status' in data and data['status'] in ['available', 'consumed', 'revoked']:
            token.status = data['status']
        
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
