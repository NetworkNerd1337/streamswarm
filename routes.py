from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import Client, Test, TestResult, TestClient
from datetime import datetime, timedelta
import json
import logging

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
    return render_template('clients.html', clients=clients)

@app.route('/tests')
def tests():
    """Test management view"""
    tests = Test.query.order_by(Test.created_at.desc()).all()
    clients = Client.query.filter_by(status='online').all()
    return render_template('tests.html', tests=tests, clients=clients)

@app.route('/test/<int:test_id>')
def test_results(test_id):
    """Test results view"""
    test = Test.query.get_or_404(test_id)
    results = TestResult.query.filter_by(test_id=test_id).order_by(TestResult.timestamp.asc()).all()
    clients = db.session.query(Client).join(TestResult).filter(TestResult.test_id == test_id).distinct().all()
    
    return render_template('test_results.html', test=test, results=results, clients=clients)

# API Routes

@app.route('/api/client/register', methods=['POST'])
def register_client():
    """Register a new client or update existing client info"""
    data = request.get_json()
    
    if not data or 'hostname' not in data or 'ip_address' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if client already exists
    client = Client.query.filter_by(hostname=data['hostname']).first()
    
    if client:
        # Update existing client
        client.ip_address = data['ip_address']
        client.status = 'online'
        client.last_seen = datetime.utcnow()
        if 'system_info' in data:
            client.system_info = json.dumps(data['system_info'])
    else:
        # Create new client
        client = Client(
            hostname=data['hostname'],
            ip_address=data['ip_address'],
            status='online',
            system_info=json.dumps(data.get('system_info', {}))
        )
        db.session.add(client)
    
    db.session.commit()
    
    return jsonify({
        'client_id': client.id,
        'status': 'registered',
        'message': 'Client registered successfully'
    })

@app.route('/api/client/<int:client_id>/heartbeat', methods=['POST'])
def client_heartbeat(client_id):
    """Update client last seen timestamp"""
    client = Client.query.get_or_404(client_id)
    client.last_seen = datetime.utcnow()
    client.status = 'online'
    db.session.commit()
    
    return jsonify({'status': 'ok'})

@app.route('/api/client/<int:client_id>/tests', methods=['GET'])
def get_client_tests(client_id):
    """Get pending tests for a client"""
    # Find tests assigned to this client that are ready to run
    assigned_tests = db.session.query(Test).join(TestClient).filter(
        TestClient.client_id == client_id,
        TestClient.status == 'assigned',
        Test.status.in_(['pending', 'running'])
    ).all()
    
    # Check if any tests should be started
    now = datetime.utcnow()
    ready_tests = []
    
    for test in assigned_tests:
        if test.scheduled_time is None or test.scheduled_time <= now:
            ready_tests.append(test.to_dict())
            
            # Update test status if it's the first time it's being started
            if test.status == 'pending':
                test.status = 'running'
                test.started_at = now
    
    db.session.commit()
    
    return jsonify({'tests': ready_tests})

@app.route('/api/test/results', methods=['POST'])
def submit_test_results():
    """Submit test results from client"""
    data = request.get_json()
    
    if not data or 'client_id' not in data or 'test_id' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        result = TestResult(
            test_id=data['test_id'],
            client_id=data['client_id'],
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
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
            bandwidth_download=data.get('bandwidth_download')
        )
        
        db.session.add(result)
        db.session.commit()
        
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
        test = Test(
            name=data['name'],
            description=data.get('description', ''),
            destination=data['destination'],
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
            'ping_packet_loss': {}
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
    
    return jsonify(data)

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics"""
    # Mark clients as offline if they haven't been seen in 5 minutes
    offline_threshold = datetime.utcnow() - timedelta(minutes=5)
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
