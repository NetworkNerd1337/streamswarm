# StreamSwarm Tutorial: Complete Guide with Use Cases

This comprehensive tutorial walks you through real-world scenarios for using StreamSwarm's distributed network monitoring system. You'll learn how to set up both server and client components, configure the database, install dependencies, and understand when and how to use each component.

## System Requirements and Installation

### Python Installation Guide

#### Linux Installation (Ubuntu/Debian)
```bash
# Update system packages
sudo apt update

# Install Python 3.9+ and essential tools
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev
sudo apt install build-essential libffi-dev libssl-dev

# Install network monitoring system dependencies
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool
sudo apt install libpcap-dev tcpdump

# Create convenient symbolic links (optional)
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
sudo ln -sf /usr/bin/pip3.9 /usr/bin/pip

# Verify installation
python --version  # Should show Python 3.9+
pip --version
```

#### Linux Installation (CentOS/RHEL/Fedora)
```bash
# Install Python and development tools
sudo dnf install python3.9 python3.9-pip python3.9-devel gcc
sudo dnf install openssl-devel libffi-devel

# Install system monitoring dependencies
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool
sudo dnf install libpcap-devel tcpdump

# For older systems, replace 'dnf' with 'yum'
```

#### Windows Installation
```powershell
# Method 1: Official Python Installer (Recommended)
# 1. Visit https://www.python.org/downloads/windows/
# 2. Download Python 3.9+ (64-bit recommended)
# 3. Run installer with these settings:
#    ✅ Add Python to PATH
#    ✅ Install for all users
#    ✅ Include pip

# Method 2: Using Package Managers
# Chocolatey:
choco install python

# Winget (Windows 10/11):
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

#### macOS Installation
```bash
# Method 1: Homebrew (Recommended)
# Install Homebrew first if needed:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install system dependencies
brew install libpcap

# Method 2: Official Installer
# Visit https://www.python.org/downloads/mac-osx/
# Download and run Python 3.9+ installer

# Verify installation
python3 --version
pip3 --version
```

### StreamSwarm Dependencies

#### Create Virtual Environment (Recommended)
```bash
# Linux/macOS
python3 -m venv streamswarm-env
source streamswarm-env/bin/activate

# Windows
python -m venv streamswarm-env
streamswarm-env\Scripts\activate
```

#### Install Python Packages
```bash
# Core application dependencies
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0
pip install psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0
pip install gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0

# Advanced network analysis and QoS monitoring
pip install scapy>=2.5.0

# Bandwidth testing capabilities
pip install speedtest-cli>=2.1.3

# Professional report generation and visualization
pip install reportlab>=4.4.0 matplotlib>=3.10.0

# Single command installation (all dependencies):
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0
```

### System Dependencies by Platform

#### Linux System Tools
```bash
# Ubuntu/Debian
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool libpcap-dev tcpdump

# CentOS/RHEL/Fedora  
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool libpcap-devel tcpdump

# Initialize hardware sensors
sudo sensors-detect --auto
```

#### Windows System Setup
```powershell
# Most functionality works without additional tools
# For enhanced features (optional):

# Install Wireshark for advanced packet analysis
# Download from: https://www.wireshark.org/download.html

# Enable Windows Subsystem for Linux (optional)
wsl --install

# Note: Run PowerShell as Administrator for system monitoring features
```

#### macOS System Tools
```bash
# Install packet capture library
brew install libpcap

# Optional network tools
brew install nmap traceroute

# Note: Some monitoring features require admin privileges
```

### Installation Verification

#### Test Dependencies
```python
# Create test script: test_dependencies.py
try:
    import flask
    import psutil
    import requests
    import scapy
    import speedtest
    import reportlab
    import matplotlib
    print("✅ All dependencies installed successfully!")
    
    # Test system access
    print(f"✅ CPU usage: {psutil.cpu_percent()}%")
    print(f"✅ Memory usage: {psutil.virtual_memory().percent}%")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
```

#### Run Test
```bash
python test_dependencies.py
```

### Common Installation Issues

#### Permission Errors
```bash
# Linux: Use virtual environment or user install
python3 -m pip install --user <package>

# Or create virtual environment (recommended)
python3 -m venv env && source env/bin/activate && pip install <package>
```

#### Windows Issues
```cmd
# Long path support (run as Administrator)
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1

# If pip not found, reinstall Python with "Add to PATH" option
```

#### Network Monitoring Permissions
```bash
# Linux: Some features require elevated privileges
sudo python client.py --server http://server:5000

# Windows: Run Command Prompt as Administrator
# macOS: Use sudo for system-level monitoring
```

# CentOS/RHEL/Fedora  
sudo yum install iputils traceroute
```
```

**Note:** See `DEPENDENCIES.md` for complete package information including a properly formatted requirements.txt template.

### System Packages
Additional system packages may be required:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3-pip python3-dev
sudo apt install libpcap-dev tcpdump  # For QoS monitoring
sudo apt install postgresql postgresql-contrib
```

**CentOS/RHEL:**
```bash
sudo yum install python3-pip python3-devel
sudo yum install libpcap-devel tcpdump
sudo yum install postgresql-server postgresql-contrib
```

## Database Setup

StreamSwarm uses PostgreSQL for data storage. Here's how to set it up:

### 1. Install and Configure PostgreSQL

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib
sudo postgresql-setup initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

### 2. Create Database and User

```bash
sudo -u postgres psql

CREATE DATABASE streamswarm;
CREATE USER streamswarm_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE streamswarm TO streamswarm_user;
\q
```

### 3. Configure Environment Variables

```bash
# Set database connection
export DATABASE_URL="postgresql://streamswarm_user:your_secure_password@localhost:5432/streamswarm"
export SESSION_SECRET="your-random-session-secret"

# Or create a .env file
echo "DATABASE_URL=postgresql://streamswarm_user:your_secure_password@localhost:5432/streamswarm" > .env
echo "SESSION_SECRET=your-random-session-secret" >> .env
```

### 4. Test Database Connection

```bash
psql "$DATABASE_URL" -c "SELECT version();"
```

**Note:** StreamSwarm automatically creates all necessary database tables (clients, tests, test_results with 45+ metric columns) when the server starts for the first time.

## Table of Contents
1. [Understanding the Architecture](#understanding-the-architecture)
2. [Server Setup and Management](#server-setup-and-management)
3. [Client Deployment Strategies](#client-deployment-strategies)
4. [Real-World Use Cases](#real-world-use-cases)
5. [Step-by-Step Tutorials](#step-by-step-tutorials)
6. [Advanced Scenarios](#advanced-scenarios)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Understanding the Architecture

### Server vs Client Roles

**The Server:**
- Central control hub for all monitoring activities
- Hosts the web dashboard for visualization and management
- Stores all collected data in PostgreSQL database
- Schedules and coordinates tests across multiple clients
- Provides REST API for programmatic access

**The Client:**
- Lightweight monitoring agent deployed on target hosts
- Executes network tests (ping, traceroute) to specified destinations
- Collects system metrics (CPU, memory, disk usage)
- Reports data back to server in real-time
- Receives test assignments and schedules from server

### When to Use What

**Deploy the Server when you need:**
- Central monitoring dashboard
- Test coordination across multiple locations
- Historical data storage and analysis
- Management interface for non-technical users
- Scheduled automated testing

**Deploy a Client when you want to:**
- Monitor network performance from a specific location
- Test connectivity to critical services
- Track system resource usage
- Participate in distributed network testing
- Collect data for centralized analysis

## Server Setup and Management

### Basic Server Deployment

**1. Production Server Setup:**
```bash
# Install dependencies
pip install flask flask-sqlalchemy psutil requests gunicorn psycopg2-binary

# Configure PostgreSQL database
export DATABASE_URL=postgresql://streamswarm:password@localhost/streamswarm_db

# Start server
python main.py
```

**2. Enterprise Configuration:**
```bash
# Production environment variables
export SERVER_HOST=0.0.0.0
export SERVER_PORT=5000
export DEBUG=False
export DATABASE_URL=postgresql://user:pass@db-server:5432/streamswarm
export SESSION_SECRET=your-secure-random-key-here

# Start with Gunicorn for production
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app
```

### Server Management Tasks

**Monitoring Server Health:**
- Access dashboard at `http://your-server:5000`
- Check "Total Clients" and "Online Clients" metrics
- Monitor "Active Tests" for ongoing monitoring activities
- Review "Recent Activity" for system health

**Database Maintenance:**
```bash
# Check database connection
python -c "from app import db; print('Database connected:', db.engine.url)"

# View stored data
python -c "from models import Client; print('Total clients:', Client.query.count())"
```

## Client Deployment Strategies

### Single Client Deployment

**Basic Client Setup:**
```bash
# Simple connection to server
python client.py --server http://192.168.1.100:5000

# Named client with verbose logging
python client.py --server http://monitoring.company.com:5000 --name "Web-Server-01" --verbose
```

### Multi-Location Deployment

**Office Network Monitoring:**
```bash
# New York Office
python client.py --server http://monitoring.company.com:5000 --name "NYC-Office-Gateway"

# London Office  
python client.py --server http://monitoring.company.com:5000 --name "London-Office-Gateway"

# Tokyo Office
python client.py --server http://monitoring.company.com:5000 --name "Tokyo-Office-Gateway"
```

**Data Center Monitoring:**
```bash
# Web servers
python client.py --server http://monitoring.company.com:5000 --name "WebServer-Rack1-01"
python client.py --server http://monitoring.company.com:5000 --name "WebServer-Rack1-02"

# Database servers
python client.py --server http://monitoring.company.com:5000 --name "DBServer-Primary"
python client.py --server http://monitoring.company.com:5000 --name "DBServer-Replica"
```

### Automated Client Deployment

**Systemd Service (Linux):**
```ini
# /etc/systemd/system/streamswarm-client.service
[Unit]
Description=StreamSwarm Network Monitoring Client
After=network.target

[Service]
Type=simple
User=monitoring
WorkingDirectory=/opt/streamswarm
ExecStart=/usr/bin/python3 client.py --server http://monitoring.company.com:5000 --name "%H"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Windows Service Setup:**
```batch
# Install as Windows service using NSSM
nssm install StreamSwarmClient "C:\Python\python.exe"
nssm set StreamSwarmClient Parameters "C:\StreamSwarm\client.py --server http://monitoring.company.com:5000 --name %COMPUTERNAME%"
nssm set StreamSwarmClient Start SERVICE_AUTO_START
nssm start StreamSwarmClient
```

## Real-World Use Cases

### Use Case 1: Website Performance Monitoring

**Scenario:** E-commerce company wants to monitor website performance from customer locations worldwide.

**Setup:**
1. **Server:** Deploy on cloud infrastructure (AWS/Azure/GCP)
2. **Clients:** Install on VPS instances in major cities (New York, London, Tokyo, Sydney)

**Implementation:**
```bash
# Server deployment
export DATABASE_URL=postgresql://user:pass@rds-instance:5432/streamswarm
python main.py

# Client deployments
# New York VPS
python client.py --server https://monitoring.ecommerce.com:5000 --name "NYC-Customer-View"

# London VPS  
python client.py --server https://monitoring.ecommerce.com:5000 --name "London-Customer-View"

# Create test targeting website
# Access dashboard → Tests → Create Test
# Destination: www.ecommerce.com
# Duration: 3600 seconds (1 hour)
# Interval: 30 seconds
# Select all geographic clients
```

**Expected Results:**
- Monitor website response times from different geographic locations
- Detect regional performance issues
- Track CDN effectiveness across continents
- Identify peak traffic impact on response times

### Use Case 2: Network Infrastructure Monitoring

**Scenario:** IT department needs to monitor internal network connectivity between office locations and critical services.

**Setup:**
1. **Server:** Deploy on internal infrastructure server
2. **Clients:** Install on gateway/router hosts at each office location

**Implementation:**
```bash
# Server on internal infrastructure
export DATABASE_URL=postgresql://dbserver:5432/streamswarm
python main.py

# Office gateway clients
# Main Office (HQ)
python client.py --server http://10.0.1.100:5000 --name "HQ-Gateway"

# Branch Office 1
python client.py --server http://10.0.1.100:5000 --name "Branch-01-Gateway"

# Branch Office 2
python client.py --server http://10.0.1.100:5000 --name "Branch-02-Gateway"

# Create tests for critical services
# Test 1: Internal file server connectivity
# Destination: fileserver.company.local
# Test 2: Internet connectivity
# Destination: 8.8.8.8
# Test 3: VPN endpoint connectivity
# Destination: vpn.company.com
```

**Expected Results:**
- Monitor inter-office network connectivity
- Detect VPN tunnel performance issues
- Track critical service availability
- Identify network bottlenecks between locations

### Use Case 3: Cloud Service Performance Monitoring

**Scenario:** Development team needs to monitor performance of microservices deployed across multiple cloud regions.

**Setup:**
1. **Server:** Deploy in primary cloud region
2. **Clients:** Deploy in each cloud region where services are running

**Implementation:**
```bash
# Server in us-east-1
export DATABASE_URL=postgresql://rds-endpoint:5432/streamswarm
python main.py

# Clients in each region
# US East
python client.py --server http://monitoring-service:5000 --name "AWS-US-East-1"

# US West
python client.py --server http://monitoring-service:5000 --name "AWS-US-West-2"

# Europe
python client.py --server http://monitoring-service:5000 --name "AWS-EU-West-1"

# Create tests for each microservice
# Test 1: User authentication service
# Destination: auth.api.company.com
# Test 2: Payment processing service
# Destination: payments.api.company.com
# Test 3: Database connectivity
# Destination: db-cluster.company.internal
```

**Expected Results:**
- Monitor cross-region service latency
- Detect regional service degradation
- Track database performance from different regions
- Optimize service placement based on performance data

### Use Case 4: ISP Performance Analysis

**Scenario:** Network administrator wants to compare performance across multiple ISP connections for load balancing decisions.

**Setup:**
1. **Server:** Deploy on neutral location (cloud or colocation)
2. **Clients:** Install on hosts behind different ISP connections

**Implementation:**
```bash
# Server on neutral cloud instance
python main.py

# Client behind ISP 1 (Verizon)
python client.py --server http://monitoring.cloud.com:5000 --name "Verizon-Connection"

# Client behind ISP 2 (Comcast)
python client.py --server http://monitoring.cloud.com:5000 --name "Comcast-Connection"

# Client behind ISP 3 (AT&T)
python client.py --server http://monitoring.cloud.com:5000 --name "ATT-Connection"

# Create tests to common destinations
# Test 1: Major CDN performance
# Destination: cloudflare.com
# Test 2: Cloud provider performance
# Destination: aws.amazon.com
# Test 3: Popular website performance
# Destination: google.com
```

**Expected Results:**
- Compare ISP performance to same destinations
- Identify best-performing ISP for different services
- Make data-driven load balancing decisions
- Detect ISP-specific connectivity issues

## Step-by-Step Tutorials

### Tutorial 1: Setting Up Basic Monitoring

**Objective:** Monitor website performance from two office locations.

**Step 1: Server Setup**
```bash
# On monitoring server (192.168.1.100)
git clone [your-streamswarm-repo]
cd streamswarm
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 psycopg2-binary>=2.9.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0 scapy>=2.5.0 email-validator>=2.0.0 werkzeug>=2.3.0
sudo apt install lm-sensors smartmontools ethtool libpcap-dev tcpdump
export DATABASE_URL=postgresql://streamswarm:password@localhost/streamswarm
python main.py
```

**Step 2: Access Web Dashboard**
1. Open browser to `http://192.168.1.100:5000`
2. Verify dashboard loads with 0 clients connected
3. Navigate to Clients and Tests pages to familiarize yourself

**Step 3: Deploy First Client**
```bash
# On first office computer (192.168.1.50)
python client.py --server http://192.168.1.100:5000 --name "Main-Office-Desktop"
```

**Step 4: Verify Client Connection**
1. Refresh dashboard - should show 1 total client, 1 online
2. Go to Clients page - should show "Main-Office-Desktop" with green "Online" status
3. Click info button to view client details

**Step 5: Deploy Second Client**
```bash
# On second office computer (192.168.2.50)
python client.py --server http://192.168.1.100:5000 --name "Branch-Office-Desktop"
```

**Step 6: Create Your First Test**
1. Go to Tests page
2. Click "Create Test"
3. Fill in:
   - Test Name: "Comprehensive Performance Check"
   - Destination: "google.com"
   - Duration: 300 seconds
   - Interval: 10 seconds
   - Select both clients
4. Click "Create Test"

**Note**: This test will measure 65+ metrics including network performance, QoS analysis, application-layer timing, infrastructure health, and system resources.

**Step 7: Monitor Test Progress**
1. Watch progress bar advance in real-time
2. Click chart icon when test completes
3. Analyze comprehensive performance differences between office locations:
   - Network performance: latency, bandwidth, MTU, TCP analysis
   - QoS metrics: DSCP classification, traffic policing, ECN capability
   - Application layer: content timing, compression, CDN performance
   - Infrastructure health: power usage, drive health, memory errors
   - System resources: CPU, memory, disk, network interface statistics
4. Export professional PDF reports with charts, analysis tables, and actionable recommendations

### Tutorial 2: Advanced Multi-Service Monitoring

**Objective:** Set up comprehensive monitoring for a web application stack.

**Step 1: Infrastructure Planning**
```
Services to monitor:
- Web application (app.company.com)
- API backend (api.company.com)
- Database connectivity (through app server)
- CDN performance (cdn.company.com)

Monitoring locations:
- Production server (direct connectivity)
- Office network (user perspective)
- External VPS (customer perspective)
```

**Step 2: Deploy Server with Production Database**
```bash
# On monitoring server
export DATABASE_URL=postgresql://monitoring_user:secure_password@db.company.com:5432/streamswarm_production
export SESSION_SECRET=your-production-secret-key
gunicorn --bind 0.0.0.0:5000 --workers 2 main:app
```

**Step 3: Deploy Strategic Clients**
```bash
# Production server client
python client.py --server http://monitoring.company.com:5000 --name "Production-Server-Internal"

# Office network client
python client.py --server http://monitoring.company.com:5000 --name "Corporate-Office-Network"

# External perspective client (on VPS)
python client.py --server http://monitoring.company.com:5000 --name "External-Customer-View"
```

**Step 4: Create Comprehensive Test Suite**

**Test 1: Web Application Performance**
- Name: "Web App User Experience"
- Destination: app.company.com
- Duration: 7200 seconds (2 hours)
- Interval: 60 seconds
- Clients: All locations

**Test 2: API Performance**
- Name: "API Backend Performance"
- Destination: api.company.com
- Duration: 7200 seconds
- Interval: 30 seconds
- Clients: Production-Server-Internal, External-Customer-View

**Test 3: CDN Performance**
- Name: "CDN Global Performance"
- Destination: cdn.company.com
- Duration: 3600 seconds (1 hour)
- Interval: 120 seconds
- Clients: All locations

**Step 5: Analysis and Optimization**
1. Compare performance across different client locations
2. Identify performance discrepancies between internal and external views
3. Use data to optimize CDN configuration
4. Set up alerts based on performance thresholds

## Advanced Scenarios

### Scenario 1: High-Availability Monitoring Setup

**Requirements:**
- Multiple server instances for redundancy
- Load balancing between monitoring servers
- Client failover capabilities

**Implementation:**
```bash
# Primary monitoring server
export DATABASE_URL=postgresql://primary:5432/streamswarm
python main.py

# Secondary monitoring server (same database)
export DATABASE_URL=postgresql://primary:5432/streamswarm
export SERVER_PORT=5001
python main.py

# Load balancer configuration (nginx)
upstream streamswarm_backend {
    server monitoring1.company.com:5000;
    server monitoring2.company.com:5001;
}

# Smart client deployment with failover
python client.py --server http://monitoring-lb.company.com --name "Resilient-Client"
```

### Scenario 2: Automated Test Scheduling

**Objective:** Automatically create tests based on business hours and critical periods.

**Implementation:**
```python
# automated_testing.py
import requests
import schedule
import time
from datetime import datetime, timedelta

def create_business_hours_test():
    """Create intensive monitoring during business hours"""
    test_data = {
        "name": f"Business Hours Monitoring {datetime.now().strftime('%Y-%m-%d')}",
        "destination": "critical-service.company.com",
        "duration": 28800,  # 8 hours
        "interval": 30,     # Every 30 seconds
        "client_ids": [1, 2, 3, 4]  # All critical location clients
    }
    
    response = requests.post(
        'http://monitoring.company.com:5000/api/test/create',
        json=test_data
    )
    print(f"Created business hours test: {response.json()}")

def create_maintenance_window_test():
    """Create light monitoring during maintenance windows"""
    test_data = {
        "name": f"Maintenance Window Check {datetime.now().strftime('%Y-%m-%d')}",
        "destination": "backup-service.company.com",
        "duration": 3600,   # 1 hour
        "interval": 300,    # Every 5 minutes
        "client_ids": [1]   # Single client monitoring
    }
    
    response = requests.post(
        'http://monitoring.company.com:5000/api/test/create',
        json=test_data
    )
    print(f"Created maintenance test: {response.json()}")

# Schedule automated tests
schedule.every().monday.at("08:00").do(create_business_hours_test)
schedule.every().saturday.at("02:00").do(create_maintenance_window_test)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Scenario 3: Performance Alerting Integration

**Objective:** Integrate with external alerting systems based on performance thresholds.

**Implementation:**
```python
# performance_monitor.py
import requests
import time
import smtplib
from email.mime.text import MIMEText

def check_performance_thresholds():
    """Monitor performance and send alerts"""
    # Get recent test data
    response = requests.get('http://monitoring.company.com:5000/api/dashboard/stats')
    data = response.json()
    
    for activity in data['recent_activity']:
        if activity['ping_latency'] and activity['ping_latency'] > 100:  # 100ms threshold
            send_alert(
                f"High latency detected: {activity['client_name']} to {activity['test_name']} - {activity['ping_latency']}ms"
            )

def send_alert(message):
    """Send email alert"""
    msg = MIMEText(f"StreamSwarm Alert: {message}")
    msg['Subject'] = 'Network Performance Alert'
    msg['From'] = 'monitoring@company.com'
    msg['To'] = 'sysadmin@company.com'
    
    server = smtplib.SMTP('smtp.company.com', 587)
    server.send_message(msg)
    server.quit()

# Run monitoring loop
while True:
    check_performance_thresholds()
    time.sleep(300)  # Check every 5 minutes
```

## Troubleshooting Common Issues

### Client Connection Problems

**Issue:** Client shows "Failed to register" error
```bash
# Check server accessibility
curl http://monitoring-server:5000/api/dashboard/stats

# Verify client can reach server
ping monitoring-server

# Check firewall rules
sudo ufw status
sudo iptables -L
```

**Issue:** Client connects but shows as offline
```bash
# Check client heartbeat
python client.py --server http://monitoring-server:5000 --verbose

# Verify server receives heartbeat
tail -f /var/log/streamswarm/server.log
```

### Server Performance Issues

**Issue:** Web dashboard loads slowly
```bash
# Check database performance
psql $DATABASE_URL -c "SELECT COUNT(*) FROM test_result;"

# Monitor server resources
top -p $(pgrep -f "python.*main.py")

# Optimize database queries
psql $DATABASE_URL -c "EXPLAIN ANALYZE SELECT * FROM test_result ORDER BY timestamp DESC LIMIT 100;"
```

**Issue:** Tests not starting automatically
```bash
# Check test assignment logic
python -c "
from models import Test, TestClient
tests = Test.query.filter_by(status='pending').all()
print(f'Pending tests: {len(tests)}')
for test in tests:
    clients = TestClient.query.filter_by(test_id=test.id).all()
    print(f'Test {test.id} assigned to {len(clients)} clients')
"
```

### Network Testing Issues

**Issue:** Ping tests return null values
```bash
# Test ping command manually
ping -c 4 google.com

# Check permissions for network utilities
which ping
ls -la $(which ping)

# Verify traceroute availability
which traceroute
```

**Issue:** Inconsistent test results
```bash
# Check system load during tests
vmstat 5

# Monitor network interfaces
iftop -i eth0

# Verify DNS resolution consistency
nslookup google.com
dig google.com
```

This comprehensive tutorial provides the foundation for successfully deploying and managing StreamSwarm in various real-world scenarios. Each use case demonstrates the flexibility and power of distributed network monitoring for different organizational needs.