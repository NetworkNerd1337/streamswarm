# StreamSwarm - Distributed Network Monitoring System

A comprehensive Python-based distributed network monitoring system with AI-powered diagnostics, secure web authentication, and real-time analytics for enterprise network management.

## Overview

StreamSwarm provides distributed network performance testing and system resource monitoring across multiple client hosts. The system features a central Flask server with a web dashboard for managing clients, scheduling tests, and visualizing performance data with AI-powered diagnostics and professional PDF reporting.

## Key Features

- **🤖 AI-Powered Network Diagnostics**: Local machine learning models for anomaly detection, health classification, and intelligent troubleshooting recommendations
- **🔒 Zero-Trust ML Architecture**: All AI processing runs locally using Scikit-learn with no external dependencies or cloud connections
- **🌐 Comprehensive Network Analysis**: Latency, packet loss, jitter, bandwidth, MTU discovery, TCP analysis, QoS monitoring
- **🔧 GNMI Managed Infrastructure Analysis**: Hop-by-hop telemetry from GNMI-enabled network devices for enterprise infrastructure
- **📊 Advanced Metrics Collection**: 65+ performance metrics including application layer, infrastructure health, and system resources
- **🏢 Multi-Client Architecture**: Distributed testing from multiple network locations with automatic client assignment
- **⏰ Enhanced Recurring Tests**: Two behavior modes - "Continue Same Test" for real-time monitoring and "Create New Tests" for historical data tracking
- **📈 Real-Time Visualization**: Web-based interface with interactive charts and comprehensive dashboards
- **📄 Professional Reporting**: Executive PDF reports with charts, analysis, and recommendations
- **🛡️ Secure Web Authentication**: Role-based access control with user management and session security
- **⚡ High Performance**: Gunicorn WSGI server with autoscaling support and PostgreSQL backend

## Quick Start

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/NetworkNerd1337/Swarm.git
cd Swarm
```

Install Python dependencies:
```bash
# All dependencies (single command)
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0 scikit-learn>=1.0.0 pandas>=1.3.0 numpy>=1.20.0 joblib>=1.0.0 pygnmi>=0.8.15

# Or step-by-step:
pip install flask flask-sqlalchemy psutil requests gunicorn psycopg2-binary werkzeug email-validator
pip install scapy speedtest-cli reportlab matplotlib
pip install scikit-learn pandas numpy joblib  # For AI/ML features
```

Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool libpcap-dev tcpdump
sudo apt install iw wireless-tools libiw-dev network-manager
sudo apt install python3-numpy python3-scipy python3-sklearn libatlas-base-dev liblapack-dev gfortran

# CentOS/RHEL/Fedora
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool libpcap-devel tcpdump
sudo dnf install iw wireless-tools NetworkManager
sudo dnf install python3-numpy python3-scipy python3-scikit-learn atlas-devel lapack-devel gcc-gfortran
```

### 2. Database Setup (Optional)

StreamSwarm works with SQLite by default. For production, set up PostgreSQL:

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
sudo dnf install postgresql-server postgresql-contrib  # CentOS/RHEL/Fedora

# Create database and user
sudo -u postgres psql
CREATE DATABASE streamswarm;
CREATE USER streamswarm_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE streamswarm TO streamswarm_user;
\q

# Set environment variable
export DATABASE_URL="postgresql://streamswarm_user:your_secure_password@localhost:5432/streamswarm"
export SESSION_SECRET="your-random-session-secret"
```

### 3. Start the Server

```bash
python main.py
```

### 4. Access Web Interface

1. Open browser to `http://localhost:5000`
2. **Default login:** username=`admin`, password=`admin123`
3. **⚠️ Important:** Change the default password immediately!
4. Access via: Username dropdown → "My Profile"

### 5. Connect Clients

On each host you want to monitor:
```bash
python client.py --server http://your-server-ip:5000 --name "Client-Name"
```

## System Architecture

### Overall Architecture
- **Frontend**: Flask-based web application with Bootstrap UI framework
- **Backend**: Python Flask server with SQLAlchemy ORM
- **Database**: PostgreSQL (SQLite for development)
- **Client Architecture**: Standalone Python clients that connect to the central server
- **Deployment**: Gunicorn WSGI server with autoscaling support

### Server Components
- **Flask Application** (`app.py`): Core application setup with database configuration
- **Models** (`models.py`): Database schema definitions using SQLAlchemy
- **Routes** (`routes.py`): API endpoints and web interface handlers
- **PDF Generator** (`pdf_generator.py`): Executive reporting functionality
- **ML Diagnostics** (`ml_diagnostics.py`): AI-powered network analysis

### Client Components
- **Client Application** (`client.py`): Standalone monitoring client with 65+ metrics collection
- **Network Testing**: Ping, traceroute, bandwidth, and QoS analysis
- **System Monitoring**: CPU, memory, disk, and network interface tracking
- **Wireless Detection**: Signal strength and wireless network analysis

## Authentication System

StreamSwarm includes a secure authentication system that protects the web GUI while keeping client API access separate.

### Default Admin Account
- **Username:** `admin`
- **Password:** `admin123`
- **Role:** Administrator

### Authentication Features
- Secure password hashing using Werkzeug
- Role-based access control (Administrator/User roles)
- Session-based authentication with Flask-Login
- Self-service password changes for all users
- Admin user management interface
- Completely separate from client API token system

### User Management (Admin Only)
Access via: Username dropdown → "User Management"

Administrators can:
- Create new user accounts
- Assign roles (Admin/User)
- Activate/deactivate accounts
- Delete users (with protections)
- View user activity

### Self-Service Features
Access via: Username dropdown → "My Profile"

All users can:
- Change their own password securely
- View profile information
- See account activity and login history

## AI/ML Diagnostic System

StreamSwarm includes comprehensive local machine learning capabilities for intelligent network analysis and performance prediction.

### Core AI Features

#### 🔮 Performance Prediction System
- **Pre-Test Forecasting**: Predict network latency before running tests
- **Configuration Analysis**: Evaluate impact of packet size, test type, and destination
- **Capacity Planning**: Compare multiple scenarios for network planning
- **Real-Time Integration**: Built into test creation workflow for immediate insights

#### 🔍 Diagnostic Analysis
- **Anomaly Detection**: Isolation Forest algorithm identifies unusual network patterns
- **Health Classification**: Random Forest algorithm categorizes network health status
- **Trend Analysis**: Gradient Boosting predicts network performance trends
- **Feature Engineering**: Analyzes 40+ metrics including network performance, system resources, and QoS data

### Zero-Trust Architecture
- All ML processing runs locally using Scikit-learn
- No external dependencies or cloud connections required
- Models trained exclusively on your network's historical data
- Fallback rule-based predictions ensure reliability

### Performance Prediction Usage

#### Quick Prediction (Test Creation)
1. Navigate to **Tests** page and click **"New Test"**
2. Configure test parameters (destination, packet size, test type)
3. Click **"Predict Performance"** button
4. Review latency forecast and confidence score
5. Create test with informed expectations

#### Advanced Analytics
1. Visit **Predictive Analytics** page from navigation menu
2. Configure multiple test scenarios for comparison
3. Analyze capacity trends and network health forecasts
4. Generate performance reports for network planning

#### Understanding Prediction Results
```
< 30ms     = Excellent (gaming, video calls work perfectly)
30-60ms    = Good      (most applications work well)
60-150ms   = Fair      (web browsing fine, gaming may lag)
> 150ms    = Poor      (noticeable delays in interactive apps)
```

#### Test Type Impact on Predictions
- **Basic Ping**: Pure network latency measurement
- **Comprehensive**: Includes DNS, TCP, SSL analysis overhead (+10-20%)
- **Bandwidth Focus**: Latency during data transfer congestion (+15-30%)

### Model Training and Management
1. System automatically trains models as test data accumulates
2. Manual training available in **AI Models** section (Admin only)
3. Models require minimum 50 samples for initial training
4. Synthetic data generation ensures comprehensive scenario coverage
5. Models continuously improve with more real network data

### Prediction Enhancement System
- **ML Model**: Primary prediction using Gradient Boosting Regressor
- **Parameter Analysis**: Destination-specific adjustments (CDNs vs international)
- **Packet Size Impact**: Larger packets increase predicted latency
- **Fallback Protection**: Rule-based predictions if ML model unavailable

### Diagnostic Analysis Usage
1. Run network tests to collect performance data
2. Navigate to test results and click **"Diagnose Results"**
3. Review AI-generated health assessment and anomaly detection
4. Follow recommended actions for network optimization

### Performance Prediction Examples

#### Example 1: Comparing CDN vs Direct Server
```bash
# Prediction for CDN (CloudFlare)
Destination: cloudflare.com
Packet Size: 64 bytes
Test Type: Basic Ping
→ Predicted: ~22ms (Excellent)
→ Insight: "Excellent connectivity to major CDN"

# Prediction for International Server
Destination: spiegel.de  
Packet Size: 64 bytes
Test Type: Basic Ping
→ Predicted: ~42ms (Good)
→ Insight: "International destination - expect higher latency"
```

#### Example 2: Packet Size Impact Analysis
```bash
# Small packets
Packet Size: 64 bytes → Predicted: ~26ms
# Medium packets  
Packet Size: 512 bytes → Predicted: ~35ms
# Large packets
Packet Size: 1472 bytes → Predicted: ~105ms
→ Insight: "Larger packet size (1472B) adds ~79ms due to processing overhead"
```

#### Example 3: Test Type Comparison
```bash
# Basic latency test
Test Type: Basic Ping → Predicted: ~26ms
# Comprehensive analysis
Test Type: Comprehensive → Predicted: ~42ms  
# Bandwidth-focused test
Test Type: Bandwidth Focus → Predicted: ~46ms
→ Insight: "Comprehensive testing provides thorough analysis with additional overhead"
```

### Best Practices for Performance Prediction
1. **Test Multiple Scenarios**: Compare different packet sizes and destinations
2. **Validate Predictions**: Run actual tests to verify prediction accuracy
3. **Monitor Confidence Scores**: Higher confidence (>80%) indicates more reliable predictions
4. **Use for Planning**: Make infrastructure decisions based on predicted performance
5. **Regular Retraining**: Allow system to learn from new network conditions

## GNMI Network Path Analysis

StreamSwarm provides advanced network infrastructure analysis using GNMI (gRPC Network Management Interface) for managed enterprise network equipment.

### What is GNMI?
GNMI enables direct telemetry collection from network devices (routers, switches, firewalls) to provide hop-by-hop analysis within managed infrastructure:

- **Device-Level Metrics**: CPU utilization, memory usage, interface statistics
- **Processing Latency**: Packet forwarding delays within network devices
- **Queue Analysis**: Buffer utilization, queue depth, drop rates
- **Interface Performance**: Bandwidth utilization, error rates, packet rates

### Enterprise Benefits
- **Bottleneck Attribution**: Identify which specific network device causes performance issues
- **Proactive Monitoring**: Detect infrastructure problems before they impact users
- **Capacity Planning**: Understand device utilization for upgrade planning
- **Troubleshooting**: Precise problem isolation within managed network infrastructure

### Setup Requirements
1. **GNMI-Enabled Devices**: Modern enterprise network equipment with GNMI support
2. **Network Access**: Client must reach device management interfaces (ports 830, 32768, 6030, 57400)
3. **Authentication**: Multiple methods supported (username/password, certificates, hybrid)
4. **Python Package**: `pip install pygnmi>=0.8.15` on client systems
5. **Certificates**: For production deployments, generate or obtain client certificates

### Authentication Methods

StreamSwarm supports multiple GNMI authentication methods for enterprise security requirements:

#### Username/Password Authentication
```python
# Traditional authentication - good for testing
gnmi_analyzer.add_device("192.168.1.1", "admin", "password123", 830)
```

#### Certificate Authentication (Recommended)
```python
# Enhanced security with client certificates
gnmi_analyzer.add_device(
    device_ip="192.168.1.1",
    port=830,
    auth_method='certificate',
    cert_file='/etc/streamswarm/certs/client.crt',
    key_file='/etc/streamswarm/certs/client.key',
    ca_file='/etc/streamswarm/certs/ca.crt'  # Optional for custom CAs
)
```

#### Certificate + Username Authentication
```python
# Hybrid approach for specific device requirements
gnmi_analyzer.add_device(
    device_ip="10.0.1.5",
    username="admin",
    port=32768,
    auth_method='cert_username',
    cert_file='/etc/streamswarm/certs/client.crt',
    key_file='/etc/streamswarm/certs/client.key'
)
```

#### Certificate Generation
```bash
# Create certificate directory
sudo mkdir -p /etc/streamswarm/certs
cd /etc/streamswarm/certs

# Generate client certificate and key
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr \
  -subj "/CN=streamswarm-client/O=YourOrg/C=US"
openssl x509 -req -days 365 -in client.csr \
  -signkey client.key -out client.crt

# Set secure permissions
sudo chmod 600 client.key
sudo chmod 644 client.crt
```

### How It Works
1. Client performs standard traceroute to discover network path
2. For each hop with GNMI access, connects using configured authentication method
3. Analyzes device performance metrics and interface statistics  
4. Results displayed as "GNMI Managed Infrastructure Analysis" in test results
5. Graceful fallback when GNMI unavailable (tests continue normally)

### Supported Device Types
- Enterprise routers (Cisco, Juniper, Arista)
- Layer 3 switches with GNMI capabilities
- Next-generation firewalls
- SD-WAN appliances
- Any GNMI-compliant network device

## Web Dashboard Features

### Dashboard Page
- **System Overview**: View total clients, online status, and active tests
- **Real-time Charts**: Monitor network latency and system resources across all clients
- **Recent Activity**: See latest test results and client activity

### Clients Page
- **Client List**: View all connected clients with their status and system information
- **Client Details**: Click the info button to see detailed metrics and test history
- **Real-time Status**: Automatic updates every 30 seconds

### Tests Page
- **Create Tests**: Schedule network monitoring tests for multiple clients
- **Manage Tests**: Start, stop, and delete tests
- **View Results**: Comprehensive results with 65+ metrics per test

### Token Management
- **API Tokens**: Generate secure tokens for client authentication
- **Token Status**: Monitor token usage and client assignments
- **Security**: Separate from web authentication system

## Client Deployment

### Automated Linux Deployment (Recommended)
For production deployments, use the provided automated startup script:

**Files:**
- `start_streamswarm_client.sh` - Automated startup script with cron support
- `CLIENT_DEPLOYMENT.md` - Comprehensive deployment guide

**Quick Setup:**
1. Download client files to target machine
2. Edit `start_streamswarm_client.sh` with your server URL and API token
3. Make executable: `chmod +x start_streamswarm_client.sh`
4. Add to cron: `@reboot /path/to/start_streamswarm_client.sh`

The script automatically handles:
- Screen session management for background execution
- Python virtual environment activation
- Git repository updates (git pull origin main)
- Network connectivity waiting (important for @reboot)
- Comprehensive logging and error handling
- Status monitoring and client health checks

### Single Client
```bash
# Basic connection
python client.py --server http://192.168.1.100:5000

# Named client with verbose logging
python client.py --server http://monitoring.company.com:5000 --name "Web-Server-01" --verbose
```

### Multiple Locations
```bash
# New York Office
python client.py --server http://monitoring.company.com:5000 --name "NYC-Office-Gateway"

# London Office  
python client.py --server http://monitoring.company.com:5000 --name "London-Office-Gateway"

# Tokyo Office
python client.py --server http://monitoring.company.com:5000 --name "Tokyo-Office-Gateway"
```

### Service Installation

**Linux (systemd):**
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

**Windows Service:**
```batch
# Install as Windows service using NSSM
nssm install StreamSwarmClient "C:\Python\python.exe"
nssm set StreamSwarmClient Parameters "C:\StreamSwarm\client.py --server http://monitoring.company.com:5000 --name %COMPUTERNAME%"
nssm set StreamSwarmClient Start SERVICE_AUTO_START
nssm start StreamSwarmClient
```

## Monitoring Metrics

StreamSwarm collects 65+ comprehensive metrics:

### Network Performance
- Ping latency, packet loss, jitter
- Bandwidth upload/download speeds
- DNS resolution time, TCP connect time
- SSL handshake time, time to first byte

### Quality of Service (QoS)
- DSCP/CoS detection and classification
- Per-class latency measurements
- Traffic policing detection
- ECN (Explicit Congestion Notification) analysis

### System Resources
- CPU usage, load averages, frequency
- Memory usage, cache, buffers, swap
- Disk usage, IOPS, throughput
- Network interface statistics

### Advanced Network Analysis
- MTU discovery and path analysis
- TCP window scaling and congestion control
- Retransmission rates and out-of-order packets
- Wireless signal strength and channel analysis

### Application Layer
- HTTP response code analysis
- Content download timing
- Compression effectiveness
- Certificate validation timing

### Infrastructure Health
- Power consumption monitoring
- Hardware temperatures
- Fan speeds and drive health
- Memory error detection

## Production Deployment

### Environment Configuration
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

### Security Best Practices
1. Change default admin password immediately
2. Use strong, unique SESSION_SECRET
3. Configure PostgreSQL with secure credentials
4. Enable HTTPS with reverse proxy (nginx/Apache)
5. Use firewall to restrict access to necessary ports
6. Regular security updates for all dependencies

## Firewall Requirements and External Dependencies

StreamSwarm requires specific network access for various features. Understanding these requirements helps configure firewalls and plan deployments.

### Server-Side External Dependencies

#### Required Internet Access
- **Geolocation Services**: `ip-api.com` (HTTP port 80)
  - **Purpose**: IP geolocation lookup for network path visualization  
  - **Feature Impact**: Without access, geolocation maps will not generate
  - **Alternative**: Disable geolocation features for air-gapped environments

#### System Package Dependencies
```bash
# Network tools (required for core functionality)
ping, traceroute, mtr

# Wireless monitoring (optional)
iw, wireless-tools, iwconfig

# System monitoring (optional advanced features)
lm-sensors, smartmontools, ethtool
```

### Client-Side External Dependencies

#### Required Internet Access
- **StreamSwarm Server**: HTTP/HTTPS connection to your server
  - **Ports**: 80/443 (default) or custom port configured
  - **Feature Impact**: Core functionality requires server connectivity

#### Optional Internet Access
- **Speed Test Servers**: Various HTTP/HTTPS endpoints
  - **Ports**: 80/443 to multiple speed test providers
  - **Feature Impact**: Bandwidth testing requires access to speed test endpoints
  - **Examples**: speedtest.net, fast.com, Google speed test servers

- **DNS Resolution**: UDP port 53 to DNS servers
  - **Purpose**: Hostname resolution for network targets
  - **Feature Impact**: Tests using hostnames instead of IP addresses

#### Network Tools Required
```bash
# Core network tools (usually pre-installed)
ping, traceroute

# Advanced network analysis (optional)
mtr, tcpdump
```

### Feature-Specific Requirements

| Feature | External Access Required | Ports/Protocols | Impact if Blocked |
|---------|-------------------------|------------------|-------------------|
| **Core Monitoring** | StreamSwarm Server | HTTP/HTTPS | Complete loss of functionality |
| **Bandwidth Testing** | Speed test servers | HTTP/HTTPS (80/443) | No bandwidth measurements |
| **Geolocation Maps** | ip-api.com | HTTP (80) | No network path visualization |
| **DNS Tests** | DNS servers | UDP (53) | Hostname-based tests fail |
| **System Monitoring** | None | Local only | Always works offline |

### Air-Gapped/Isolated Environments

StreamSwarm supports restricted environments with limited internet access:

#### Fully Offline Capabilities
- System resource monitoring (CPU, memory, disk)
- Local network interface statistics  
- Ping tests to internal IP addresses
- Basic network connectivity tests

#### Recommendations for Restricted Environments
1. **Disable geolocation features** if ip-api.com access unavailable
2. **Use IP addresses** instead of hostnames for network targets
3. **Configure internal speed test servers** for bandwidth testing
4. **Use SQLite database** to avoid external database dependencies

### Firewall Configuration Examples

#### Server Firewall (iptables)
```bash
# Allow inbound connections to StreamSwarm server
iptables -A INPUT -p tcp --dport 5000 -j ACCEPT

# Allow outbound for geolocation services
iptables -A OUTPUT -p tcp --dport 80 -d ip-api.com -j ACCEPT

# Allow outbound DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
```

#### Corporate Firewall Rules
```
# Required: Client to Server communication
ALLOW TCP 80/443 FROM client_subnets TO streamswarm_server

# Optional: Speed test access (if bandwidth testing needed)
ALLOW TCP 80/443 FROM client_subnets TO speedtest_servers

# Optional: Geolocation services (server-side)
ALLOW TCP 80 FROM streamswarm_server TO ip-api.com

# Required: DNS resolution
ALLOW UDP 53 FROM any TO dns_servers
```

## Enhanced Recurring Tests

StreamSwarm offers advanced recurring test functionality with two distinct behavior modes to support different monitoring strategies:

### Recurrence Behavior Options

#### Continue Same Test (Real-Time Monitoring)
- **Best for**: Dashboards, alerting systems, real-time monitoring
- **Behavior**: Reuses the same test record for each execution, overwriting previous results
- **Benefits**: Efficient resource usage, single test ID, ideal for continuous monitoring
- **Use cases**: Service availability checks, network health monitoring, performance alerting

#### Create New Tests (Historical Tracking)
- **Best for**: Trend analysis, compliance reporting, historical data collection
- **Behavior**: Creates a new test record for each scheduled execution
- **Benefits**: Preserves all historical results, enables detailed trending analysis
- **Use cases**: SLA compliance reporting, capacity planning, performance trend analysis

### Configuration Guidelines

**Minimum Requirements:**
- Recurrence interval must be at least test duration + 10 minutes
- Sufficient client resources for all recurring tests
- Consider storage requirements for "Create New Tests" mode

**Best Practices:**
- Use "Continue Same Test" for real-time dashboards and alerting
- Use "Create New Tests" for regulatory compliance and historical reporting
- Monitor client allocation to ensure recurring tests don't conflict
- Set appropriate recurrence intervals based on network requirements

**Example Configurations:**

*Website SLA Monitoring:*
- Test Duration: 5 minutes
- Recurrence: Every 15 minutes
- Mode: Continue Same Test
- Result: Real-time availability dashboard

*Monthly Performance Reports:*
- Test Duration: 10 minutes
- Recurrence: Daily at 2 AM
- Mode: Create New Tests
- Result: Historical trend data for reporting

## Troubleshooting

### Common Issues

**Permission Errors:**
```bash
# Linux: Use virtual environment or user install
python3 -m pip install --user <package>
# Or create virtual environment (recommended)
python3 -m venv env && source env/bin/activate && pip install <package>
```

**Database Connection Issues:**
```bash
# Test database connection
psql "$DATABASE_URL" -c "SELECT version();"

# Check environment variables
echo $DATABASE_URL
echo $SESSION_SECRET
```

**Client Connection Problems:**
```bash
# Check server accessibility
curl http://your-server:5000/api/status

# Verify API token
python client.py --server http://your-server:5000 --verbose
```

**Wireless Detection Issues:**
```bash
# Install wireless tools
sudo apt install iw wireless-tools libiw-dev network-manager

# Test wireless detection
iw dev
iwconfig
```

### Support and Documentation

- **In-App Tutorial**: Access via navigation menu after login
- **API Documentation**: Available in web interface
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Share use cases and configurations

## Development

### Project Structure
```
StreamSwarm/
├── app.py              # Flask application setup
├── main.py             # Application entry point
├── models.py           # Database models
├── routes.py           # API and web routes
├── client.py           # Monitoring client
├── ml_diagnostics.py   # AI/ML analysis engine
├── pdf_generator.py    # Report generation
├── config.py           # Configuration settings
├── templates/          # Web interface templates
├── static/            # CSS, JS, images
└── ml_models/         # Trained ML models
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Changelog

### Recent Updates (June 2025)
- **✅ Complete Authentication System**: Flask-Login based with role-based access control
- **✅ AI/ML Diagnostics**: Local Scikit-learn models for network analysis  
- **✅ Enhanced Security**: Protected all web routes with authentication
- **✅ User Management**: Admin interface for user account management
- **✅ Self-Service Features**: User profile and password change functionality
- **✅ Visual Branding**: Integrated SWARM logo with dark theme
- **✅ Comprehensive Documentation**: Consolidated all guides into single README

### Technical Achievements
- **769 test results** analyzed with **99% ML classification accuracy**
- **65+ monitoring metrics** across network, system, and application layers
- **Zero-trust architecture** with local ML processing
- **Separate authentication systems** for web GUI and client API
- **Professional PDF reporting** with charts and recommendations
- **Real-time dashboard** with interactive visualizations