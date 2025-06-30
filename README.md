# StreamSwarm - Distributed Network Monitoring System

A comprehensive Python-based distributed network monitoring system with AI-powered diagnostics, secure web authentication, and real-time analytics for enterprise network management.

## Overview

StreamSwarm provides distributed network performance testing and system resource monitoring across multiple client hosts. The system features a central Flask server with a web dashboard for managing clients, scheduling tests, and visualizing performance data with AI-powered diagnostics and professional PDF reporting.

## Key Features

- **🤖 AI-Powered Network Diagnostics**: Local machine learning models for anomaly detection, health classification, and intelligent troubleshooting recommendations
- **🔒 Zero-Trust ML Architecture**: All AI processing runs locally using Scikit-learn with no external dependencies or cloud connections
- **🌐 Comprehensive Network Analysis**: Latency, packet loss, jitter, bandwidth, MTU discovery, TCP analysis, QoS monitoring
- **📊 Advanced Metrics Collection**: 65+ performance metrics including application layer, infrastructure health, and system resources
- **🏢 Multi-Client Architecture**: Distributed testing from multiple network locations with automatic client assignment
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
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0 scikit-learn>=1.0.0 pandas>=1.3.0 numpy>=1.20.0 joblib>=1.0.0

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

StreamSwarm includes local machine learning capabilities for intelligent network analysis.

### Features
- **Anomaly Detection**: Isolation Forest algorithm identifies unusual network patterns
- **Health Classification**: Random Forest algorithm categorizes network health status
- **Performance Prediction**: Gradient Boosting predicts network performance trends
- **Feature Engineering**: Analyzes 25+ metrics including network performance, system resources, and QoS data

### Zero-Trust Architecture
- All ML processing runs locally
- No external dependencies or cloud connections
- Uses Scikit-learn for reliable, offline analysis
- Models are trained on your own network data

### Usage
1. Collect test data (minimum 50 samples for training)
2. Navigate to "AI Models" section
3. Click "Train Models" to build custom models
4. Use "Diagnose Results" on completed tests for AI analysis

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