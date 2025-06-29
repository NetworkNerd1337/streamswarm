# StreamSwarm

A Python-based client-server network monitoring system with AI-powered diagnostics, web dashboard, and zero-trust ML capabilities for distributed testing and analysis.

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that collects 65+ performance metrics including network testing, bandwidth measurement, QoS analysis, application-layer monitoring, performance profiling, and infrastructure health across multiple client hosts. The system features AI-powered diagnostic capabilities using local machine learning models, a centralized web dashboard for managing clients, scheduling tests, and visualizing comprehensive performance data with professional PDF reporting.

## Features

- **AI-Powered Network Diagnostics**: Local machine learning models for anomaly detection, health classification, and intelligent troubleshooting recommendations
- **Zero-Trust ML Architecture**: All AI processing runs locally using Scikit-learn with no external dependencies or cloud connections
- **Network Performance Monitoring**: Comprehensive latency, packet loss, jitter, bandwidth, and connectivity testing
- **Advanced Network Analysis**: MTU discovery, TCP window scaling, congestion control, retransmission analysis
- **Quality of Service**: DSCP/CoS detection, per-class latency, traffic policing, ECN analysis
- **Application Layer Monitoring**: HTTP response analysis, content timing, compression, certificate validation
- **Performance Profiling**: DNS/HTTP cache effectiveness, CDN performance, application response timing
- **Infrastructure Health**: Power consumption, fan speeds, SMART drive health, memory error detection
- **System Resource Tracking**: CPU, memory, disk, and network interface monitoring (65+ total metrics)
- **Multi-Client Architecture**: Distributed testing from multiple network locations with auto-assignment
- **Professional Reporting**: Executive PDF reports with charts, analysis, and recommendations
- **Real-Time Visualization**: Web-based interface with comprehensive charts and test scheduling
- **Secure Web Authentication**: Role-based access control with user management and session security

## Quick Start

1. **Download StreamSwarm:**
   ```bash
   git clone https://github.com/NetworkNerd1337/Swarm.git
   cd Swarm
   ```

2. **Install Dependencies:**
   ```bash
   # Python packages (server)
   pip install flask flask-sqlalchemy psutil requests gunicorn psycopg2-binary speedtest-cli reportlab matplotlib scapy email-validator werkzeug scikit-learn pandas numpy joblib
   
   # System packages (Ubuntu/Debian)
   sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool libpcap-dev tcpdump
   sudo apt install iw wireless-tools libiw-dev network-manager
   
   # AI/ML dependencies (optional but recommended for server)
   sudo apt install python3-numpy python3-scipy python3-sklearn
   sudo apt install libatlas-base-dev liblapack-dev gfortran
   ```

3. **Start the server:**
   ```bash
   python main.py
   ```

4. **Access the web interface:**
   - Open browser to `http://localhost:5000`
   - **Default login:** username=`admin`, password=`admin123`
   - **⚠️ Important:** Change the default password immediately!
   - Access via: Username dropdown → "My Profile"

4. **Connect clients:**
   ```bash
   python client.py --server http://your-server-ip:5000
   ```

5. **Access dashboard:** Open `http://localhost:5000` in your browser

6. **Create tests:** Use the web interface to schedule network monitoring tests

See [USAGE.md](USAGE.md) for detailed instructions.

## AI/ML Diagnostic System

StreamSwarm includes an advanced AI diagnostic system that provides intelligent analysis of network performance data using local machine learning models.

### Machine Learning Features

**Core Capabilities:**
- **Anomaly Detection**: Identifies unusual network patterns using Isolation Forest algorithm
- **Health Classification**: Classifies network health status using Random Forest
- **Performance Prediction**: Predicts network performance trends using Gradient Boosting
- **Issue Classification**: Automatically categorizes and prioritizes network issues
- **Intelligent Recommendations**: Provides specific troubleshooting guidance

**Zero-Trust Architecture:**
- All ML processing runs locally on your server using Scikit-learn
- No external API calls or cloud dependencies required
- Training data never leaves your network infrastructure
- Models are stored locally in the `ml_models/` directory

### AI Usage Quick Start

1. **Collect Data**: Run at least 50 network monitoring tests
2. **Train Models**: Visit "AI Models" page and click "Train Models"
3. **Run Diagnostics**: Click "Diagnose Results" on any completed test
4. **Review Analysis**: Get health scores, issue detection, and recommendations

**Health Score Interpretation:**
- **80-100**: Healthy network performance
- **60-79**: Warning - some issues detected  
- **0-59**: Critical - significant problems identified

### Supported ML Models

- **Isolation Forest**: Anomaly detection in network metrics
- **Random Forest**: Health status classification
- **Gradient Boosting**: Performance trend prediction
- **Feature Engineering**: 25+ metrics including latency, bandwidth, system resources

See [TUTORIAL.md](TUTORIAL.md) for comprehensive AI/ML setup and usage instructions.

## Architecture

### System Components

- **Server Application**: Flask web server with PostgreSQL database
- **Web Dashboard**: Bootstrap-based interface with Chart.js visualization
- **Client Application**: Lightweight monitoring agent for remote hosts
- **Database**: PostgreSQL for reliable data storage and querying

### Data Flow

1. Clients register with server and send periodic heartbeats
2. Tests are created and scheduled through web interface
3. Server assigns tests to available clients
4. Clients execute tests and stream results back to server
5. Web dashboard displays real-time charts and historical data

## Installation

### System Requirements

#### Server Requirements
- Python 3.8+ (3.9+ recommended)
- PostgreSQL (recommended) or SQLite
- 2GB RAM minimum, 4GB+ recommended for AI/ML features
- Network access for client connections
- Administrative privileges for system monitoring
- Additional ML dependencies: scikit-learn, pandas, numpy, joblib

#### Client Requirements
- Python 3.8+ (3.9+ recommended)
- Network utilities (ping, traceroute)
- Outbound network access to server
- Administrative privileges for advanced metrics

### Python Installation

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python 3.9+ and pip
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev

# Install system dependencies for advanced monitoring
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool
sudo apt install libpcap-dev tcpdump build-essential

# Create symbolic links (optional)
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
sudo ln -sf /usr/bin/pip3.9 /usr/bin/pip
```

#### Linux (CentOS/RHEL/Fedora)
```bash
# Install Python 3.9+ and development tools
sudo dnf install python3.9 python3.9-pip python3.9-devel gcc

# Install system dependencies
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool
sudo dnf install libpcap-devel tcpdump

# For older versions, use yum instead of dnf
```

#### Windows
```powershell
# Option 1: Download from python.org
# Visit https://www.python.org/downloads/windows/
# Download Python 3.9+ installer and run with "Add to PATH" checked

# Option 2: Using Chocolatey (if installed)
choco install python

# Option 3: Using winget (Windows 10/11)
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

#### macOS
```bash
# Option 1: Using Homebrew (recommended)
brew install python@3.9

# Option 2: Download from python.org
# Visit https://www.python.org/downloads/mac-osx/

# Install system dependencies
brew install libpcap

# Verify installation
python3 --version
pip3 --version
```

### Application Setup

#### 1. Download StreamSwarm Application
```bash
# Clone from GitHub repository
git clone https://github.com/NetworkNerd1337/Swarm.git
cd Swarm

# Or download as ZIP file
# Visit https://github.com/NetworkNerd1337/Swarm
# Click "Code" → "Download ZIP" → Extract to desired location
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Linux/macOS
python3 -m venv streamswarm-env
source streamswarm-env/bin/activate

# Windows
python -m venv streamswarm-env
streamswarm-env\Scripts\activate
```

#### 3. Install Python Dependencies
```bash
# All dependencies in one command
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0

# Or install from requirements file (if available)
pip install -r requirements.txt
```

#### 4. Install System Dependencies

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool libpcap-dev tcpdump

# CentOS/RHEL/Fedora
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool libpcap-devel tcpdump
```

**Windows:**
```powershell
# Most functionality works without additional tools
# For advanced features, consider installing:
# - Wireshark (for packet analysis)
# - Windows Subsystem for Linux (WSL) for Unix tools
```

#### 5. Configure Database (Optional)
```bash
# For PostgreSQL (recommended for production)
export DATABASE_URL=postgresql://user:password@localhost/streamswarm

# Windows PowerShell
$env:DATABASE_URL="postgresql://user:password@localhost/streamswarm"
```

#### 6. Start the Server
```bash
# Development mode
python main.py

# Production mode with Gunicorn (Linux/macOS)
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app

# Windows production (use waitress)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 main:app
```

#### 7. Deploy Clients on Monitoring Hosts
```bash
# On each client machine
python client.py --server http://server-ip:5000 --name "Client-Name"
```

### Verification

After installation, verify everything works:

1. **Server verification:** Access `http://localhost:5000` in your browser
2. **Client verification:** Check client appears in the dashboard
3. **Create a test:** Use the web interface to schedule a monitoring test
4. **View results:** Confirm data collection and visualization works

### Troubleshooting

**Common Issues:**

- **Permission errors:** Run with `sudo` on Linux/macOS or as Administrator on Windows
- **Port conflicts:** Change port in configuration or stop conflicting services
- **Module import errors:** Ensure all dependencies are installed in the correct Python environment
- **Network connectivity:** Verify firewall settings allow connections on port 5000

## Configuration

The application can be configured using environment variables:

- `DATABASE_URL`: Database connection string
- `SERVER_HOST`: Server bind address (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 5000)
- `DEFAULT_TEST_DURATION`: Default test duration in seconds (default: 300)
- `DEFAULT_TEST_INTERVAL`: Default measurement interval (default: 5)

## Use Cases

- **Network Operations**: Monitor connectivity and performance across multiple locations
- **Infrastructure Monitoring**: Track system resources on distributed servers
- **Performance Testing**: Measure network paths and identify bottlenecks
- **Capacity Planning**: Collect historical data for infrastructure decisions
- **Troubleshooting**: Correlate network and system metrics during incidents

