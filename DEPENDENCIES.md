# StreamSwarm Dependencies

This document provides comprehensive installation instructions for StreamSwarm's distributed network monitoring system with 65+ performance metrics across Linux and Windows environments.

## Python Installation

### Linux Installation

#### Ubuntu/Debian
```bash
# Update package repositories
sudo apt update

# Install Python 3.9+ and development tools
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev
sudo apt install build-essential libffi-dev libssl-dev

# Install system monitoring dependencies
sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool
sudo apt install libpcap-dev tcpdump

# Create symbolic links for convenience (optional)
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
sudo ln -sf /usr/bin/pip3.9 /usr/bin/pip
```

#### CentOS/RHEL/Fedora
```bash
# Install Python and development tools
sudo dnf install python3.9 python3.9-pip python3.9-devel gcc openssl-devel libffi-devel

# Install system monitoring dependencies
sudo dnf install iputils traceroute lm_sensors smartmontools ethtool
sudo dnf install libpcap-devel tcpdump

# For older versions, replace 'dnf' with 'yum'
```

### Windows Installation

#### Method 1: Official Python Installer (Recommended)
1. Visit https://www.python.org/downloads/windows/
2. Download Python 3.9+ installer (64-bit recommended)
3. Run installer with these options:
   - ✅ Add Python to PATH
   - ✅ Install for all users
   - ✅ Include pip
4. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### Method 2: Package Managers
```powershell
# Using Chocolatey
choco install python

# Using winget (Windows 10/11)
winget install Python.Python.3.11

# Using Scoop
scoop install python
```

### macOS Installation

#### Using Homebrew (Recommended)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install system dependencies
brew install libpcap
```

#### Official Installer
1. Visit https://www.python.org/downloads/mac-osx/
2. Download Python 3.9+ installer
3. Follow installation wizard

## Python Package Requirements

Copy and paste the following commands to install all required dependencies:

```bash
# Core application dependencies
pip install flask>=2.3.0
pip install flask-sqlalchemy>=3.0.0
pip install sqlalchemy>=2.0.0
pip install psycopg2-binary>=2.9.0
pip install psutil>=5.9.0
pip install requests>=2.28.0
pip install gunicorn>=21.0.0
pip install werkzeug>=2.3.0
pip install email-validator>=2.0.0

# Advanced network analysis and QoS monitoring
pip install scapy>=2.5.0

# Bandwidth testing capabilities
pip install speedtest-cli>=2.1.3

# Professional report generation and visualization
pip install reportlab>=4.4.0
pip install matplotlib>=3.10.0
```

## Single Command Installation

Install all dependencies at once:

```bash
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0
```

## Package Descriptions

- **flask**: Web framework for the server component
- **flask-sqlalchemy**: Database ORM integration
- **sqlalchemy**: Database abstraction layer
- **psycopg2-binary**: PostgreSQL database adapter
- **psutil**: System resource monitoring
- **requests**: HTTP client for client-server communication
- **gunicorn**: Production WSGI server
- **werkzeug**: WSGI utilities
- **email-validator**: Email validation utilities
- **scapy**: Advanced packet analysis for QoS monitoring, DSCP/CoS detection, and traffic classification
- **speedtest-cli**: Internet bandwidth testing and speed measurement with multi-method approach
- **reportlab**: Professional PDF document generation with executive-level reporting
- **matplotlib**: Chart and graph generation for performance trend analysis

## Optional: Create requirements.txt

If you prefer using a requirements.txt file, create one with this content:

```
flask>=2.3.0
flask-sqlalchemy>=3.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
psutil>=5.9.0
requests>=2.28.0
gunicorn>=21.0.0
werkzeug>=2.3.0
email-validator>=2.0.0
scapy>=2.5.0
speedtest-cli>=2.1.3
reportlab>=4.4.0
matplotlib>=3.10.0
```

Then install with:
```bash
pip install -r requirements.txt
```

## System Dependencies by Platform

### Linux System Dependencies

#### Ubuntu/Debian
```bash
# Essential network tools
sudo apt install iputils-ping traceroute

# Advanced monitoring tools
sudo apt install lm-sensors smartmontools ethtool

# Development libraries for packet analysis
sudo apt install libpcap-dev tcpdump

# Optional: Hardware monitoring setup
sudo sensors-detect --auto
```

#### CentOS/RHEL/Fedora
```bash
# Essential network tools
sudo dnf install iputils traceroute

# Advanced monitoring tools
sudo dnf install lm_sensors smartmontools ethtool

# Development libraries
sudo dnf install libpcap-devel tcpdump

# Initialize sensors
sudo sensors-detect --auto
```

### Windows System Dependencies

**Most functionality works without additional tools, but for enhanced features:**

```powershell
# Optional: Install Wireshark for advanced packet analysis
# Download from: https://www.wireshark.org/download.html

# Optional: Windows Subsystem for Linux (WSL) for Unix tools
wsl --install
```

**Windows-specific notes:**
- Admin privileges required for some system monitoring features
- Windows Defender may require exclusions for network monitoring
- Some Unix-specific tools have Windows alternatives built into the application

### macOS System Dependencies

```bash
# Install packet capture library
brew install libpcap

# Optional: Install network monitoring tools
brew install nmap traceroute

# Note: Some features may require admin privileges
```

## Virtual Environment Setup (Recommended)

### Linux/macOS
```bash
# Create virtual environment
python3 -m venv streamswarm-env

# Activate virtual environment
source streamswarm-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Windows
```cmd
# Create virtual environment
python -m venv streamswarm-env

# Activate virtual environment
streamswarm-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Installation Verification

After installing dependencies, verify the setup:

```python
# Test script to verify dependencies
import flask
import psutil
import requests
import scapy
import speedtest
import reportlab
import matplotlib
print("✓ All dependencies installed successfully")
```

## Troubleshooting

### Common Installation Issues

**Linux Permission Errors:**
```bash
# If pip permission denied
python3 -m pip install --user <package>

# Or use virtual environment (recommended)
python3 -m venv env && source env/bin/activate
```

**Windows Long Path Issues:**
```cmd
# Enable long paths in Windows
# Run as Administrator:
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

**macOS Permission Issues:**
```bash
# If system integrity protection blocks installation
pip install --user <package>

# Or use Homebrew Python instead of system Python
brew install python@3.9
```

**Network Monitoring Permissions:**
- Linux: Run with `sudo` for advanced network monitoring
- Windows: Run as Administrator for system-level metrics
- macOS: Some features require admin privileges

## Performance Optimization

### Database Setup for Production

**PostgreSQL (Recommended):**
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# CentOS/RHEL
sudo dnf install postgresql postgresql-server postgresql-contrib

# Windows: Download from https://www.postgresql.org/download/windows/
# macOS: brew install postgresql
```

**Configure database:**
```sql
CREATE DATABASE streamswarm;
CREATE USER streamswarm_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE streamswarm TO streamswarm_user;
```

### Memory and Performance

**Recommended system specifications:**
- **Development:** 2GB RAM, 2 CPU cores
- **Small deployment:** 4GB RAM, 4 CPU cores
- **Large deployment:** 8GB+ RAM, 8+ CPU cores

**Tuning for high-frequency monitoring:**
```bash
# Increase file descriptor limits (Linux)
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf
```