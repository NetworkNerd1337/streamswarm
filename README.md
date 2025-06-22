# StreamSwarm

A Python-based client-server network monitoring system with web dashboard for distributed testing and analysis.

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that collects 65+ performance metrics including network testing, bandwidth measurement, QoS analysis, application-layer monitoring, performance profiling, and infrastructure health across multiple client hosts. The system features a centralized web dashboard for managing clients, scheduling tests, and visualizing comprehensive performance data with professional PDF reporting.

## Features

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

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install flask flask-sqlalchemy psutil requests gunicorn psycopg2-binary speedtest-cli reportlab matplotlib scapy email-validator werkzeug
   sudo apt install iputils-ping traceroute lm-sensors smartmontools ethtool libpcap-dev tcpdump
   ```

2. **Start the server:**
   ```bash
   python main.py
   ```

3. **Connect clients:**
   ```bash
   python client.py --server http://your-server-ip:5000
   ```

3. **Access dashboard:** Open `http://localhost:5000` in your browser

4. **Create tests:** Use the web interface to schedule network monitoring tests

See [USAGE.md](USAGE.md) for detailed instructions.

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

### Server Requirements
- Python 3.7+
- PostgreSQL (recommended) or SQLite
- 1GB RAM minimum, 2GB+ recommended
- Network access for client connections

### Client Requirements
- Python 3.7+
- Network utilities (ping, traceroute)
- Outbound network access to server

### Setup Instructions

1. **Clone or download the application files**

2. **Install Python dependencies:**
   ```bash
   pip install flask flask-sqlalchemy psutil requests gunicorn psycopg2-binary
   ```

3. **Configure database (optional):**
   ```bash
   export DATABASE_URL=postgresql://user:pass@localhost/streamswarm
   ```

4. **Start the server:**
   ```bash
   python main.py
   ```

5. **Deploy clients on monitoring hosts:**
   ```bash
   python client.py --server http://server-ip:5000
   ```

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

