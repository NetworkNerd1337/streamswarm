# StreamSwarm

A Python-based client-server network monitoring system with web dashboard for distributed testing and analysis.

## Overview

StreamSwarm enables distributed network monitoring by deploying lightweight clients across multiple hosts to measure network performance, system resources, and connectivity to specified destinations. All data is collected and analyzed through a centralized web dashboard.

## Features

- **Distributed Client Architecture**: Deploy clients on multiple hosts for comprehensive network monitoring
- **Real-time System Monitoring**: CPU, memory, and disk usage tracking
- **Network Performance Testing**: Ping, traceroute, and latency measurements
- **Web Dashboard**: Interactive charts and graphs for data visualization
- **Test Scheduling**: Create and schedule tests for multiple clients simultaneously
- **Multi-client Support**: Manage and monitor multiple clients from a single interface
- **SQLite Database**: Reliable data storage with easy backup and migration

## Quick Start

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Connect clients:**
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

