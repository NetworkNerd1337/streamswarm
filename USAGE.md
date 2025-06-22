# StreamSwarm Usage Guide

StreamSwarm is a distributed network monitoring system that enables real-time network performance testing and system resource monitoring across multiple client hosts.

## Quick Start

### 1. Start the Server

Run the server application:
```bash
python main.py
```

The web dashboard will be available at `http://localhost:5000`

### 2. Connect Clients

On each host you want to monitor, run the client:
```bash
python client.py --server http://your-server-ip:5000
```

Optional parameters:
- `--name`: Custom client name (defaults to hostname)
- `--verbose`: Enable detailed logging

Example:
```bash
python client.py --server http://192.168.1.100:5000 --name "Web-Server-01" --verbose
```

### 3. Access the Dashboard

Open your web browser and navigate to the server URL. You'll see three main sections:

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
- **Real-time Progress**: Watch test progress with live progress bars

## Creating and Running Tests

### 1. Create a New Test

1. Go to the **Tests** page
2. Click **"Create Test"**
3. Fill in the test parameters:
   - **Test Name**: Descriptive name for your test
   - **Destination**: Target hostname or IP address (e.g., google.com, 8.8.8.8)
   - **Description**: Optional description
   - **Duration**: How long the test should run (in seconds)
   - **Interval**: How often to collect data (in seconds)
   - **Schedule**: Optional future start time
   - **Select Clients**: Choose which connected clients should participate

### 2. Monitor Test Results

1. Click the chart icon next to any test to view detailed results
2. View real-time charts showing:
   - Network latency over time
   - Packet loss statistics
   - CPU, memory, and disk usage for each client
3. Compare performance across multiple clients
4. Export data for further analysis

## Client System Requirements

### Supported Operating Systems
- Ubuntu/Debian Linux
- CentOS/RHEL Linux
- macOS
- Windows 10/11

### Required Packages
- Python 3.7+
- pip (Python package manager)

### Network Requirements
- Outbound HTTP/HTTPS access to the server
- Ability to run ping and traceroute commands
- No special firewall rules needed (clients connect to server)

## Understanding Test Data

### Network Metrics
- **Ping Latency**: Round-trip time to destination in milliseconds
- **Packet Loss**: Percentage of packets that failed to reach destination
- **Traceroute Hops**: Number of network hops to reach destination
- **DNS Resolution Time**: Time to resolve hostnames to IP addresses
- **TCP Connect Time**: Time to establish TCP connections
- **SSL Handshake Time**: Time to complete SSL/TLS handshakes
- **Jitter**: Network timing variability measurement

### Bandwidth Performance
- **Upload Speed**: Internet upload bandwidth in Mbps (using speedtest-cli)
- **Download Speed**: Internet download bandwidth in Mbps (using speedtest-cli)
- **HTTP Bandwidth**: Direct throughput testing to specific destinations
- **TCP Throughput**: Custom socket-based bandwidth measurement

### System Metrics
- **CPU Usage**: Current processor utilization percentage
- **Memory Usage**: RAM utilization percentage  
- **Disk Usage**: Storage utilization percentage
- **Load Averages**: 1, 5, and 15-minute system load averages
- **Network Interface Stats**: Bytes/packets sent/received, errors, drops
- **Process Monitoring**: Active processes, TCP connections, file descriptors

### Quality of Service
- **DSCP Values**: Differentiated Services Code Point markings
- **CoS Values**: Class of Service markings (802.1Q)
- **Traffic Classification**: Automatic traffic type detection
- **QoS Policy Compliance**: Validation against expected policies
- **Per-DSCP Latency**: Latency measurements by traffic class
- **Traffic Policing Detection**: Rate limiting and shaping identification
- **ECN Analysis**: Explicit Congestion Notification capability testing
- **Queue Depth Estimation**: Network buffer analysis

### Advanced Network Analysis
- **MTU Discovery**: Path MTU detection with fragmentation analysis
- **TCP Window Analysis**: Window scaling and congestion control monitoring
- **Retransmission Analysis**: Packet loss and recovery pattern detection
- **Connection Analysis**: Out-of-order packets and duplicate ACK tracking
- **Flow Control Monitoring**: TCP/UDP flow control event detection

### Application Layer Performance
- **HTTP Response Analysis**: Status code tracking (2xx, 3xx, 4xx, 5xx)
- **Content Performance**: Download timing and compression analysis
- **Connection Optimization**: Keep-alive and reuse effectiveness
- **Certificate Analysis**: SSL/TLS validation timing
- **CDN Performance**: Content delivery network effectiveness scoring

### Performance Profiling
- **Cache Analysis**: DNS and HTTP cache hit ratios
- **Application Timing**: End-to-end response time measurement
- **Database Performance**: Query time estimation for web applications
- **Multipath Detection**: MPTCP and ECMP path diversity analysis

### Infrastructure Health Monitoring
- **Power Management**: Energy consumption tracking where accessible
- **Cooling Systems**: Fan speed and thermal monitoring
- **Storage Health**: SMART drive health with temperature tracking
- **Memory Integrity**: ECC error detection and reporting
- **Network Hardware**: Physical layer error rates and statistics

## Troubleshooting

### Client Won't Connect
1. Verify server is running and accessible
2. Check firewall settings on server
3. Ensure client can reach server IP/hostname
4. Verify port 5000 is not blocked

### No Test Data Appearing
1. Confirm clients are online (green status)
2. Check that tests are assigned to online clients
3. Verify destination is reachable from client networks
4. Review client logs for error messages
5. Check bandwidth testing requirements (speedtest-cli installed)
6. Verify network permissions for QoS monitoring (may require root)
7. Ensure system monitoring tools are installed (lm-sensors, smartctl, ethtool)
8. Check hardware access permissions for infrastructure monitoring

### Performance Issues
1. Reduce test frequency (increase interval)
2. Limit number of concurrent tests
3. Monitor server resource usage
4. Consider database optimization for large datasets
5. Disable bandwidth testing for high-frequency tests
6. Use HTTP bandwidth testing instead of speedtest-cli for faster results
7. Limit infrastructure monitoring on systems without hardware access
8. Consider client permissions for comprehensive metric collection

## Advanced Configuration

### Environment Variables

You can customize the application behavior using environment variables:

```bash
# Server Configuration
export SERVER_HOST=0.0.0.0
export SERVER_PORT=5000
export DEBUG=False

# Database Configuration
export DATABASE_URL=postgresql://user:pass@localhost/streamswarm

# Test Defaults
export DEFAULT_TEST_DURATION=300
export DEFAULT_TEST_INTERVAL=5
export CLIENT_HEARTBEAT_INTERVAL=30

# Bandwidth Testing Configuration
export SPEEDTEST_ENABLED=true
export HTTP_BANDWIDTH_TEST_SIZE=10485760  # 10MB default
export TCP_BANDWIDTH_TIMEOUT=30

# Network Testing
export PING_COUNT=4
export PING_TIMEOUT=30
export TRACEROUTE_TIMEOUT=60
```

### Database Setup

The application supports both SQLite (default) and PostgreSQL:

**SQLite** (Development):
```bash
export DATABASE_URL=sqlite:///streamswarm.db
```

**PostgreSQL** (Production):
```bash
export DATABASE_URL=postgresql://username:password@hostname:port/database
```

### Production Deployment

For production use, consider:

1. **Use PostgreSQL** for better performance and reliability
2. **Set up SSL/TLS** for secure communications
3. **Configure firewall** to restrict access to necessary ports
4. **Monitor disk space** for database growth
5. **Set up log rotation** for application logs
6. **Use a reverse proxy** (nginx/Apache) for the web interface

## API Reference

StreamSwarm provides a REST API for programmatic access:

### Client Registration
```
POST /api/client/register
Content-Type: application/json

{
  "hostname": "client-01",
  "ip_address": "192.168.1.50",
  "system_info": {...}
}
```

### Submit Test Results
```
POST /api/test/results
Content-Type: application/json

{
  "client_id": 1,
  "test_id": 1,
  "timestamp": "2025-01-01T12:00:00",
  "cpu_percent": 25.5,
  "memory_percent": 60.2,
  "ping_latency": 15.3,
  "ping_packet_loss": 0.0
}
```

### Get Test Data
```
GET /api/test/{test_id}/data
```

### Dashboard Statistics
```
GET /api/dashboard/stats
```

## Security Considerations

1. **Network Security**: Clients communicate with server over HTTP by default
2. **Data Privacy**: All test data is stored in the database
3. **Access Control**: Web interface has no built-in authentication
4. **System Access**: Clients need permission to run network utilities

For production use, consider implementing:
- HTTPS/TLS encryption
- User authentication and authorization
- Network ACLs and firewall rules
- Regular security updates

## Support and Contributing

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review application logs for error messages
3. Verify system requirements are met
4. Test with minimal configuration first

The system is designed to be simple, reliable, and scalable for distributed network monitoring across multiple hosts and networks.