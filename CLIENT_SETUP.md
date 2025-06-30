# StreamSwarm Client Setup Guide

## Basic Client Setup

The basic client functionality works with just:
1. Copy `client.py` to your client machine
2. Run: `python client.py --server-url http://your-server:5000 --api-token your-token`

## Required Python Packages

### Core Dependencies
```bash
pip install requests psutil
```

### Advanced Network Testing (Optional)
```bash
pip install speedtest-cli scapy
```

### System Requirements
- `ping` and `traceroute` commands (usually pre-installed on Linux/macOS)
- Root/admin privileges may be needed for some advanced network tests
- Network tools: `mtr` (optional for enhanced traceroute analysis)

## Geolocation Path Analysis

Interactive network path visualization is now handled entirely by the server. Clients only need to:
1. Have `traceroute` command available
2. Send traceroute data to the server

**No additional client packages required** - all map generation and geolocation processing happens server-side.

## Firewall and Network Requirements

### Required External Access
The client needs outbound access to:
- **StreamSwarm Server**: HTTP/HTTPS (ports 80/443 or custom port)
- **Speed Test Servers**: Various HTTP/HTTPS endpoints for bandwidth testing
- **DNS Servers**: UDP port 53 for hostname resolution

### Features Requiring Internet Access
- **Bandwidth Testing**: Requires access to speed test endpoints
- **Network Path Analysis**: Traceroute data collection (local network tools)
- **Server Communication**: API calls to StreamSwarm server

### Offline Capabilities
The following features work without internet access:
- System monitoring (CPU, memory, disk)
- Local network interface statistics
- Basic ping tests to local/configured destinations

## Troubleshooting

### Common Issues
1. **Permission Errors**: Some network tests require root/admin privileges
2. **Missing Commands**: Install `traceroute` if not available: `apt install traceroute` (Ubuntu/Debian)
3. **Firewall Blocking**: Ensure outbound access to server and speed test endpoints
4. **API Token Issues**: Verify token with server administrator

### Verification Commands
```bash
# Test basic connectivity
python -c "import requests, psutil; print('Core packages OK')"

# Test network tools
ping -c 1 8.8.8.8
traceroute google.com

# Test advanced packages (if installed)
python -c "import speedtest; print('speedtest-cli OK')"
```