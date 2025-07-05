# StreamSwarm Client Deployment Guide

This guide covers deploying StreamSwarm monitoring clients on Linux systems with automated startup using cron.

## Quick Setup

### 1. Client Files Required
Download these files to your client machine:
- `client.py` - Main client application
- `config.py` - Configuration module
- `geolocation_service.py` - Geolocation analysis (optional)
- `gnmi_client.py` - GNMI network analysis (optional)
- `scripts/start_streamswarm_client.sh` - Startup script

### 2. Install Dependencies
```bash
# Install system packages
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv screen git

# Create virtual environment
python3 -m venv ~/streamswarm-venv

# Activate virtual environment
source ~/streamswarm-venv/bin/activate

# Install Python packages
pip install requests psutil speedtest-cli scapy joblib scikit-learn matplotlib pandas numpy
```

### 3. Configure Startup Script
Edit `start_streamswarm_client.sh` and update these variables:
```bash
# Your StreamSwarm server URL
SERVER_URL="https://your-server.replit.app"

# API token from server admin panel
API_TOKEN="your-api-token-here"

# Client name (optional, defaults to hostname)
CLIENT_NAME="$(hostname)-client"

# Virtual environment path
VENV_PATH="/home/$(whoami)/streamswarm-venv"

# Application directory
APP_DIR="/home/$(whoami)/streamswarm-client"
```

### 4. Make Script Executable
```bash
chmod +x start_streamswarm_client.sh
```

### 5. Add to Cron for Auto-Start
```bash
# Edit cron
crontab -e

# Add this line to start client on boot
@reboot /path/to/start_streamswarm_client.sh
```

## Advanced Configuration

### Environment Variables
Instead of editing the script, you can set environment variables:
```bash
export STREAMSWARM_SERVER_URL="https://your-server.replit.app"
export STREAMSWARM_API_TOKEN="your-api-token"
export STREAMSWARM_CLIENT_NAME="custom-client-name"
```

### Manual Client Start
To start the client manually:
```bash
# Activate virtual environment
source ~/streamswarm-venv/bin/activate

# Start client
python client.py --server-url "https://your-server.replit.app" --api-token "your-token" --client-name "client-name"
```

### Screen Session Management
The startup script uses screen sessions for background execution:
```bash
# View running client
screen -r streamswarm-client

# Detach from session (Ctrl+A then D)
# Kill session
screen -S streamswarm-client -X quit
```

## Monitoring & Troubleshooting

### Check Client Status
```bash
# Use the generated status script
./client_status.sh

# Or check manually
screen -list
tail -f /var/log/streamswarm-client.log
```

### Common Issues

**Network not available at boot:**
- The script waits for network connectivity before starting
- Increase wait time if needed in the script

**Virtual environment not found:**
- Verify the VENV_PATH is correct
- Create virtual environment: `python3 -m venv ~/streamswarm-venv`

**Git pull fails:**
- Check if git repository is properly cloned
- Verify network access to git repository
- Script continues with existing code if git pull fails

**Client connection fails:**
- Verify SERVER_URL is correct and accessible
- Check API_TOKEN is valid (get from server admin panel)
- Ensure firewall allows outbound HTTPS connections

### Log Files
- Main log: `/var/log/streamswarm-client.log`
- Client output: visible in screen session
- System logs: `journalctl -u cron` for cron issues

## Security Considerations

### API Token Security
- Keep API tokens secure and rotate regularly
- Use environment variables instead of hardcoding in scripts
- Set appropriate file permissions: `chmod 600 start_streamswarm_client.sh`

### Network Security
- Client only makes outbound HTTPS connections
- No inbound ports required
- Consider firewall rules for additional security

### File Permissions
```bash
# Secure the startup script
chmod 700 start_streamswarm_client.sh

# Secure log files
sudo chmod 640 /var/log/streamswarm-client.log
```

## Multi-Client Deployment

### Using Configuration Management
For deploying multiple clients, consider:
- Ansible playbooks
- Puppet/Chef configurations
- Docker containers (if containerization is acceptable)

### Centralized Configuration
Create a configuration template and deploy across clients:
```bash
# Template approach
cp start_streamswarm_client.template.sh start_streamswarm_client.sh
sed -i 's/SERVER_URL_PLACEHOLDER/https:\/\/your-server.replit.app/g' start_streamswarm_client.sh
sed -i 's/API_TOKEN_PLACEHOLDER/your-token/g' start_streamswarm_client.sh
```

## Performance Optimization

### System Resources
- Client uses minimal CPU/memory
- Network bandwidth depends on test frequency
- Consider system load when scheduling tests

### Test Optimization
- Balance test frequency vs. system load
- Use appropriate packet sizes for your network
- Monitor client system metrics in server dashboard

## Backup & Recovery

### Client Configuration Backup
```bash
# Backup client configuration
tar -czf streamswarm-client-backup.tar.gz \
  start_streamswarm_client.sh \
  client.py \
  config.py \
  geolocation_service.py \
  gnmi_client.py
```

### Quick Recovery
```bash
# Restore from backup
tar -xzf streamswarm-client-backup.tar.gz
chmod +x start_streamswarm_client.sh
# Edit configuration and restart
```

## Support

For issues with client deployment:
1. Check the troubleshooting section above
2. Review client logs: `/var/log/streamswarm-client.log`
3. Verify server connectivity and API token
4. Check system requirements and dependencies