#!/bin/bash

# StreamSwarm Client Debug Log Viewer
# This script helps view client logs for reboot and heartbeat debugging

echo "=== StreamSwarm Client Debug Log Viewer ==="
echo "Looking for HEARTBEAT and REBOOT messages in client logs..."
echo

# Check if log file exists
if [ ! -f "/var/log/streamswarm-client.log" ]; then
    echo "Log file /var/log/streamswarm-client.log not found."
    echo "Trying to find client logs in other locations..."
    
    # Look for client process and its logs
    CLIENT_PID=$(pgrep -f "python.*client.py" | head -1)
    if [ -n "$CLIENT_PID" ]; then
        echo "Found client process with PID: $CLIENT_PID"
        echo "Client command line:"
        ps -p $CLIENT_PID -o cmd --no-headers
        echo
    fi
    
    # Look for recent client logs in common locations
    echo "Checking for client logs in common locations..."
    find /var/log /tmp /home -name "*streamswarm*" -type f 2>/dev/null | grep -i log
    find /var/log /tmp /home -name "*client*" -type f 2>/dev/null | grep -i log
    echo
    
    # Check screen sessions
    echo "Checking for screen sessions:"
    screen -ls 2>/dev/null || echo "No screen sessions found"
    echo
    
    exit 1
fi

echo "Found client log file: /var/log/streamswarm-client.log"
echo "File size: $(ls -lh /var/log/streamswarm-client.log | awk '{print $5}')"
echo "Last modified: $(ls -l /var/log/streamswarm-client.log | awk '{print $6, $7, $8}')"
echo

echo "=== Recent HEARTBEAT Messages ==="
tail -n 1000 /var/log/streamswarm-client.log | grep "HEARTBEAT:" | tail -10
echo

echo "=== Recent REBOOT Messages ==="
tail -n 1000 /var/log/streamswarm-client.log | grep "REBOOT:" | tail -10
echo

echo "=== Last 20 Log Lines ==="
tail -n 20 /var/log/streamswarm-client.log
echo

echo "=== Client Process Status ==="
CLIENT_PID=$(pgrep -f "python.*client.py" | head -1)
if [ -n "$CLIENT_PID" ]; then
    echo "Client is running with PID: $CLIENT_PID"
    echo "Process info:"
    ps -p $CLIENT_PID -o pid,ppid,cmd,etime,stat --no-headers
    echo
    echo "Process tree:"
    pstree -p $CLIENT_PID 2>/dev/null || echo "pstree not available"
else
    echo "Client process not found"
fi
echo

echo "=== Real-time Log Monitoring ==="
echo "To monitor logs in real-time, run:"
echo "  tail -f /var/log/streamswarm-client.log | grep -E 'HEARTBEAT|REBOOT'"
echo
echo "To test reboot functionality, check the web interface and then run:"
echo "  tail -f /var/log/streamswarm-client.log | grep REBOOT"