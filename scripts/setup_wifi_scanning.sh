#!/bin/bash
# StreamSwarm WiFi Scanning Setup Script
# This script configures passwordless WiFi scanning for the StreamSwarm client

echo "StreamSwarm WiFi Scanning Setup"
echo "================================"

# Check if user is root
if [[ $EUID -eq 0 ]]; then
    echo "Error: Please run this script as a regular user, not as root"
    echo "Usage: bash setup_wifi_scanning.sh"
    exit 1
fi

# Check if iw command is available
if ! command -v iw &> /dev/null; then
    echo "Installing iw command..."
    sudo apt-get update
    sudo apt-get install -y iw
fi

# Get current username
USERNAME=$(whoami)
echo "Configuring WiFi scanning permissions for user: $USERNAME"

# Option 1: Add user to netdev group (preferred method)
echo "Adding user to netdev group..."
sudo usermod -a -G netdev $USERNAME

# Option 2: Configure passwordless sudo for iw command (fallback)
echo "Configuring passwordless sudo for WiFi scanning..."
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/sbin/iw" | sudo tee /etc/sudoers.d/streamswarm-wifi > /dev/null

# Set proper permissions on sudoers file
sudo chmod 440 /etc/sudoers.d/streamswarm-wifi

echo ""
echo "WiFi scanning setup complete!"
echo ""
echo "IMPORTANT: You must log out and log back in for group changes to take effect."
echo ""
echo "To verify setup:"
echo "1. Log out and log back in"
echo "2. Run: groups | grep netdev"
echo "3. Test WiFi scanning: iw dev wlan0 scan"
echo ""
echo "Your StreamSwarm client can now perform WiFi environmental scans without sudo!"