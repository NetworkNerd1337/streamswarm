#!/bin/bash

# Define the process command string to check for
PROCESS_NAME="^python client.py"

# Check if the process is running
if pgrep -f "$PROCESS_NAME" > /dev/null
then
    echo "$(date): Process '$PROCESS_NAME' is running."
else
    echo "$(date): Process '$PROCESS_NAME' is NOT running. Initiating reboot."
    # Trigger a system reboot (requires root privileges)
    /sbin/reboot
fi