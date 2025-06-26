#!/usr/bin/env python3
"""
Quick test to verify signal strength detection and create sample data
"""

import subprocess
import os
import psutil
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_signal_detection():
    """Test signal strength detection on current system"""
    
    # Get wireless interfaces
    net_if_stats = psutil.net_if_stats()
    wireless_interfaces = []
    
    for name in net_if_stats:
        if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
            wireless_interfaces.append(name)
        elif os.path.exists(f"/sys/class/net/{name}/wireless"):
            wireless_interfaces.append(name)
    
    print(f"Wireless interfaces found: {wireless_interfaces}")
    
    if not wireless_interfaces:
        print("No wireless interfaces detected")
        return None
    
    # Test each wireless interface
    for interface in wireless_interfaces:
        print(f"\n=== Testing {interface} ===")
        
        # Test iwconfig
        try:
            result = subprocess.run(['iwconfig', interface], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print(f"iwconfig output:\n{result.stdout}")
                
                # Parse signal
                for line in result.stdout.split('\n'):
                    if 'Signal level=' in line:
                        print(f"Signal line found: {line.strip()}")
                        try:
                            signal_part = line.split('Signal level=')[1].split()[0]
                            signal_part = signal_part.replace('dBm', '').replace('dbm', '')
                            if '/' in signal_part:
                                signal_part = signal_part.split('/')[0]
                            
                            signal_clean = signal_part.strip()
                            if signal_clean.replace('-', '').replace('.', '').isdigit():
                                signal_val = float(signal_clean)
                                print(f"SUCCESS: Parsed signal strength: {signal_val} dBm")
                                return signal_val
                        except Exception as e:
                            print(f"Failed to parse: {e}")
            else:
                print(f"iwconfig failed with code {result.returncode}")
                print(f"stderr: {result.stderr}")
        except Exception as e:
            print(f"iwconfig error: {e}")
        
        # Test /proc/net/wireless
        try:
            with open('/proc/net/wireless', 'r') as f:
                content = f.read()
                print(f"/proc/net/wireless content:\n{content}")
                
                for line in content.split('\n'):
                    if interface in line and not line.strip().startswith('Inter-'):
                        parts = line.split()
                        print(f"Line parts: {parts}")
                        if len(parts) >= 4:
                            signal_str = parts[3].rstrip('.')
                            print(f"Raw signal: '{signal_str}'")
                            if signal_str.replace('-', '').replace('.', '').isdigit():
                                signal_val = float(signal_str)
                                print(f"SUCCESS: /proc signal: {signal_val} dBm")
                                return signal_val
        except Exception as e:
            print(f"/proc/net/wireless error: {e}")
    
    print("No signal strength detected from any method")
    return None

if __name__ == "__main__":
    signal = test_signal_detection()
    if signal:
        print(f"\nDetected signal strength: {signal} dBm")
    else:
        print("\nNo signal strength detected")