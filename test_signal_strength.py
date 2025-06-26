#!/usr/bin/env python3
"""
Test script to verify signal strength detection is working
"""

import sys
import os
import platform
import psutil
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wireless_detection():
    """Test wireless interface detection"""
    logger.info("Testing wireless interface detection...")
    
    # Get network interfaces
    net_if_stats = psutil.net_if_stats()
    wireless_interfaces = []
    
    # Identify wireless interfaces
    for name in net_if_stats:
        if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
            wireless_interfaces.append(name)
        elif os.path.exists(f"/sys/class/net/{name}/wireless"):
            wireless_interfaces.append(name)
    
    logger.info(f"All network interfaces: {list(net_if_stats.keys())}")
    logger.info(f"Detected wireless interfaces: {wireless_interfaces}")
    
    return wireless_interfaces

def test_signal_strength_methods(interface):
    """Test different signal strength detection methods"""
    logger.info(f"Testing signal strength detection for interface: {interface}")
    
    # Method 1: iwlib Python library
    try:
        import iwlib
        logger.info("iwlib library available")
        stats = iwlib.get_iwconfig(interface)
        logger.info(f"iwlib stats for {interface}: {stats}")
        if stats and 'stats' in stats:
            if 'level' in stats['stats']:
                signal_val = float(stats['stats']['level'])
                logger.info(f"iwlib signal strength: {signal_val} dBm")
                return signal_val
    except Exception as e:
        logger.error(f"iwlib method failed: {e}")
    
    # Method 2: iwconfig command
    try:
        logger.info("Testing iwconfig command...")
        result = subprocess.run(['iwconfig', interface], 
                              capture_output=True, text=True, timeout=3)
        logger.info(f"iwconfig return code: {result.returncode}")
        logger.info(f"iwconfig output: {result.stdout}")
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'Signal level=' in line:
                    logger.info(f"Found signal line: {line.strip()}")
                    signal_part = line.split('Signal level=')[1].split()[0]
                    if signal_part.replace('-', '').replace('.', '').isdigit():
                        signal_val = float(signal_part)
                        logger.info(f"iwconfig signal strength: {signal_val} dBm")
                        return signal_val
    except Exception as e:
        logger.error(f"iwconfig method failed: {e}")
    
    # Method 3: nmcli
    try:
        logger.info("Testing nmcli command...")
        result = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SIGNAL', 'dev', 'wifi'], 
                              capture_output=True, text=True, timeout=3)
        logger.info(f"nmcli return code: {result.returncode}")
        logger.info(f"nmcli output: {result.stdout}")
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('yes:'):
                    parts = line.split(':')
                    if len(parts) >= 2 and parts[1]:
                        try:
                            signal_val = int(parts[1])
                            # Convert percentage to dBm (rough approximation)
                            signal_dbm = -100 + (signal_val * 70 / 100)
                            logger.info(f"nmcli signal strength: {signal_dbm} dBm (from {signal_val}%)")
                            return signal_dbm
                        except ValueError:
                            pass
    except Exception as e:
        logger.error(f"nmcli method failed: {e}")
    
    logger.warning("No signal strength detection method worked")
    return None

def main():
    logger.info("Starting signal strength detection test")
    logger.info(f"Platform: {platform.system()}")
    
    # Test wireless detection
    wireless_interfaces = test_wireless_detection()
    
    if not wireless_interfaces:
        logger.error("No wireless interfaces detected")
        return False
    
    # Test signal strength for each interface
    for interface in wireless_interfaces:
        logger.info(f"\n--- Testing interface: {interface} ---")
        signal_strength = test_signal_strength_methods(interface)
        if signal_strength is not None:
            logger.info(f"SUCCESS: Signal strength detected: {signal_strength} dBm")
            return True
        else:
            logger.warning(f"FAILED: No signal strength detected for {interface}")
    
    logger.error("Signal strength detection failed for all interfaces")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)