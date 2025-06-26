#!/usr/bin/env python3
"""
Debug wireless detection to identify why signal strength isn't being captured
"""

import subprocess
import os
import psutil
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_all_methods():
    """Test all wireless detection methods comprehensively"""
    
    # 1. Check available network interfaces
    logger.info("=== Network Interface Detection ===")
    net_if_stats = psutil.net_if_stats()
    all_interfaces = list(net_if_stats.keys())
    logger.info(f"All interfaces: {all_interfaces}")
    
    # 2. Identify potential wireless interfaces
    wireless_candidates = []
    for name in all_interfaces:
        if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
            wireless_candidates.append(name)
        elif os.path.exists(f"/sys/class/net/{name}/wireless"):
            wireless_candidates.append(name)
    
    logger.info(f"Wireless candidates: {wireless_candidates}")
    
    # 3. Test iwconfig for each interface
    logger.info("\n=== Testing iwconfig ===")
    for interface in wireless_candidates:
        try:
            result = subprocess.run(['iwconfig', interface], 
                                  capture_output=True, text=True, timeout=5)
            logger.info(f"iwconfig {interface} - return code: {result.returncode}")
            if result.returncode == 0:
                logger.info(f"iwconfig {interface} output:\n{result.stdout}")
            else:
                logger.info(f"iwconfig {interface} stderr:\n{result.stderr}")
        except Exception as e:
            logger.error(f"iwconfig {interface} failed: {e}")
    
    # 4. Test nmcli
    logger.info("\n=== Testing nmcli ===")
    try:
        result = subprocess.run(['nmcli', 'dev', 'status'], 
                              capture_output=True, text=True, timeout=5)
        logger.info(f"nmcli dev status - return code: {result.returncode}")
        if result.returncode == 0:
            logger.info(f"nmcli dev status output:\n{result.stdout}")
        
        result = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SIGNAL', 'dev', 'wifi'], 
                              capture_output=True, text=True, timeout=5)
        logger.info(f"nmcli wifi - return code: {result.returncode}")
        if result.returncode == 0:
            logger.info(f"nmcli wifi output:\n{result.stdout}")
        else:
            logger.info(f"nmcli wifi stderr:\n{result.stderr}")
    except Exception as e:
        logger.error(f"nmcli failed: {e}")
    
    # 5. Test /proc/net/wireless
    logger.info("\n=== Testing /proc/net/wireless ===")
    try:
        if os.path.exists('/proc/net/wireless'):
            with open('/proc/net/wireless', 'r') as f:
                content = f.read()
                logger.info(f"/proc/net/wireless content:\n{content}")
        else:
            logger.info("/proc/net/wireless does not exist")
    except Exception as e:
        logger.error(f"/proc/net/wireless failed: {e}")
    
    # 6. Test iwlib if available
    logger.info("\n=== Testing iwlib ===")
    try:
        import iwlib
        logger.info("iwlib module available")
        for interface in wireless_candidates:
            try:
                stats = iwlib.get_iwconfig(interface)
                logger.info(f"iwlib {interface} stats: {stats}")
            except Exception as e:
                logger.error(f"iwlib {interface} failed: {e}")
    except ImportError:
        logger.info("iwlib module not available")
    except Exception as e:
        logger.error(f"iwlib failed: {e}")

if __name__ == "__main__":
    test_all_methods()