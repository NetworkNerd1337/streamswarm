#!/usr/bin/env python3
"""
StreamSwarm Wireless Detection Debug Tool

This script helps diagnose wireless detection issues on Linux clients.
Run this on your client machines to identify what's preventing wireless details collection.
"""

import subprocess
import os
import platform
import sys

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False, "", "Timeout"
    except FileNotFoundError:
        print("❌ Command not found")
        return False, "", "Command not found"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, "", str(e)

def test_python_imports():
    """Test Python library imports"""
    print(f"\n{'='*60}")
    print("Testing Python Library Imports")
    print('='*60)
    
    # Test iwlib
    try:
        import iwlib
        print("✓ iwlib imported successfully")
        
        # Test iwlib functions
        try:
            interfaces = iwlib.get_iwlist()
            print(f"✓ iwlib.get_iwlist() works, found {len(interfaces)} interfaces")
            for iface in interfaces:
                print(f"  - {iface}")
        except Exception as e:
            print(f"❌ iwlib.get_iwlist() failed: {e}")
            
    except ImportError:
        print("❌ iwlib not available - install with: pip install iwlib")
    
    # Test psutil
    try:
        import psutil
        print("✓ psutil imported successfully")
        
        net_if_stats = psutil.net_if_stats()
        wireless_interfaces = []
        for name, stats in net_if_stats.items():
            if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath']):
                wireless_interfaces.append(name)
        
        print(f"✓ psutil detected {len(wireless_interfaces)} wireless interfaces: {wireless_interfaces}")
        
    except ImportError:
        print("❌ psutil not available")

def test_wireless_interface_detection():
    """Test wireless interface detection methods"""
    print(f"\n{'='*60}")
    print("Testing Wireless Interface Detection")
    print('='*60)
    
    # Find wireless interfaces
    wireless_interfaces = []
    
    # Method 1: Check /sys/class/net
    try:
        net_path = "/sys/class/net"
        if os.path.exists(net_path):
            for iface in os.listdir(net_path):
                wireless_path = f"{net_path}/{iface}/wireless"
                if os.path.exists(wireless_path):
                    wireless_interfaces.append(iface)
                    print(f"✓ Found wireless interface via /sys/class/net: {iface}")
    except Exception as e:
        print(f"❌ /sys/class/net check failed: {e}")
    
    # Method 2: Check interface names
    try:
        import psutil
        net_if_stats = psutil.net_if_stats()
        for name in net_if_stats:
            if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
                if name not in wireless_interfaces:
                    wireless_interfaces.append(name)
                print(f"✓ Found wireless interface via naming: {name}")
    except Exception as e:
        print(f"❌ psutil interface check failed: {e}")
    
    if not wireless_interfaces:
        print("❌ No wireless interfaces detected")
        return []
    
    print(f"\n📶 Total wireless interfaces found: {wireless_interfaces}")
    return wireless_interfaces

def test_wireless_tools(interface):
    """Test wireless tools on a specific interface"""
    print(f"\n{'='*60}")
    print(f"Testing Wireless Tools on Interface: {interface}")
    print('='*60)
    
    # Test iwconfig
    success, stdout, stderr = run_command(['iwconfig', interface], f"iwconfig {interface}")
    if success:
        if 'ESSID:' in stdout:
            print("✓ iwconfig shows ESSID information")
        else:
            print("⚠ iwconfig output doesn't contain ESSID")
    
    # Test iw
    success, stdout, stderr = run_command(['iw', 'dev', interface, 'link'], f"iw dev {interface} link")
    if success:
        if 'SSID:' in stdout:
            print("✓ iw shows SSID information")
        else:
            print("⚠ iw output doesn't contain SSID")
    
    # Test nmcli
    success, stdout, stderr = run_command(['nmcli', '-t', '-f', 'ACTIVE,SSID,SIGNAL,FREQ', 'dev', 'wifi'], "nmcli dev wifi")
    if success:
        active_connections = [line for line in stdout.split('\n') if line.startswith('yes:')]
        if active_connections:
            print(f"✓ nmcli shows {len(active_connections)} active WiFi connections")
        else:
            print("⚠ nmcli shows no active WiFi connections")
    
    # Test iwlib Python library
    try:
        import iwlib
        print(f"\nTesting iwlib on {interface}:")
        
        # Get iwconfig stats
        stats = iwlib.get_iwconfig(interface)
        if stats:
            print(f"✓ iwlib.get_iwconfig() returned data:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("❌ iwlib.get_iwconfig() returned no data")
        
        # Try scanning
        try:
            scan_results = iwlib.scan(interface)
            if scan_results:
                print(f"✓ iwlib.scan() found {len(scan_results)} networks")
            else:
                print("❌ iwlib.scan() returned no results")
        except Exception as e:
            print(f"❌ iwlib.scan() failed: {e}")
            
    except ImportError:
        print("❌ iwlib not available for testing")
    except Exception as e:
        print(f"❌ iwlib testing failed: {e}")

def test_permissions():
    """Test permission requirements"""
    print(f"\n{'='*60}")
    print("Testing Permissions and System Access")
    print('='*60)
    
    # Test /proc/net/wireless access
    try:
        with open('/proc/net/wireless', 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            if len(lines) > 2:  # Header lines plus data
                print(f"✓ /proc/net/wireless accessible, found {len(lines)-2} wireless interfaces")
                print(f"Content preview:\n{content[:200]}...")
            else:
                print("⚠ /proc/net/wireless contains no interface data")
    except FileNotFoundError:
        print("❌ /proc/net/wireless not found")
    except PermissionError:
        print("❌ /proc/net/wireless permission denied")
    except Exception as e:
        print(f"❌ /proc/net/wireless access failed: {e}")
    
    # Test running as root
    if os.geteuid() == 0:
        print("✓ Running as root - full permissions available")
    else:
        print("⚠ Running as regular user - some wireless info may be limited")
        print("  Try running with: sudo python3 wireless_debug.py")

def main():
    print("StreamSwarm Wireless Detection Debug Tool")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Test Python imports
    test_python_imports()
    
    # Test system commands
    run_command(['which', 'iwconfig'], "Check if iwconfig is available")
    run_command(['which', 'iw'], "Check if iw is available") 
    run_command(['which', 'nmcli'], "Check if nmcli is available")
    
    # Test interface detection
    interfaces = test_wireless_interface_detection()
    
    # Test permissions
    test_permissions()
    
    # Test each wireless interface
    for interface in interfaces:
        test_wireless_tools(interface)
    
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print('='*60)
    
    if not interfaces:
        print("❌ No wireless interfaces detected")
        print("   - Ensure WiFi hardware is present and enabled")
        print("   - Check if wireless drivers are loaded: lsmod | grep -i wifi")
    else:
        print("✓ Wireless interfaces detected")
        print("\nIf wireless details still aren't collected:")
        print("1. Ensure all system dependencies are installed:")
        print("   sudo apt install wireless-tools iw network-manager libiw-dev")
        print("2. Ensure Python iwlib is installed:")
        print("   pip install iwlib")
        print("3. Try running the client with verbose logging:")
        print("   python client.py --server YOUR_SERVER --token YOUR_TOKEN --verbose")
        print("4. Check if NetworkManager is running:")
        print("   sudo systemctl status NetworkManager")
        print("5. If using a wireless USB adapter, ensure drivers are properly installed")

if __name__ == "__main__":
    main()