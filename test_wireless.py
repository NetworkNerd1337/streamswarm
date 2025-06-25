#!/usr/bin/env python3
"""
Test script to verify wireless detection functionality
"""
import subprocess
import platform
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wireless_tools():
    """Test if wireless detection tools are available"""
    print("Testing wireless detection tools availability...")
    
    tools = ['iwconfig', 'nmcli', 'iw']
    available_tools = []
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, timeout=2)
            if result.returncode == 0:
                available_tools.append(tool)
                print(f"✓ {tool} is available")
            else:
                print(f"✗ {tool} returned error code {result.returncode}")
        except FileNotFoundError:
            print(f"✗ {tool} is not installed")
        except subprocess.TimeoutExpired:
            print(f"✗ {tool} timed out")
        except Exception as e:
            print(f"✗ {tool} error: {e}")
    
    print(f"\nAvailable tools: {available_tools}")
    return available_tools

def test_network_interfaces():
    """Test network interface detection"""
    print("\nTesting network interface detection...")
    
    try:
        import psutil
        interfaces = psutil.net_if_addrs()
        for interface_name, addresses in interfaces.items():
            print(f"\nInterface: {interface_name}")
            
            # Check if interface is wireless
            wireless_path = f"/sys/class/net/{interface_name}/wireless"
            is_wireless = os.path.exists(wireless_path)
            print(f"  Wireless: {is_wireless}")
            
            if is_wireless:
                print(f"  Wireless path exists: {wireless_path}")
                
                # Test iwconfig
                try:
                    result = subprocess.run(['iwconfig', interface_name], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"  iwconfig output preview: {result.stdout[:200]}...")
                    else:
                        print(f"  iwconfig failed with code {result.returncode}")
                except Exception as e:
                    print(f"  iwconfig error: {e}")
                
                # Test nmcli
                try:
                    result = subprocess.run(['nmcli', '-t', '-f', 'DEVICE,STATE,CONNECTION', 'dev'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"  nmcli dev output preview: {result.stdout[:200]}...")
                    else:
                        print(f"  nmcli failed with code {result.returncode}")
                except Exception as e:
                    print(f"  nmcli error: {e}")
            
            # Show addresses
            for addr in addresses:
                print(f"  {addr.family.name}: {addr.address}")
                
    except ImportError:
        print("psutil not available")
    except Exception as e:
        print(f"Error getting interfaces: {e}")

def test_wireless_detection_logic():
    """Test the actual wireless detection logic from client.py"""
    print("\nTesting wireless detection logic...")
    
    try:
        import psutil
        interfaces = psutil.net_if_addrs()
        
        for interface_name, addresses in interfaces.items():
            wireless_path = f"/sys/class/net/{interface_name}/wireless"
            if os.path.exists(wireless_path):
                print(f"\nTesting wireless interface: {interface_name}")
                
                wireless_info = {
                    'ssid': None,
                    'signal_strength': None,
                    'frequency': None,
                    'security': None
                }
                
                # Method 1: iwconfig
                try:
                    result = subprocess.run(['iwconfig', interface_name], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        print(f"  iwconfig output: {output}")
                        
                        # Parse SSID
                        if 'ESSID:' in output:
                            essid_line = [line for line in output.split('\n') if 'ESSID:' in line][0]
                            if 'ESSID:"' in essid_line:
                                ssid = essid_line.split('ESSID:"')[1].split('"')[0]
                                if ssid and ssid != 'off/any':
                                    wireless_info['ssid'] = ssid
                                    print(f"  Found SSID: {ssid}")
                                else:
                                    print(f"  SSID is off/any or empty")
                        
                        # Parse signal strength
                        if 'Signal level=' in output:
                            signal_line = [line for line in output.split('\n') if 'Signal level=' in line][0]
                            if 'Signal level=' in signal_line:
                                signal_part = signal_line.split('Signal level=')[1].split()[0]
                                wireless_info['signal_strength'] = signal_part
                                print(f"  Found signal: {signal_part}")
                                
                except Exception as e:
                    print(f"  iwconfig error: {e}")
                
                # Check if no SSID found - test tool availability logic
                if not wireless_info['ssid']:
                    print("  No SSID found, testing tool availability...")
                    tools_available = []
                    for tool in ['iwconfig', 'nmcli', 'iw']:
                        try:
                            subprocess.run([tool, '--version'], capture_output=True, timeout=2)
                            tools_available.append(tool)
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            pass
                    
                    if tools_available:
                        status_msg = f"Disconnected (tools available: {', '.join(tools_available)})"
                        wireless_info['ssid'] = status_msg
                        print(f"  Status: {status_msg}")
                    else:
                        wireless_info['ssid'] = "Wireless tools not available"
                        print(f"  Status: Wireless tools not available")
                
                print(f"  Final wireless_info: {wireless_info}")
                
    except Exception as e:
        print(f"Error in wireless detection test: {e}")

if __name__ == "__main__":
    print("StreamSwarm Wireless Detection Test")
    print("=" * 40)
    
    print(f"Platform: {platform.system()} {platform.release()}")
    
    available_tools = test_wireless_tools()
    test_network_interfaces()
    test_wireless_detection_logic()
    
    print("\n" + "=" * 40)
    print("Test completed!")