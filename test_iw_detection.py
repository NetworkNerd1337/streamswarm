#!/usr/bin/env python3
"""
Test script to verify iw command wireless detection works properly
"""
import subprocess
import sys
import json

def test_iw_commands():
    """Test various iw commands to ensure wireless detection works"""
    print("Testing iw command availability and functionality...")
    
    # Test 1: Check if iw command is available
    try:
        result = subprocess.run(['iw', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ iw command is available")
            print(f"  Version: {result.stdout.strip()}")
        else:
            print("✗ iw command failed")
            return False
    except FileNotFoundError:
        print("✗ iw command not found - install with: sudo apt install iw")
        return False
    except Exception as e:
        print(f"✗ Error testing iw command: {e}")
        return False
    
    # Test 2: List wireless devices
    try:
        result = subprocess.run(['iw', 'dev'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ iw dev command works")
            output = result.stdout
            if 'Interface' in output:
                interfaces = []
                for line in output.split('\n'):
                    if 'Interface' in line:
                        interface = line.split('Interface')[1].strip()
                        interfaces.append(interface)
                print(f"  Found wireless interfaces: {interfaces}")
                return interfaces
            else:
                print("  No wireless interfaces found")
                return []
        else:
            print("✗ iw dev command failed")
            return []
    except Exception as e:
        print(f"✗ Error listing wireless devices: {e}")
        return []

def test_interface_info(interface):
    """Test getting info for a specific interface"""
    print(f"\nTesting interface info for {interface}...")
    
    # Test interface info
    try:
        result = subprocess.run(['iw', 'dev', interface, 'info'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ iw dev {interface} info works")
            output = result.stdout
            print("  Interface info:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"    {line.strip()}")
            
            # Test our parsing logic
            test_wireless_parsing(output, "info")
        else:
            print(f"✗ iw dev {interface} info failed")
    except Exception as e:
        print(f"✗ Error getting interface info: {e}")
    
    # Test link info
    try:
        result = subprocess.run(['iw', 'dev', interface, 'link'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ iw dev {interface} link works")
            output = result.stdout
            if 'Connected to' in output:
                print("  Connection info:")
                for line in output.split('\n'):
                    if line.strip():
                        print(f"    {line.strip()}")
                
                # Test our parsing logic
                test_wireless_parsing(output, "link")
            else:
                print("  Interface not connected")
        else:
            print(f"✗ iw dev {interface} link failed")
    except Exception as e:
        print(f"✗ Error getting link info: {e}")

def test_wireless_parsing(output, command_type):
    """Test parsing logic similar to the client code"""
    print(f"\n  Testing {command_type} parsing:")
    wireless_info = {}
    
    if command_type == "info":
        # Parse detailed interface information
        for line in output.split('\n'):
            line = line.strip()
            
            # Extract MAC address
            if line.startswith('addr '):
                mac_addr = line.split('addr ')[1].strip()
                wireless_info['mac_address'] = mac_addr
                print(f"    ✓ MAC Address: {mac_addr}")
            
            # Extract channel and frequency info
            elif 'channel' in line.lower() and 'MHz' in line:
                if '(' in line and 'MHz' in line:
                    try:
                        freq_part = line.split('(')[1].split(' MHz')[0]
                        wireless_info['frequency'] = f"{freq_part} MHz"
                        print(f"    ✓ Frequency: {freq_part} MHz")
                    except:
                        pass
                
                if 'channel' in line.lower():
                    try:
                        channel_part = line.lower().split('channel')[1].strip().split()[0]
                        wireless_info['channel'] = channel_part
                        print(f"    ✓ Channel: {channel_part}")
                    except:
                        pass
            
            # Extract transmission power
            elif 'txpower' in line.lower():
                try:
                    txpower_part = line.split('txpower')[1].strip()
                    wireless_info['txpower'] = txpower_part
                    print(f"    ✓ TX Power: {txpower_part}")
                except:
                    pass
    
    elif command_type == "link":
        # Parse connection info
        for line in output.split('\n'):
            if 'SSID:' in line:
                ssid = line.split('SSID:')[1].strip()
                if ssid:
                    wireless_info['ssid'] = ssid
                    print(f"    ✓ SSID: {ssid}")
            
            elif 'signal:' in line:
                try:
                    signal_part = line.split('signal:')[1].strip().split()[0]
                    signal_part = signal_part.replace('dBm', '').replace('dbm', '')
                    wireless_info['signal_strength'] = signal_part
                    print(f"    ✓ Signal: {signal_part} dBm")
                except:
                    pass
    
    if not wireless_info:
        print(f"    ⚠ No data extracted from {command_type} output")

def test_simulated_output():
    """Test parsing with simulated iw command output"""
    print("\nTesting with simulated iw output...")
    
    # Simulate typical iw dev wlan0 info output
    simulated_info = """Interface wlan0
        ifindex 3
        wdev 0x1
        addr aa:bb:cc:dd:ee:ff
        type managed
        wiphy 0
        channel 6 (2437 MHz), width: 20 MHz, center1: 2437 MHz
        txpower 20.00 dBm"""
    
    # Simulate typical iw dev wlan0 link output  
    simulated_link = """Connected to 11:22:33:44:55:66 (on wlan0)
        SSID: RTHNET
        freq: 2437
        signal: -22 dBm
        tx bitrate: 144.4 MBit/s MCS 15 short GI"""
    
    print("  Simulated info output:")
    test_wireless_parsing(simulated_info, "info")
    
    print("  Simulated link output:")
    test_wireless_parsing(simulated_link, "link")

if __name__ == "__main__":
    print("StreamSwarm iw Command Test")
    print("=" * 40)
    
    interfaces = test_iw_commands()
    
    if interfaces:
        for interface in interfaces:
            test_interface_info(interface)
    else:
        print("\nNo wireless interfaces found to test")
    
    # Always test simulated output to verify parsing logic
    test_simulated_output()
    
    print("\nTest completed")