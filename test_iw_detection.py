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
            else:
                print("  Interface not connected")
        else:
            print(f"✗ iw dev {interface} link failed")
    except Exception as e:
        print(f"✗ Error getting link info: {e}")

if __name__ == "__main__":
    print("StreamSwarm iw Command Test")
    print("=" * 40)
    
    interfaces = test_iw_commands()
    
    if interfaces:
        for interface in interfaces:
            test_interface_info(interface)
    else:
        print("\nNo wireless interfaces found to test")
    
    print("\nTest completed")