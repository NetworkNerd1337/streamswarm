#!/usr/bin/env python3
"""
Bandwidth Diagnostic Tool for StreamSwarm
Tests bandwidth using multiple methods to validate measurements
"""

import time
import requests
import subprocess
import sys

def test_simple_bandwidth():
    """Simple bandwidth test using httpbin.org"""
    print("=== Simple Bandwidth Test ===")
    
    try:
        # Download test - client downloads from internet
        print("Testing download speed...")
        test_size = 2097152  # 2MB
        start_time = time.time()
        response = requests.get(f'https://httpbin.org/bytes/{test_size}', timeout=20, stream=True)
        
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=8192):
            total_bytes += len(chunk)
            if total_bytes >= test_size:
                break
        
        elapsed_time = time.time() - start_time
        download_mbps = (total_bytes * 8) / (elapsed_time * 1000000)
        print(f"Download: {total_bytes} bytes in {elapsed_time:.2f}s = {download_mbps:.2f} Mbps")
        
        # Upload test - client uploads to internet
        print("Testing upload speed...")
        test_data = b'0' * 1048576  # 1MB
        start_time = time.time()
        response = requests.post('https://httpbin.org/post', data=test_data, timeout=20)
        elapsed_time = time.time() - start_time
        upload_mbps = (len(test_data) * 8) / (elapsed_time * 1000000)
        print(f"Upload: {len(test_data)} bytes in {elapsed_time:.2f}s = {upload_mbps:.2f} Mbps")
        
        print(f"Upload/Download Ratio: {upload_mbps/download_mbps:.2f}:1")
        
    except Exception as e:
        print(f"Simple test failed: {e}")

def test_curl_bandwidth():
    """Test using curl for comparison"""
    print("\n=== Curl Bandwidth Test ===")
    
    try:
        # Download test with curl
        print("Testing download with curl...")
        result = subprocess.run([
            'curl', '-w', '%{speed_download}', '-o', '/dev/null', '-s', 
            'https://httpbin.org/bytes/2097152'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            download_speed = float(result.stdout.strip())
            download_mbps = (download_speed * 8) / 1000000
            print(f"Curl download: {download_mbps:.2f} Mbps")
        
        # Upload test with curl
        print("Testing upload with curl...")
        result = subprocess.run([
            'curl', '-w', '%{speed_upload}', '-o', '/dev/null', '-s',
            '-d', '@/dev/zero', '--data-raw', 'x' * 1048576,
            'https://httpbin.org/post'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            upload_speed = float(result.stdout.strip())
            upload_mbps = (upload_speed * 8) / 1000000
            print(f"Curl upload: {upload_mbps:.2f} Mbps")
            
    except Exception as e:
        print(f"Curl test failed: {e}")

def test_speedtest():
    """Test using speedtest-cli"""
    print("\n=== Speedtest-cli Test ===")
    
    try:
        result = subprocess.run(['speedtest-cli', '--simple'], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                print(f"Speedtest: {line}")
        else:
            print("Speedtest-cli not available or failed")
    except Exception as e:
        print(f"Speedtest failed: {e}")

def main():
    print("StreamSwarm Bandwidth Diagnostic Tool")
    print("=====================================")
    
    test_simple_bandwidth()
    test_curl_bandwidth() 
    test_speedtest()
    
    print("\n=== Analysis ===")
    print("If upload speeds are consistently higher than download speeds:")
    print("1. This may indicate asymmetric routing through CDN/proxy infrastructure")
    print("2. The test endpoints may have upload optimization that favors your connection")
    print("3. Your ISP or intermediate networks may have traffic shaping affecting downloads")
    print("4. The measurements are accurate for the current network path and conditions")
    print("\nTo get your true ISP speeds, try running this diagnostic directly on your Pi:")
    print("python3 bandwidth_diagnostic.py")

if __name__ == "__main__":
    main()