#!/usr/bin/env python3
"""
Simple test script for TCP handshake timing analysis
"""

import socket
import time
import select
import errno


def test_tcp_handshake_timing(hostname, port=80):
    """Test TCP handshake timing analysis"""
    metrics = {
        'tcp_handshake_syn_time': None,
        'tcp_handshake_synack_time': None,
        'tcp_handshake_ack_time': None,
        'tcp_handshake_total_time': None,
        'tcp_handshake_network_delay': None,
        'tcp_handshake_server_processing': None,
        'tcp_handshake_analysis': None
    }
    
    try:
        # Method 1: Non-blocking socket for detailed timing measurement
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        
        # Record timing points
        t0_start = time.time()
        
        # Initiate SYN packet
        try:
            result = sock.connect((hostname, port))
        except socket.error as e:
            if e.errno not in [errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY]:
                raise
        
        t1_syn_sent = time.time()
        metrics['tcp_handshake_syn_time'] = (t1_syn_sent - t0_start) * 1000  # ms
        
        # Wait for connection to complete (SYN-ACK received)
        ready = select.select([], [sock], [], 10.0)  # 10 second timeout
        
        if ready[1]:  # Socket is ready for writing (connection established)
            t2_synack_received = time.time()
            
            # Check if connection is actually established
            error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if error == 0:
                # Connection established successfully
                t3_ack_sent = time.time()  # ACK is sent automatically
                
                # Calculate timing metrics
                syn_time = (t1_syn_sent - t0_start) * 1000
                synack_time = (t2_synack_received - t1_syn_sent) * 1000
                ack_time = (t3_ack_sent - t2_synack_received) * 1000
                total_time = (t3_ack_sent - t0_start) * 1000
                
                metrics['tcp_handshake_syn_time'] = round(syn_time, 3)
                metrics['tcp_handshake_synack_time'] = round(synack_time, 3)
                metrics['tcp_handshake_ack_time'] = round(ack_time, 3)
                metrics['tcp_handshake_total_time'] = round(total_time, 3)
                
                # Estimate network delay and server processing
                estimated_one_way_delay = synack_time / 2  # Rough estimate
                estimated_server_processing = max(0, synack_time - estimated_one_way_delay)
                
                metrics['tcp_handshake_network_delay'] = round(estimated_one_way_delay, 3)
                metrics['tcp_handshake_server_processing'] = round(estimated_server_processing, 3)
                
                # Basic analysis
                if total_time < 50:
                    analysis = "Excellent handshake performance"
                elif total_time < 100:
                    analysis = "Good handshake performance"
                elif total_time < 200:
                    analysis = "Moderate handshake performance"
                else:
                    analysis = "Slow handshake performance - investigate network or server"
                
                metrics['tcp_handshake_analysis'] = analysis
                
            else:
                metrics['tcp_handshake_error'] = f"Connection failed with error {error}"
        else:
            metrics['tcp_handshake_error'] = "Connection timeout during handshake"
        
        sock.close()
        
    except Exception as e:
        print(f"TCP handshake analysis failed for {hostname}:{port}: {e}")
        metrics['tcp_handshake_error'] = str(e)
    
    return metrics


if __name__ == "__main__":
    print("Testing TCP Handshake Timing Analysis")
    print("=" * 50)
    
    # Test with multiple targets
    targets = [
        ('www.google.com', 80),
        ('www.github.com', 80),
        ('httpbin.org', 80)
    ]
    
    for hostname, port in targets:
        print(f"\nTesting {hostname}:{port}")
        print("-" * 30)
        
        result = test_tcp_handshake_timing(hostname, port)
        
        for key, value in result.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}ms" if 'time' in key or 'delay' in key else f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")