#!/usr/bin/env python3
"""
HTTP-based VoIP Service for StreamSwarm
Works within Replit's network constraints by using HTTP instead of UDP SIP
"""

import json
import time
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HTTPVoIPService:
    """
    HTTP-based VoIP service that simulates SIP/RTP functionality
    Uses HTTP endpoints instead of UDP SIP for Replit compatibility
    """
    
    def __init__(self):
        self.active_sessions = {}
        self.rtp_sessions = {}
        
    def create_sip_session(self, client_id, call_id):
        """Create a new SIP session and allocate RTP port"""
        try:
            # Allocate RTP port (simulated)
            rtp_port = random.randint(10000, 20000)
            
            session_data = {
                'session_id': call_id,
                'client_id': client_id,
                'rtp_port': rtp_port,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.active_sessions[call_id] = session_data
            
            # Create RTP echo session
            self.rtp_sessions[call_id] = {
                'port': rtp_port,
                'packets_sent': 0,
                'packets_received': 0,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"HTTP VoIP session created: {call_id}, RTP port: {rtp_port}")
            
            return {
                'success': True,
                'session_id': call_id,
                'rtp_port': rtp_port,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Failed to create SIP session: {e}")
            return {'success': False, 'error': str(e)}
    
    def simulate_rtp_echo(self, session_id, packets_sent):
        """Simulate RTP echo response for testing"""
        try:
            if session_id not in self.rtp_sessions:
                return {
                    'success': False,
                    'error': 'Session not found',
                    'packet_loss_rate': 100.0,
                    'rtp_jitter_avg': 0.0,
                    'rtp_latency_avg': 0.0
                }
            
            session = self.rtp_sessions[session_id]
            session['packets_sent'] = packets_sent
            
            # Simulate realistic VoIP metrics based on network conditions
            # In production, this would be actual RTP echo measurements
            packet_loss_rate = random.uniform(0.1, 2.0)  # 0.1-2% packet loss
            jitter_avg = random.uniform(5.0, 25.0)  # 5-25ms jitter
            latency_avg = random.uniform(20.0, 80.0)  # 20-80ms latency
            
            packets_received = int(packets_sent * (1 - packet_loss_rate / 100))
            session['packets_received'] = packets_received
            
            return {
                'success': True,
                'session_id': session_id,
                'packets_sent': packets_sent,
                'packets_received': packets_received,
                'packet_loss_rate': packet_loss_rate,
                'rtp_jitter_avg': jitter_avg,
                'rtp_latency_avg': latency_avg,
                'rtp_stream_duration': 30.0
            }
            
        except Exception as e:
            logger.error(f"RTP echo simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'packet_loss_rate': 100.0,
                'rtp_jitter_avg': 0.0,
                'rtp_latency_avg': 0.0
            }
    
    def terminate_session(self, session_id):
        """Terminate a VoIP session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.rtp_sessions:
                del self.rtp_sessions[session_id]
                
            logger.info(f"HTTP VoIP session terminated: {session_id}")
            return {'success': True, 'session_id': session_id}
            
        except Exception as e:
            logger.error(f"Failed to terminate session: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_session_status(self, session_id):
        """Get status of a VoIP session"""
        if session_id in self.active_sessions:
            return {
                'success': True,
                'session': self.active_sessions[session_id],
                'rtp_stats': self.rtp_sessions.get(session_id, {})
            }
        else:
            return {'success': False, 'error': 'Session not found'}
    
    def get_service_status(self):
        """Get overall service status"""
        return {
            'active_sessions': len(self.active_sessions),
            'rtp_sessions': len(self.rtp_sessions),
            'running': True,
            'service_type': 'HTTP-based VoIP'
        }

# Global HTTP VoIP service instance
http_voip_service = HTTPVoIPService()