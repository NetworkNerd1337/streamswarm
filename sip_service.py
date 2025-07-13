#!/usr/bin/env python3
"""
SIP Service Component for StreamSwarm VoIP Analysis
Provides server-side SIP functionality for client testing
"""

import socket
import threading
import time
import json
import logging
import struct
import random
from datetime import datetime, timedelta
import zoneinfo

logger = logging.getLogger(__name__)

class SIPService:
    """
    Lightweight SIP service for VoIP analysis testing
    Acts as SIP server endpoint for client testing
    """
    
    def __init__(self, host='0.0.0.0', sip_port=5060, rtp_port_range=(10000, 20000)):
        self.host = host
        self.sip_port = sip_port
        self.rtp_port_range = rtp_port_range
        self.running = False
        self.sip_socket = None
        self.rtp_sessions = {}
        self.registered_clients = {}
        self.call_sessions = {}
        
    def start(self):
        """Start the SIP service"""
        try:
            self.sip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sip_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sip_socket.bind((self.host, self.sip_port))
            self.running = True
            
            # Start SIP listener thread
            self.sip_thread = threading.Thread(target=self._sip_listener, daemon=True)
            self.sip_thread.start()
            
            logger.info(f"SIP service started on {self.host}:{self.sip_port}")
            
        except Exception as e:
            logger.error(f"Failed to start SIP service: {e}")
            raise
    
    def stop(self):
        """Stop the SIP service"""
        self.running = False
        if self.sip_socket:
            self.sip_socket.close()
        logger.info("SIP service stopped")
    
    def _sip_listener(self):
        """Main SIP message listener"""
        while self.running:
            try:
                data, addr = self.sip_socket.recvfrom(4096)
                message = data.decode('utf-8')
                
                # Process SIP message in separate thread
                threading.Thread(
                    target=self._process_sip_message,
                    args=(message, addr),
                    daemon=True
                ).start()
                
            except Exception as e:
                if self.running:
                    logger.error(f"SIP listener error: {e}")
    
    def _process_sip_message(self, message, addr):
        """Process incoming SIP message"""
        try:
            lines = message.strip().split('\n')
            if not lines:
                return
                
            request_line = lines[0]
            headers = {}
            
            # Parse headers
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            # Handle different SIP methods
            if request_line.startswith('REGISTER'):
                self._handle_register(request_line, headers, addr)
            elif request_line.startswith('INVITE'):
                self._handle_invite(request_line, headers, addr)
            elif request_line.startswith('BYE'):
                self._handle_bye(request_line, headers, addr)
            elif request_line.startswith('OPTIONS'):
                self._handle_options(request_line, headers, addr)
                
        except Exception as e:
            logger.error(f"Error processing SIP message: {e}")
    
    def _handle_register(self, request_line, headers, addr):
        """Handle SIP REGISTER request"""
        try:
            call_id = headers.get('Call-ID', '')
            from_header = headers.get('From', '')
            contact = headers.get('Contact', '')
            
            # Store client registration
            client_id = f"{addr[0]}:{addr[1]}"
            self.registered_clients[client_id] = {
                'addr': addr,
                'call_id': call_id,
                'from': from_header,
                'contact': contact,
                'registered_at': datetime.now(zoneinfo.ZoneInfo('America/New_York')),
                'expires': datetime.now(zoneinfo.ZoneInfo('America/New_York')) + timedelta(seconds=3600)
            }
            
            # Send 200 OK response
            response = self._create_sip_response(
                "200 OK",
                headers,
                {
                    'Contact': contact,
                    'Expires': '3600'
                }
            )
            
            self.sip_socket.sendto(response.encode('utf-8'), addr)
            logger.info(f"SIP REGISTER successful for {client_id}")
            
        except Exception as e:
            logger.error(f"Error handling REGISTER: {e}")
            self._send_error_response(400, "Bad Request", headers, addr)
    
    def _handle_invite(self, request_line, headers, addr):
        """Handle SIP INVITE request"""
        try:
            call_id = headers.get('Call-ID', '')
            from_header = headers.get('From', '')
            to_header = headers.get('To', '')
            
            # Allocate RTP port for media session
            rtp_port = self._allocate_rtp_port()
            
            # Create call session
            session_id = f"{call_id}_{addr[0]}_{addr[1]}"
            self.call_sessions[session_id] = {
                'addr': addr,
                'call_id': call_id,
                'from': from_header,
                'to': to_header,
                'rtp_port': rtp_port,
                'start_time': datetime.now(zoneinfo.ZoneInfo('America/New_York')),
                'status': 'active'
            }
            
            # Create SDP (Session Description Protocol) for media
            sdp = self._create_sdp(rtp_port)
            
            # Send 200 OK response with SDP
            response = self._create_sip_response(
                "200 OK",
                headers,
                {
                    'Content-Type': 'application/sdp',
                    'Content-Length': str(len(sdp))
                },
                sdp
            )
            
            self.sip_socket.sendto(response.encode('utf-8'), addr)
            
            # Start RTP echo service for this session
            self._start_rtp_echo(session_id, rtp_port)
            
            logger.info(f"SIP INVITE successful for {session_id}, RTP port {rtp_port}")
            
        except Exception as e:
            logger.error(f"Error handling INVITE: {e}")
            self._send_error_response(500, "Internal Server Error", headers, addr)
    
    def _handle_bye(self, request_line, headers, addr):
        """Handle SIP BYE request"""
        try:
            call_id = headers.get('Call-ID', '')
            session_id = f"{call_id}_{addr[0]}_{addr[1]}"
            
            # End call session
            if session_id in self.call_sessions:
                session = self.call_sessions[session_id]
                session['status'] = 'ended'
                session['end_time'] = datetime.now(zoneinfo.ZoneInfo('America/New_York'))
                
                # Stop RTP echo service
                self._stop_rtp_echo(session_id)
                
                # Send 200 OK response
                response = self._create_sip_response("200 OK", headers)
                self.sip_socket.sendto(response.encode('utf-8'), addr)
                
                logger.info(f"SIP BYE successful for {session_id}")
            else:
                self._send_error_response(481, "Call/Transaction Does Not Exist", headers, addr)
                
        except Exception as e:
            logger.error(f"Error handling BYE: {e}")
            self._send_error_response(500, "Internal Server Error", headers, addr)
    
    def _handle_options(self, request_line, headers, addr):
        """Handle SIP OPTIONS request"""
        try:
            response = self._create_sip_response(
                "200 OK",
                headers,
                {
                    'Allow': 'INVITE, ACK, BYE, CANCEL, OPTIONS, REGISTER',
                    'Accept': 'application/sdp',
                    'Supported': 'replaces'
                }
            )
            
            self.sip_socket.sendto(response.encode('utf-8'), addr)
            logger.info(f"SIP OPTIONS successful for {addr}")
            
        except Exception as e:
            logger.error(f"Error handling OPTIONS: {e}")
            self._send_error_response(500, "Internal Server Error", headers, addr)
    
    def _create_sip_response(self, status_line, request_headers, additional_headers=None, body=None):
        """Create SIP response message"""
        response = f"SIP/2.0 {status_line}\r\n"
        
        # Copy essential headers from request
        for header in ['Via', 'From', 'To', 'Call-ID', 'CSeq']:
            if header in request_headers:
                response += f"{header}: {request_headers[header]}\r\n"
        
        # Add additional headers
        if additional_headers:
            for key, value in additional_headers.items():
                response += f"{key}: {value}\r\n"
        
        # Add server header
        response += f"Server: StreamSwarm-SIP/1.0\r\n"
        response += f"User-Agent: StreamSwarm-SIP/1.0\r\n"
        
        response += "\r\n"
        
        # Add body if provided
        if body:
            response += body
            
        return response
    
    def _send_error_response(self, code, reason, headers, addr):
        """Send SIP error response"""
        response = self._create_sip_response(f"{code} {reason}", headers)
        self.sip_socket.sendto(response.encode('utf-8'), addr)
    
    def _create_sdp(self, rtp_port):
        """Create SDP (Session Description Protocol) for media session"""
        sdp = f"""v=0
o=StreamSwarm 123456 654321 IN IP4 {self.host}
s=VoIP Analysis Session
c=IN IP4 {self.host}
t=0 0
m=audio {rtp_port} RTP/AVP 0 8
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=sendrecv
"""
        return sdp
    
    def _allocate_rtp_port(self):
        """Allocate available RTP port"""
        for port in range(self.rtp_port_range[0], self.rtp_port_range[1], 2):
            if port not in [session['rtp_port'] for session in self.call_sessions.values()]:
                return port
        raise Exception("No available RTP ports")
    
    def _start_rtp_echo(self, session_id, rtp_port):
        """Start RTP echo service for session"""
        try:
            rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtp_socket.bind((self.host, rtp_port))
            
            self.rtp_sessions[session_id] = {
                'socket': rtp_socket,
                'port': rtp_port,
                'packets_received': 0,
                'packets_sent': 0,
                'start_time': datetime.now(zoneinfo.ZoneInfo('America/New_York'))
            }
            
            # Start RTP echo thread
            threading.Thread(
                target=self._rtp_echo_worker,
                args=(session_id, rtp_socket),
                daemon=True
            ).start()
            
            logger.info(f"RTP echo service started for {session_id} on port {rtp_port}")
            
        except Exception as e:
            logger.error(f"Failed to start RTP echo for {session_id}: {e}")
    
    def _rtp_echo_worker(self, session_id, rtp_socket):
        """RTP echo worker thread"""
        while session_id in self.rtp_sessions and session_id in self.call_sessions:
            try:
                # Set socket timeout for clean shutdown
                rtp_socket.settimeout(1.0)
                
                try:
                    data, addr = rtp_socket.recvfrom(1024)
                    
                    # Update packet count
                    if session_id in self.rtp_sessions:
                        self.rtp_sessions[session_id]['packets_received'] += 1
                        
                        # Echo the packet back
                        rtp_socket.sendto(data, addr)
                        self.rtp_sessions[session_id]['packets_sent'] += 1
                        
                except socket.timeout:
                    continue
                    
            except Exception as e:
                logger.error(f"RTP echo error for {session_id}: {e}")
                break
    
    def _stop_rtp_echo(self, session_id):
        """Stop RTP echo service for session"""
        if session_id in self.rtp_sessions:
            session = self.rtp_sessions[session_id]
            session['socket'].close()
            del self.rtp_sessions[session_id]
            logger.info(f"RTP echo service stopped for {session_id}")
    
    def get_session_stats(self, session_id):
        """Get statistics for a session"""
        if session_id in self.call_sessions:
            call_session = self.call_sessions[session_id]
            rtp_session = self.rtp_sessions.get(session_id, {})
            
            return {
                'call_session': call_session,
                'rtp_session': rtp_session,
                'packets_received': rtp_session.get('packets_received', 0),
                'packets_sent': rtp_session.get('packets_sent', 0)
            }
        return None


# Global SIP service instance
sip_service = SIPService()


def start_sip_service():
    """Start the global SIP service"""
    try:
        sip_service.start()
        logger.info("SIP service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start SIP service: {e}")
        return False


def stop_sip_service():
    """Stop the global SIP service"""
    try:
        sip_service.stop()
        logger.info("SIP service stopped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to stop SIP service: {e}")
        return False


def get_sip_service_status():
    """Get SIP service status"""
    return {
        'running': sip_service.running,
        'registered_clients': len(sip_service.registered_clients),
        'active_sessions': len(sip_service.call_sessions),
        'rtp_sessions': len(sip_service.rtp_sessions)
    }