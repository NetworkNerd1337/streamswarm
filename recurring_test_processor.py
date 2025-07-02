#!/usr/bin/env python3
"""
Recurring Test Processor for StreamSwarm
Processes recurring tests and creates new test instances when needed.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from app import app, db
from models import Test, TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecurringTestProcessor:
    """Processor for handling recurring tests"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.check_interval = 60  # Check every minute
        
    def start(self):
        """Start the recurring test processor"""
        if self.running:
            logger.warning("RecurringTestProcessor is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("RecurringTestProcessor started")
        
    def stop(self):
        """Stop the recurring test processor"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("RecurringTestProcessor stopped")
        
    def _process_loop(self):
        """Main processing loop"""
        with app.app_context():
            while self.running:
                try:
                    self._process_recurring_tests()
                except Exception as e:
                    logger.error(f"Error processing recurring tests: {e}")
                
                # Sleep for check interval
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
    
    def _process_recurring_tests(self):
        """Process all recurring tests that need to be executed"""
        now = datetime.now()
        
        # Find recurring tests that need to be executed
        tests_to_process = db.session.query(Test).filter(
            Test.is_recurring == True,
            Test.next_execution <= now,
            Test.status.in_(['completed', 'pending'])  # Only process completed or pending tests
        ).all()
        
        if not tests_to_process:
            return
            
        logger.info(f"Processing {len(tests_to_process)} recurring tests")
        
        for original_test in tests_to_process:
            try:
                self._create_recurring_test_instance(original_test)
                self._update_next_execution(original_test)
            except Exception as e:
                logger.error(f"Error processing recurring test {original_test.id}: {e}")
        
        db.session.commit()
    
    def _create_recurring_test_instance(self, original_test):
        """Create a new test instance from a recurring test"""
        # Create new test instance
        new_test = Test(
            name=f"{original_test.name} (Auto-{datetime.now().strftime('%Y%m%d-%H%M')})",
            description=f"Recurring instance of: {original_test.description}" if original_test.description else "Auto-generated recurring test",
            destination=original_test.destination,
            duration=original_test.duration,
            interval=original_test.interval,
            packet_size=original_test.packet_size,
            scheduled_time=datetime.now(),  # Schedule immediately
            is_recurring=False,  # Instance is not recurring itself
            parent_test_id=original_test.id,  # Link to original recurring test
            status='pending'
        )
        
        db.session.add(new_test)
        db.session.flush()  # Get the new test ID
        
        # Copy client assignments from original test
        original_clients = db.session.query(TestClient).filter_by(test_id=original_test.id).all()
        
        for original_client in original_clients:
            new_test_client = TestClient(
                test_id=new_test.id,
                client_id=original_client.client_id,
                status='assigned'
            )
            db.session.add(new_test_client)
        
        logger.info(f"Created recurring test instance {new_test.id} from original test {original_test.id}")
        
    def _update_next_execution(self, test):
        """Update the next execution time for a recurring test"""
        if not test.recurrence_interval:
            logger.warning(f"Recurring test {test.id} missing recurrence interval")
            return
            
        # Set next execution time
        if test.next_execution:
            test.next_execution = test.next_execution + timedelta(seconds=test.recurrence_interval)
        else:
            test.next_execution = datetime.now() + timedelta(seconds=test.recurrence_interval)
        
        logger.info(f"Updated next execution for test {test.id} to {test.next_execution}")

# Global processor instance
processor = None

def start_recurring_processor():
    """Start the global recurring test processor"""
    global processor
    if processor is None:
        processor = RecurringTestProcessor()
    processor.start()

def stop_recurring_processor():
    """Stop the global recurring test processor"""
    global processor
    if processor:
        processor.stop()

if __name__ == "__main__":
    # Run as standalone script for testing
    start_recurring_processor()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_recurring_processor()