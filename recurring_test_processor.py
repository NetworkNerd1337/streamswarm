#!/usr/bin/env python3
"""
Recurring Test Processor for StreamSwarm
Processes recurring tests and creates new test instances when needed.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
import zoneinfo
from app import app, db
from models import Test, TestClient, TestResult

def get_eastern_time():
    """Get current time in Eastern timezone"""
    return datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)

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
        now = get_eastern_time()
        
        # Find recurring tests that need to be executed
        # Include both active and completed tests, as "Create New Tests" mode marks original as completed
        tests_to_process = db.session.query(Test).filter(
            Test.is_recurring == True,
            Test.next_execution <= now,
            Test.parent_test_id.is_(None)  # Only process original recurring tests, not child tests
        ).all()
        
        if not tests_to_process:
            return
            
        logger.info(f"Processing {len(tests_to_process)} recurring tests")
        
        for original_test in tests_to_process:
            try:
                if original_test.recurrence_type == 'new':
                    # For "Create New Tests" mode, the completion handler manages the scheduling
                    # The processor only needs to update next_execution for future reference
                    logger.info(f"Skipping processor action for 'Create New Tests' mode test {original_test.id} - managed by completion handler")
                    self._update_next_execution(original_test)
                else:
                    # For "Continue Same Test" mode, the processor handles restarting
                    self._restart_recurring_test(original_test)
                    self._update_next_execution(original_test)
            except Exception as e:
                logger.error(f"Error processing recurring test {original_test.id}: {e}")
        
        db.session.commit()
    
    def _restart_recurring_test(self, original_test):
        """Restart the original test for recurring execution"""
        # Reset the original test to restart it
        original_test.scheduled_time = get_eastern_time()  # Schedule immediately
        original_test.status = 'pending'
        original_test.started_at = None
        original_test.completed_at = None
        
        # Reset all client assignments to 'assigned' status
        test_clients = db.session.query(TestClient).filter_by(test_id=original_test.id).all()
        for test_client in test_clients:
            test_client.status = 'assigned'
        
        # Clear any previous test results to start fresh
        db.session.query(TestResult).filter_by(test_id=original_test.id).delete()
        
        logger.info(f"Restarted recurring test {original_test.id} for next execution")
    
    def _create_new_recurring_test(self, original_test):
        """Create a new test based on the original recurring test settings"""
        from models import Test, TestClient  # Import here to avoid circular imports
        
        # Create new test with same settings
        new_test = Test(
            name=original_test.name,
            description=original_test.description,
            destination=original_test.destination,
            scheduled_time=get_eastern_time(),  # Schedule immediately
            duration=original_test.duration,
            interval=original_test.interval,
            packet_size=original_test.packet_size,
            test_config=original_test.test_config,
            status='pending',
            is_recurring=False,  # New test is not recurring itself
            recurrence_interval=None,
            recurrence_type='continue',
            parent_test_id=original_test.id,  # Link to original recurring test
            next_execution=None
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
        
        # Update the original test's next execution but keep it active for future recurrences
        self._update_next_execution(original_test)
        # Don't mark original as completed - it needs to stay active to generate future tests
        
        # Ensure the new test is scheduled immediately, not at the next recurrence time
        new_test.scheduled_time = get_eastern_time()
        new_test.status = 'pending'
        
        logger.info(f"Created new test {new_test.id} from recurring test {original_test.id}")
        
    def _update_next_execution(self, test):
        """Update the next execution time for a recurring test"""
        if not test.recurrence_interval:
            logger.warning(f"Recurring test {test.id} missing recurrence interval")
            return
            
        # Set next execution time
        if test.next_execution:
            test.next_execution = test.next_execution + timedelta(seconds=test.recurrence_interval)
        else:
            test.next_execution = get_eastern_time() + timedelta(seconds=test.recurrence_interval)
        
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