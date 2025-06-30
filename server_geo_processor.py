"""
Server-side geolocation processor for StreamSwarm
Processes traceroute data to generate geolocation maps when clients don't have geolocation capabilities
"""

import json
import logging
from typing import Dict, List, Optional
from geolocation_service import GeolocationService
from models import TestResult, db

logger = logging.getLogger(__name__)

class ServerGeolocationProcessor:
    """
    Server-side geolocation processor that enhances test results with geolocation data
    when clients don't have geolocation capabilities
    """
    
    def __init__(self):
        self.geo_service = GeolocationService()
        logger.info("Server-side geolocation processor initialized")
    
    def process_pending_results(self) -> int:
        """
        Process test results that have traceroute data but no geolocation data
        Returns number of results processed
        """
        processed_count = 0
        
        # Find results with traceroute data but no geolocation data
        pending_results = TestResult.query.filter(
            TestResult.traceroute_data.isnot(None),
            TestResult.path_map_html.is_(None)
        ).limit(50).all()  # Process in batches to avoid overwhelming the system
        
        for result in pending_results:
            try:
                if self._process_single_result(result):
                    processed_count += 1
                    # Commit each result individually to ensure it's saved
                    db.session.commit()
                    logger.info(f"Successfully processed and saved geolocation for result {result.id}")
                    
            except Exception as e:
                logger.error(f"Error processing result {result.id}: {str(e)}")
                db.session.rollback()  # Rollback failed transaction
                continue
        
        logger.info(f"Completed processing: {processed_count} geolocation results processed")
        
        return processed_count
    
    def _process_single_result(self, result: TestResult) -> bool:
        """
        Process a single test result to add geolocation data
        Returns True if processing was successful
        """
        try:
            # Parse traceroute data
            if not result.traceroute_data:
                return False
                
            traceroute_lines = json.loads(result.traceroute_data)
            if not traceroute_lines:
                return False
            
            # Get the test destination
            from models import Test
            test = Test.query.get(result.test_id)
            if not test:
                return False
                
            destination = test.destination
            
            # Perform geolocation analysis
            path_analysis = self.geo_service.analyze_traceroute_path(traceroute_lines, destination)
            
            if path_analysis and path_analysis.get('hops'):
                # Generate map
                map_html = self.geo_service.create_path_map(path_analysis, destination)
                
                # Update the result with geolocation data
                result.path_map_html = map_html
                result.path_total_distance_km = path_analysis.get('total_distance_km')
                result.path_geographic_efficiency = path_analysis.get('geographic_efficiency')
                
                logger.debug(f"Enhanced result {result.id} with geolocation data")
                return True
            
        except Exception as e:
            logger.error(f"Error processing geolocation for result {result.id}: {str(e)}")
            return False
        
        return False
    
    def process_result_by_id(self, result_id: int) -> bool:
        """
        Process a specific test result by ID
        Returns True if processing was successful
        """
        result = TestResult.query.get(result_id)
        if not result:
            return False
            
        success = self._process_single_result(result)
        if success:
            db.session.commit()
            
        return success

# Global processor instance
geo_processor = ServerGeolocationProcessor()