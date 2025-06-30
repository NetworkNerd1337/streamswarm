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
            logger.debug(f"Starting geolocation processing for result {result.id}")
            
            # Parse traceroute data
            if not result.traceroute_data:
                logger.debug(f"Result {result.id} has no traceroute data")
                return False
                
            traceroute_lines = json.loads(result.traceroute_data)
            if not traceroute_lines:
                logger.debug(f"Result {result.id} has empty traceroute data")
                return False
            
            logger.debug(f"Result {result.id} has {len(traceroute_lines)} traceroute lines")
            
            # Get the test destination
            from models import Test
            test = Test.query.get(result.test_id)
            if not test:
                logger.debug(f"No test found for result {result.id}")
                return False
                
            destination = test.destination
            logger.debug(f"Processing result {result.id} for destination: {destination}")
            
            # Perform geolocation analysis
            path_analysis = self.geo_service.analyze_traceroute_path(traceroute_lines, destination)
            logger.debug(f"Path analysis completed for result {result.id}")
            
            if path_analysis and path_analysis.get('hops'):
                logger.debug(f"Result {result.id} has {len(path_analysis.get('hops', []))} hops")
                
                # Generate map
                map_html = self.geo_service.create_path_map(path_analysis, destination)
                logger.debug(f"Generated map HTML for result {result.id}, size: {len(map_html) if map_html else 0} chars")
                
                # Update the result with geolocation data
                result.path_map_html = map_html
                result.path_total_distance_km = path_analysis.get('total_distance_km')
                result.path_geographic_efficiency = path_analysis.get('geographic_efficiency')
                
                logger.info(f"Enhanced result {result.id} with geolocation data - map size: {len(map_html) if map_html else 0}")
                return True
            else:
                logger.debug(f"Result {result.id} path analysis failed or has no hops")
            
        except Exception as e:
            logger.error(f"Error processing geolocation for result {result.id}: {str(e)}", exc_info=True)
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