#!/usr/bin/env python3
"""
Test script to manually debug geolocation processing for test 87 result 779
"""

import json
from app import app, db
from models import TestResult
from server_geo_processor import geo_processor

def test_geolocation_processing():
    with app.app_context():
        # Get the specific result that's failing
        result = TestResult.query.get(779)
        if not result:
            print("Result 779 not found")
            return
            
        print(f"Processing result {result.id} from test {result.test_id}")
        print(f"Has traceroute data: {result.traceroute_data is not None}")
        
        if result.traceroute_data:
            # Parse the traceroute data
            traceroute_lines = json.loads(result.traceroute_data)
            print(f"Number of traceroute lines: {len(traceroute_lines)}")
            
            # Show first few lines
            for i, line in enumerate(traceroute_lines[:5]):
                print(f"Line {i}: {line}")
            
            # Test the geolocation processing
            try:
                success = geo_processor._process_single_result(result)
                print(f"Processing success: {success}")
                
                if success:
                    print(f"Map HTML length: {len(result.path_map_html) if result.path_map_html else 0}")
                    print(f"Total distance: {result.path_total_distance_km}")
                    print(f"Geographic efficiency: {result.path_geographic_efficiency}")
                    
                    # Commit the changes
                    db.session.commit()
                    print("Changes committed to database")
                else:
                    print("Processing failed - no geolocation data generated")
                    
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_geolocation_processing()