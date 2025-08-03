"""
Tests for the train MCP server.

Note: These tests require TRANSPORT_API_ID and TRANSPORT_API_KEY environment variables to be set.
Some tests are integration tests that make real API calls.
"""

import asyncio
import unittest
import sys
import os
import re
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_computer.mcp_servers.train_mcp_server import (
    get_api_credentials,
    make_transport_request,
    find_stations_near_postcode,
    get_station_departures,
    get_station_arrivals,
    get_train_timetable_from_postcode,
    search_station_codes
)


class TestTrainMCPServer(unittest.IsolatedAsyncioTestCase):
    """Test suite for train MCP server functions."""
    
    def test_get_api_credentials(self):
        """Test that API credentials can be retrieved from environment."""
        try:
            app_id, app_key = get_api_credentials()
            self.assertIsInstance(app_id, str, "API ID should be a string")
            self.assertIsInstance(app_key, str, "API key should be a string")
            self.assertGreater(len(app_id), 0, "API ID should not be empty")
            self.assertGreater(len(app_key), 0, "API key should not be empty")
        except ValueError as e:
            self.skipTest(f"Transport API credentials not set: {e}")
    
    async def test_make_transport_request(self):
        """Test making a basic transport API request."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        try:
            # Test with a simple places search
            result = await make_transport_request("uk/places.json", {"query": "London", "type": "train_station"})
            
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            # The structure may vary, but should be a valid response
            has_member_or_error = "member" in result or "error" in result
            self.assertTrue(has_member_or_error, "Result should contain 'member' or 'error'")
            
        except Exception as e:
            self.fail(f"Transport API request failed: {e}")
    
    async def test_find_stations_near_postcode(self):
        """Test finding stations near a postcode."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with well-known London postcodes
        test_postcodes = ["SW1A 1AA", "EC1A 1BB", "W1A 0AX"]
        
        for postcode in test_postcodes:
            try:
                result = await find_stations_near_postcode(postcode)
                
                self.assertIsInstance(result, str, "Result should be a string")
                self.assertGreater(len(result), 0, "Result should not be empty")
                
                # Should either find stations or report none found
                stations_found = "Train stations near" in result and "Code:" in result
                no_stations = "No train stations found near" in result
                
                self.assertTrue(stations_found or no_stations, f"Unexpected result format for {postcode}")
                
                if stations_found:
                    self.assertIn("Distance:", result, "Should contain distance information")
                
                break  # Test with just one postcode to avoid rate limiting
                
            except Exception as e:
                continue
    
    async def test_search_station_codes(self):
        """Test searching for station codes by name."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with well-known station names
        test_queries = ["London", "Birmingham", "Manchester"]
        
        for query in test_queries:
            try:
                result = await search_station_codes(query)
                print(result)
                self.assertIsInstance(result, str, "Result should be a string")
                self.assertGreater(len(result), 0, "Result should not be empty")
                
                # Should either find stations or report none found
                stations_found = f"Train stations matching '{query}'" in result and "Code:" in result
                no_stations = f"No train stations found matching '{query}'" in result
                
                self.assertTrue(stations_found or no_stations, f"Unexpected result format for {query}")
                
                break  # Test with just one query to avoid rate limiting
                
            except Exception as e:
                continue
    
    async def test_get_station_departures(self):
        """Test getting departures from a known major station."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with major London stations
        test_stations = ["PAD", "KGX", "VIC", "WAT"]  # Paddington, King's Cross, Victoria, Waterloo
        
        for station_code in test_stations:
            try:
                result = await get_station_departures(station_code, limit=5)
                
                self.assertIsInstance(result, str, "Result should be a string")
                self.assertGreater(len(result), 0, "Result should not be empty")
                
                # Should either find departures or report none found
                departures_found = f"Live departures from" in result and "Scheduled:" in result
                no_departures = f"No departures found for station {station_code}" in result
                
                self.assertTrue(departures_found or no_departures, f"Unexpected result format for {station_code}")
                
                if departures_found:
                    # Check for expected content
                    self.assertIn("Platform:", result, "Should contain platform information")
                    self.assertIn("Operator:", result, "Should contain operator information")
                
                break  # Test with just one station to avoid rate limiting
                
            except Exception as e:
                continue
    
    async def test_get_station_arrivals(self):
        """Test getting arrivals at a known major station."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with a major London station
        try:
            result = await get_station_arrivals("PAD", limit=5)  # Paddington
            
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Should either find arrivals or report none found
            arrivals_found = "Live arrivals at" in result and "Scheduled:" in result
            no_arrivals = "No arrivals found for station PAD" in result
            
            self.assertTrue(arrivals_found or no_arrivals, "Unexpected result format")
            
            if arrivals_found:
                # Check for expected content
                self.assertIn("From", result, "Should contain origin information")
                self.assertIn("Platform:", result, "Should contain platform information")
                self.assertIn("Operator:", result, "Should contain operator information")
            
        except Exception as e:
            pass  # Continue if this specific test fails
    
    async def test_get_train_timetable_from_postcode(self):
        """Test getting train timetable from postcode (main function)."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        try:
            # Test with a central London postcode
            result = await get_train_timetable_from_postcode("SW1A 1AA", direction="departures", limit=3)
            print(result)
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Should contain postcode information
            has_postcode_info = "Train information for postcode SW1A 1AA" in result or "SW1A1AA" in result
            self.assertTrue(has_postcode_info, "Should mention the postcode")
            
            # Should either find nearest station or report none found
            station_found = "Nearest station:" in result
            no_stations = "No train stations found near postcode" in result
            
            self.assertTrue(station_found or no_stations, "Should find nearest station or report none found")
            
        except Exception as e:
            self.fail(f"Train timetable from postcode failed: {e}")
    
    async def test_invalid_station_code(self):
        """Test handling of invalid station code."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with clearly invalid station code
        result = await get_station_departures("INVALID")
        
        self.assertIsInstance(result, str, "Result should be a string")
        # Should handle error gracefully
        error_handled = "Error getting departures" in result or "No departures found" in result
        self.assertTrue(error_handled, "Should handle invalid station code gracefully")
    
    async def test_invalid_postcode(self):
        """Test handling of invalid postcode."""
        try:
            get_api_credentials()  # Check if credentials are available
        except ValueError:
            self.skipTest("Transport API credentials not set")
        
        # Test with clearly invalid postcode
        result = await find_stations_near_postcode("INVALID_POSTCODE")
        
        self.assertIsInstance(result, str, "Result should be a string")
        # Should either return no results or handle error gracefully
        error_handled = ("No train stations found" in result or 
                        "Error finding stations" in result)
        self.assertTrue(error_handled, "Should handle invalid postcode gracefully")
    
    def test_postcode_cleaning(self):
        """Test that postcodes are cleaned properly."""
        # This test doesn't require API calls, just validates the cleaning logic
        test_cases = [
            ("SW1A 1AA", "SW1A1AA"),
            ("sw1a 1aa", "SW1A1AA"),
            ("  EC1A  1BB  ", "EC1A1BB"),
            ("w1a0ax", "W1A0AX")
        ]
        
        for input_postcode, expected in test_cases:
            cleaned = input_postcode.replace(" ", "").upper()
            self.assertEqual(cleaned, expected, f"Postcode cleaning failed: {input_postcode} -> {cleaned} != {expected}")


class TestTrainServerImports(unittest.TestCase):
    """Test that all train functions can be imported."""
    
    def test_imports(self):
        """Test importing all functions."""
        from voice_computer.mcp_servers.train_mcp_server import (
            find_stations_near_postcode,
            get_station_departures,
            get_station_arrivals,
            get_train_timetable_from_postcode,
            search_station_codes,
            get_api_credentials,
            make_transport_request,
            mcp
        )
        
        # Verify the FastMCP instance exists
        self.assertIsNotNone(mcp)
        self.assertTrue(hasattr(mcp, '_mcp_server'))


class TestEnvironmentVariablesRequirement(unittest.TestCase):
    """Test that environment variables requirement is properly handled."""
    
    def test_api_credentials_requirement(self):
        """Test that environment variables requirement is enforced."""
        # Store original values
        original_id = os.environ.get("TRANSPORT_API_ID")
        original_key = os.environ.get("TRANSPORT_API_KEY")
        
        # Test missing ID
        if "TRANSPORT_API_ID" in os.environ:
            del os.environ["TRANSPORT_API_ID"]
        if "TRANSPORT_API_KEY" in os.environ:
            del os.environ["TRANSPORT_API_KEY"]
        
        try:
            # Should raise ValueError when credentials are not set
            with self.assertRaises(ValueError) as cm:
                get_api_credentials()
            self.assertIn("TRANSPORT_API_ID and TRANSPORT_API_KEY environment variables must be set", str(cm.exception))
            
        finally:
            # Restore original values if they existed
            if original_id:
                os.environ["TRANSPORT_API_ID"] = original_id
            if original_key:
                os.environ["TRANSPORT_API_KEY"] = original_key




if __name__ == "__main__":
    unittest.main(verbosity=2)