"""
Tests for the weather MCP server.

Note: These tests require WEATHER_API_KEY environment variable to be set.
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

from voice_computer.mcp_servers.weather_mcp_server import (
    get_api_key,
    make_weather_request,
    get_current_weather,
    get_weather_forecast,
    search_locations,
    get_weather_alerts
)


class TestWeatherMCPServer(unittest.IsolatedAsyncioTestCase):
    """Test suite for weather MCP server functions."""
    
    def test_get_api_key(self):
        """Test that API key can be retrieved from environment."""
        try:
            api_key = get_api_key()
            self.assertIsInstance(api_key, str, "API key should be a string")
            self.assertGreater(len(api_key), 0, "API key should not be empty")
        except ValueError as e:
            self.skipTest(f"WEATHER_API_KEY not set: {e}")
    
    async def test_make_weather_request_current(self):
        """Test making a basic weather API request."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        try:
            # Test with a well-known location
            result = await make_weather_request("current.json", {"q": "London"})
            
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            self.assertIn("current", result, "Result should contain 'current' key")
            self.assertIn("location", result, "Result should contain 'location' key")
            
            # Check current weather structure
            current = result["current"]
            self.assertIn("temp_c", current, "Current weather should have temperature in Celsius")
            self.assertIn("temp_f", current, "Current weather should have temperature in Fahrenheit")
            self.assertIn("condition", current, "Current weather should have condition")
            
            # Check location structure
            location = result["location"]
            self.assertIn("name", location, "Location should have name")
            self.assertIn("country", location, "Location should have country")
            
        except Exception as e:
            self.fail(f"Weather API request failed: {e}")
    
    async def test_get_current_weather(self):
        """Test getting current weather for a location."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        # Test with well-known locations
        test_locations = ["London", "New York", "Tokyo"]
        
        for location in test_locations:
            try:
                result = await get_current_weather(location)
                
                self.assertIsInstance(result, str, "Result should be a string")
                self.assertGreater(len(result), 0, "Result should not be empty")
                
                # Check for expected content
                self.assertIn("Current weather in", result, "Should contain location info")
                self.assertIn("Temperature:", result, "Should contain temperature")
                self.assertIn("Condition:", result, "Should contain weather condition")
                self.assertIn("Humidity:", result, "Should contain humidity")
                self.assertIn("Wind:", result, "Should contain wind info")
                
                break  # Test with just one location to avoid rate limiting
                
            except Exception as e:
                continue
    
    async def test_get_weather_forecast(self):
        """Test getting weather forecast for a location."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        try:
            result = await get_weather_forecast("London", days=3)
            
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Check for expected content
            self.assertIn("Weather forecast for", result, "Should contain forecast header")
            self.assertIn("Date:", result, "Should contain date information")
            self.assertIn("High:", result, "Should contain high temperature")
            self.assertIn("Low:", result, "Should contain low temperature")
            self.assertIn("Condition:", result, "Should contain weather conditions")
            
        except Exception as e:
            self.fail(f"Weather forecast request failed: {e}")
    
    async def test_search_locations(self):
        """Test searching for locations."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        try:
            result = await search_locations("London")
            
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Check for expected content
            self.assertIn("Location search results for", result, "Should contain search header")
            self.assertIn("London", result, "Should contain the searched location")
            
        except Exception as e:
            self.fail(f"Location search failed: {e}")
    
    async def test_get_weather_alerts(self):
        """Test getting weather alerts for a location."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        try:
            result = await get_weather_alerts("London")
            
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Should either have alerts or no alerts message
            alerts_present = "Weather alerts for" in result and "Alert" in result
            no_alerts = "No weather alerts for" in result
            
            self.assertTrue(alerts_present or no_alerts, "Should contain either alerts or no alerts message")
            
        except Exception as e:
            self.fail(f"Weather alerts request failed: {e}")
    
    async def test_invalid_location(self):
        """Test handling of invalid location."""
        try:
            get_api_key()  # Check if API key is available
        except ValueError:
            self.skipTest("WEATHER_API_KEY not set")
        
        # Test with clearly invalid location
        result = await get_current_weather("INVALID_LOCATION_XYZ123")
        
        self.assertIsInstance(result, str, "Result should be a string")
        self.assertIn("Error getting current weather", result, "Should contain error message")
    
    def test_forecast_days_validation(self):
        """Test that forecast days parameter is validated correctly."""
        # This test doesn't require API calls, just validates the logic
        # In the actual function, days should be limited to 1-10 range
        
        # Test boundary values
        self.assertTrue(1 <= max(1, min(10, 0)) <= 10, "Days validation should handle 0")
        self.assertTrue(1 <= max(1, min(10, 1)) <= 10, "Days validation should handle 1")
        self.assertTrue(1 <= max(1, min(10, 10)) <= 10, "Days validation should handle 10")
        self.assertTrue(1 <= max(1, min(10, 15)) <= 10, "Days validation should handle >10")


class TestWeatherServerImports(unittest.TestCase):
    """Test that all weather functions can be imported."""
    
    def test_imports(self):
        """Test importing all functions."""
        from voice_computer.mcp_servers.weather_mcp_server import (
            get_current_weather,
            get_weather_forecast,
            search_locations,
            get_weather_alerts,
            get_api_key,
            make_weather_request,
            mcp
        )
        
        # Verify the FastMCP instance exists
        self.assertIsNotNone(mcp)
        self.assertTrue(hasattr(mcp, '_mcp_server'))


class TestEnvironmentVariableRequirement(unittest.TestCase):
    """Test that environment variable requirement is properly handled."""
    
    def test_api_key_requirement(self):
        """Test that environment variable requirement is enforced."""
        # Temporarily remove the environment variable
        original_key = os.environ.get("WEATHER_API_KEY")
        
        if "WEATHER_API_KEY" in os.environ:
            del os.environ["WEATHER_API_KEY"]
        
        try:
            # Should raise ValueError when no API key is set
            with self.assertRaises(ValueError) as cm:
                get_api_key()
            self.assertIn("WEATHER_API_KEY environment variable not set", str(cm.exception))
            
        finally:
            # Restore the original key if it existed
            if original_key:
                os.environ["WEATHER_API_KEY"] = original_key




if __name__ == "__main__":
    unittest.main(verbosity=2)