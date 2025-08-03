"""
Tests for the time MCP server.
"""

import unittest
import sys
import os
import re
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_computer.mcp_servers.time_mcp_server import (
    current_time,
    current_date,
    current_day_of_week,
    current_datetime
)


class TestTimeMCPServer(unittest.TestCase):
    """Test suite for time MCP server functions."""
    
    def test_current_time_format(self):
        """Test that current_time returns valid HH:MM:SS format."""
        time_str = current_time()
        
        # Should match HH:MM:SS format
        pattern = r'^([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$'
        self.assertIsNotNone(re.match(pattern, time_str), f"Invalid time format: {time_str}")
        
        # Should be able to parse as datetime
        try:
            datetime.strptime(time_str, "%H:%M:%S")
        except ValueError as e:
            self.fail(f"Could not parse time string: {time_str} - {e}")
    
    def test_current_date_format(self):
        """Test that current_date returns valid YYYY-MM-DD format."""
        date_str = current_date()
        
        # Should match YYYY-MM-DD format
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        self.assertIsNotNone(re.match(pattern, date_str), f"Invalid date format: {date_str}")
        
        # Should be able to parse as datetime
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            self.fail(f"Could not parse date string: {date_str} - {e}")
    
    def test_current_day_of_week(self):
        """Test that current_day_of_week returns a valid day name."""
        day_str = current_day_of_week()
        
        valid_days = [
            "Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday", "Sunday"
        ]
        
        self.assertIn(day_str, valid_days, f"Invalid day of week: {day_str}")
    
    def test_current_datetime_format(self):
        """Test that current_datetime returns a readable format."""
        datetime_str = current_datetime()
        
        # Should contain day name, month name, and time
        day_found = any(day in datetime_str for day in [
            "Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday", "Sunday"
        ])
        self.assertTrue(day_found, f"No day name found in: {datetime_str}")
        
        month_found = any(month in datetime_str for month in [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        self.assertTrue(month_found, f"No month name found in: {datetime_str}")
        
        # Should contain "at" keyword
        self.assertIn(" at ", datetime_str, f"Missing 'at' keyword in: {datetime_str}")
    
    def test_time_consistency(self):
        """Test that time functions return consistent values when called rapidly."""
        # Get all time values rapidly
        time1 = current_time()
        date1 = current_date()
        day1 = current_day_of_week()
        datetime1 = current_datetime()
        
        # Values should be consistent (allowing for second changes)
        time2 = current_time()
        date2 = current_date()
        day2 = current_day_of_week()
        datetime2 = current_datetime()
        
        # Date and day should be exactly the same
        self.assertEqual(date1, date2, "Date changed between calls")
        self.assertEqual(day1, day2, "Day changed between calls")
        
        # Time should be very close (within a few seconds)
        t1 = datetime.strptime(time1, "%H:%M:%S")
        t2 = datetime.strptime(time2, "%H:%M:%S")
        time_diff = abs((t2 - t1).total_seconds())
        self.assertLessEqual(time_diff, 2, f"Time difference too large: {time_diff} seconds")
    
    def test_current_time_real_time(self):
        """Test that current_time returns approximately the real current time."""
        system_time = datetime.now()
        mcp_time_str = current_time()
        mcp_time = datetime.strptime(mcp_time_str, "%H:%M:%S")
        
        # Extract just the time components from system time
        system_time_only = system_time.replace(
            year=mcp_time.year, 
            month=mcp_time.month, 
            day=mcp_time.day
        )
        
        # Should be within a few seconds
        time_diff = abs((mcp_time - system_time_only).total_seconds())
        self.assertLessEqual(time_diff, 5, f"MCP time differs from system time by {time_diff} seconds")
    
    def test_current_date_real_date(self):
        """Test that current_date returns the real current date."""
        system_date = datetime.now().strftime("%Y-%m-%d")
        mcp_date = current_date()
        
        self.assertEqual(system_date, mcp_date, f"MCP date {mcp_date} != system date {system_date}")
    
    def test_current_day_real_day(self):
        """Test that current_day_of_week returns the real current day."""
        system_day = datetime.now().strftime("%A")
        mcp_day = current_day_of_week()
        
        self.assertEqual(system_day, mcp_day, f"MCP day {mcp_day} != system day {system_day}")


class TestTimeServerImports(unittest.TestCase):
    """Test that all time functions can be imported."""
    
    def test_imports(self):
        """Test importing all functions."""
        from voice_computer.mcp_servers.time_mcp_server import (
            current_time,
            current_date,
            current_day_of_week,
            current_datetime,
            mcp
        )
        
        # Verify the FastMCP instance exists
        self.assertIsNotNone(mcp)
        self.assertTrue(hasattr(mcp, '_mcp_server'))


class TestTimeFunctionReturnTypes(unittest.TestCase):
    """Test that all time functions return string values."""
    
    def test_return_types(self):
        """Test return types and non-empty values."""
        time_result = current_time()
        date_result = current_date()
        day_result = current_day_of_week()
        datetime_result = current_datetime()
        
        self.assertIsInstance(time_result, str, "current_time should return string")
        self.assertIsInstance(date_result, str, "current_date should return string")
        self.assertIsInstance(day_result, str, "current_day_of_week should return string")
        self.assertIsInstance(datetime_result, str, "current_datetime should return string")
        
        # All should be non-empty
        self.assertGreater(len(time_result), 0, "current_time returned empty string")
        self.assertGreater(len(date_result), 0, "current_date returned empty string")
        self.assertGreater(len(day_result), 0, "current_day_of_week returned empty string")
        self.assertGreater(len(datetime_result), 0, "current_datetime returned empty string")


if __name__ == "__main__":
    unittest.main(verbosity=2)