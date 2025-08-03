"""
Tests for the math MCP server.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_computer.mcp_servers.math_mcp_server import (
    add_two_numbers,
    subtract_two_numbers,
    multiply_two_numbers,
    divide_two_numbers,
    square_root
)


class TestMathMCPServer(unittest.TestCase):
    """Test suite for math MCP server functions."""
    
    def test_add_two_numbers(self):
        """Test addition function."""
        self.assertEqual(add_two_numbers(5, 3), 8)
        self.assertEqual(add_two_numbers(-2, 7), 5)
        self.assertEqual(add_two_numbers(0, 0), 0)
        self.assertEqual(add_two_numbers(2.5, 3.7), 6.2)
        self.assertEqual(add_two_numbers(-5, -3), -8)
    
    def test_subtract_two_numbers(self):
        """Test subtraction function."""
        self.assertEqual(subtract_two_numbers(10, 3), 7)
        self.assertEqual(subtract_two_numbers(5, 8), -3)
        self.assertEqual(subtract_two_numbers(0, 5), -5)
        self.assertEqual(subtract_two_numbers(7.5, 2.5), 5.0)
        self.assertEqual(subtract_two_numbers(-3, -8), 5)
    
    def test_multiply_two_numbers(self):
        """Test multiplication function."""
        self.assertEqual(multiply_two_numbers(4, 5), 20)
        self.assertEqual(multiply_two_numbers(-3, 4), -12)
        self.assertEqual(multiply_two_numbers(0, 100), 0)
        self.assertEqual(multiply_two_numbers(2.5, 4), 10.0)
        self.assertEqual(multiply_two_numbers(-2, -6), 12)
    
    def test_divide_two_numbers(self):
        """Test division function."""
        self.assertEqual(divide_two_numbers(10, 2), 5)
        self.assertEqual(divide_two_numbers(15, 3), 5)
        self.assertEqual(divide_two_numbers(7, 2), 3.5)
        self.assertEqual(divide_two_numbers(-10, 2), -5)
        self.assertEqual(divide_two_numbers(-15, -3), 5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            divide_two_numbers(10, 0)
        self.assertIn("Cannot divide by zero", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            divide_two_numbers(-5, 0)
        self.assertIn("Cannot divide by zero", str(cm.exception))
    
    def test_square_root(self):
        """Test square root function."""
        self.assertEqual(square_root(4), 2.0)
        self.assertEqual(square_root(9), 3.0)
        self.assertEqual(square_root(0), 0.0)
        self.assertEqual(square_root(16), 4.0)
        self.assertLess(abs(square_root(2) - 1.4142135623730951), 1e-10)
        self.assertEqual(square_root(0.25), 0.5)
    
    def test_square_root_negative(self):
        """Test square root of negative number raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            square_root(-4)
        self.assertIn("Cannot calculate square root of negative number", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            square_root(-1)
        self.assertIn("Cannot calculate square root of negative number", str(cm.exception))
    
    def test_type_handling(self):
        """Test that functions handle both int and float types."""
        # Test mixing int and float
        self.assertEqual(add_two_numbers(5, 3.5), 8.5)
        self.assertEqual(subtract_two_numbers(10.0, 3), 7.0)
        self.assertEqual(multiply_two_numbers(2, 3.5), 7.0)
        self.assertEqual(divide_two_numbers(7.5, 2), 3.75)
        self.assertEqual(square_root(9.0), 3.0)


class TestMathServerImports(unittest.TestCase):
    """Test that all math functions can be imported."""
    
    def test_imports(self):
        """Test importing all functions."""
        from voice_computer.mcp_servers.math_mcp_server import (
            add_two_numbers,
            subtract_two_numbers, 
            multiply_two_numbers,
            divide_two_numbers,
            square_root,
            mcp
        )
        
        # Verify the FastMCP instance exists
        self.assertIsNotNone(mcp)
        self.assertTrue(hasattr(mcp, '_mcp_server'))


if __name__ == "__main__":
    unittest.main(verbosity=2)