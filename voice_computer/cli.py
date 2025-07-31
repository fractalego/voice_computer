"""
Command-line interface entry point for voice computer.
"""

import sys
import os

# Add the parent directory to the path to ensure we can import run_voice_computer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_voice_computer import cli_main


def main():
    """Entry point for the voice-computer command."""
    cli_main()


if __name__ == "__main__":
    main()