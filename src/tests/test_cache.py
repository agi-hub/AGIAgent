#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for cache functionality in ToolExecutor.

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys

# Add parent directory to path to import ToolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import ToolExecutor
from tools.print_system import print_current


def test_cache_mechanisms():
    """Test the enhanced cache mechanisms"""
    print_current("=== Testing Cache-Friendly Text Formatting ===")
    
    # Create a mock ToolExecutor instance for testing
    try:
        executor = ToolExecutor(
            api_key="test_key",
            model="test_model", 
            api_base="test_base",
            debug_mode=True
        )
        
        print_current("\n1. Testing tool results standardization:")
        
        raw_tool_results = """<tool_execute tool_name="read_file" tool_number="1">
   - Executing tool 1: read_file
Executing tool: read_file with params: ['target_file', 'start_line_one_indexed']
File content here...
</tool_execute>

<tool_execute tool_name="edit_file" tool_number="2">
   - Executing tool 2: edit_file  
Executing tool: edit_file with params: ['target_file', 'instructions']
Edit completed successfully
</tool_execute>"""
        
        standardized = executor._standardize_tool_results_format(raw_tool_results)
        print_current("Standardized tool results:")
        print(standardized)
        
        print_current("\n2. Testing history formatting:")
        
        # Test history formatting with standardized separators
        test_history = [
            {
                "prompt": "Create a Python function",
                "result": "I'll create a Python function for you.\n\n--- Tool Execution Results ---\nFunction created successfully."
            }
        ]
        
        message_parts = []
        executor._add_full_history_to_message(message_parts, test_history)
        
        print_current("Formatted history:")
        print('\n'.join(message_parts))
        
        print_current("\n=== Cache-Friendly Formatting Tests Complete ===")
        
    except Exception as e:
        print_current(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cache_mechanisms() 