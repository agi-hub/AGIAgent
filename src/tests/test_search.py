#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for search result formatting functionality in ToolExecutor.

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys

# Add parent directory to path to import ToolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import ToolExecutor
from tools.print_system import print_current


def test_search_result_formatting():
    """Test search result formatting functionality"""
    print_current("🧪 Testing search result formatting...")
    
    # Test codebase search result
    codebase_result = {
        'results': [
            {
                'file_path': 'test.py',
                'line_number': 10,
                'content': 'def example_function():',
                'score': 0.95
            }
        ]
    }
    
    executor = ToolExecutor()
    formatted = executor._format_search_result_for_terminal(codebase_result, 'codebase_search')
    print_current("Codebase search result formatting:")
    print(formatted)
    print()
    
    # Test web search result
    web_result = {
        'results': [
            {
                'title': 'Example Title',
                'url': 'https://example.com',
                'snippet': 'This is an example snippet',
                'score': 0.9
            }
        ]
    }
    
    formatted = executor._format_search_result_for_terminal(web_result, 'web_search')
    print_current("Web search result formatting:")
    print(formatted)


if __name__ == "__main__":
    test_search_result_formatting() 