#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for history functionality in ToolExecutor.

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys

# Add parent directory to path to import ToolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import ToolExecutor
from tools.print_system import print_current


def test_history_summarization():
    """Test the improved history summarization functionality"""
    print_current("🧪 Testing improved history summarization functionality...")
    
    # Create a mock ToolExecutor instance for testing
    try:
        executor = ToolExecutor(
            api_key="test_key",
            model="test_model", 
            api_base="test_base",
            debug_mode=True
        )
        
        # Create large test history
        large_history = []
        for i in range(10):
            large_history.append({
                "prompt": f"Test task {i}: " + "x" * 10000,  # 10k chars each
                "result": f"Result for task {i}: " + "y" * 10000  # 10k chars each
            })
        
        print_current(f"\n1. Testing with large history ({len(large_history)} records)")
        
        # Calculate total length
        total_length = sum(len(str(record.get("content", ""))) + len(str(record.get("result", ""))) + len(str(record.get("prompt", ""))) for record in large_history)
        print_current(f"   Total history length: {total_length:,} characters")
        
        # Test history hash computation
        hash1 = executor._compute_history_hash(large_history)
        hash2 = executor._compute_history_hash(large_history)  # Should be the same
        print_current(f"   History hash consistency: {hash1 == hash2}")
        print_current(f"   History hash: {hash1[:16]}...")
        
        # Test recent history subset
        recent_subset = executor._get_recent_history_subset(large_history, max_length=50000)
        recent_length = sum(len(str(record.get("content", ""))) + len(str(record.get("result", ""))) + len(str(record.get("prompt", ""))) for record in recent_subset)
        print_current(f"   Recent subset: {len(recent_subset)} records, {recent_length:,} characters")
        
        # Test cache info
        cache_info = executor.get_history_summary_cache_info()
        print_current(f"   Cache info: {cache_info}")
        
        # Test cache clearing
        executor.clear_history_summary_cache()
        
        print_current("\n2. Testing cache functionality")
        
        # Simulate adding items to cache
        executor.history_summary_cache = {"test_hash": "test_summary"}
        executor.last_summarized_history_length = 100000
        
        cache_info_after = executor.get_history_summary_cache_info()
        print_current(f"   Cache info after adding item: {cache_info_after}")
        
        # Test cache clearing again
        executor.clear_history_summary_cache()
        cache_info_cleared = executor.get_history_summary_cache_info()
        print_current(f"   Cache info after clearing: {cache_info_cleared}")
        
        print_current("\n=== History Summarization Tests Complete ===")
        
    except Exception as e:
        print_current(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_history_summarization() 