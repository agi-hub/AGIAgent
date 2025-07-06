#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner for all ToolExecutor tests.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --test <test_name>

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import argparse

# Add parent directory to path to import test modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test functions
from tests.test_tool_calling import test_chat_based_tool_calling, test_json_tool_calling_parsing
from tests.test_statistics import test_llm_statistics
from tests.test_cache import test_cache_mechanisms
from tests.test_history import test_history_summarization
from tests.test_search import test_search_result_formatting

from tools.print_system import print_current


def run_all_tests():
    """Run all available tests."""
    print_current("🧪 Running all ToolExecutor tests...")
    print_current("=" * 60)
    
    tests = [
        ("Tool Calling (Chat-based)", test_chat_based_tool_calling),
        ("Tool Calling (JSON Parsing)", test_json_tool_calling_parsing),
        ("LLM Statistics", test_llm_statistics),
        ("Cache Mechanisms", test_cache_mechanisms),
        ("History Summarization", test_history_summarization),
        ("Search Result Formatting", test_search_result_formatting),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print_current(f"\n📋 Running: {test_name}")
            print_current("-" * 40)
            test_func()
            print_current(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print_current(f"❌ {test_name}: FAILED")
            print_current(f"   Error: {e}")
            failed += 1
    
    print_current("\n" + "=" * 60)
    print_current(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print_current("❌ Some tests failed!")
        return False
    else:
        print_current("✅ All tests passed!")
        return True


def run_specific_test(test_name):
    """Run a specific test by name."""
    test_map = {
        "chat_tools": test_chat_based_tool_calling,
        "json_parsing": test_json_tool_calling_parsing,
        "statistics": test_llm_statistics,
        "cache": test_cache_mechanisms,
        "history": test_history_summarization,
        "search": test_search_result_formatting,
    }
    
    if test_name not in test_map:
        print_current(f"❌ Unknown test: {test_name}")
        print_current(f"Available tests: {', '.join(test_map.keys())}")
        return False
    
    try:
        print_current(f"🧪 Running specific test: {test_name}")
        print_current("=" * 60)
        test_map[test_name]()
        print_current(f"✅ Test {test_name}: PASSED")
        return True
    except Exception as e:
        print_current(f"❌ Test {test_name}: FAILED")
        print_current(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Run ToolExecutor tests')
    parser.add_argument('--test', '-t', help='Run specific test (chat_tools, json_parsing, statistics, cache, history, search)')
    parser.add_argument('--list', '-l', action='store_true', help='List available tests')
    
    args = parser.parse_args()
    
    if args.list:
        print_current("Available tests:")
        tests = [
            ("chat_tools", "Test chat-based tool calling functionality"),
            ("json_parsing", "Test JSON tool calling format parsing"),
            ("statistics", "Test LLM statistics calculation"),
            ("cache", "Test cache mechanisms"),
            ("history", "Test history summarization"),
            ("search", "Test search result formatting"),
        ]
        for test_name, description in tests:
            print_current(f"  {test_name:12} - {description}")
        return
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 