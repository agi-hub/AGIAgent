#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for LLM statistics functionality in ToolExecutor.

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys

# Add parent directory to path to import ToolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import ToolExecutor
from tools.print_system import print_current
from utils.cacheeff import estimate_token_count, analyze_cache_potential


def test_llm_statistics():
    """Test LLM statistics calculation functionality"""
    print_current("🧪 Testing LLM statistics calculation...")
    
    executor = ToolExecutor()
    
    # Test messages
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请帮我写一个Python函数来计算斐波那契数列。"},
        {"role": "assistant", "content": "当然！我来帮你写一个计算斐波那契数列的Python函数。"}
    ]
    
    # Test response content
    test_response = """这是一个计算斐波那契数列的函数：

```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# 测试函数
for i in range(10):
    print_current(f"fibonacci({i}) = {fibonacci(i)}")
```

这个函数使用递归方法计算斐波那契数列。"""
    
    print_current("Testing statistics display:")
    executor._display_llm_statistics(test_messages, test_response)
    
    print_current("\nTesting token estimation for different text types:")
    
    # Test English text
    english_text = "This is a sample English text for testing token estimation."
    english_tokens = estimate_token_count(english_text)
    print_current(f"English text ({len(english_text)} chars): {english_tokens} tokens")
    
    # Test Chinese text
    chinese_text = "这是一段用于测试token估算的中文文本。"
    chinese_tokens = estimate_token_count(chinese_text)
    print_current(f"Chinese text ({len(chinese_text)} chars): {chinese_tokens} tokens")
    
    # Test code text
    code_text = """def example_function():
    return {"key": "value"}"""
    code_tokens = estimate_token_count(code_text)
    print_current(f"Code text ({len(code_text)} chars): {code_tokens} tokens")
    
    print_current("\nTesting cache analysis:")
    
    # Test messages with history
    cache_test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """Current request: Write a function

============================================================
Previous execution history:
============================================================

Round 1:
User: Create a class
Assistant: I'll create a class for you.

Tool execution results:
## Tool 1: edit_file
**Parameters:** target_file=test.py
**Result:**
✅ Completed
- content: class Example: pass

============================================================
Current request: Write a function"""}
    ]
    
    cache_stats = analyze_cache_potential(cache_test_messages)
    print_current(f"Cache analysis results:")
    print_current(f"  Has history: {cache_stats['has_history']}")
    print_current(f"  Total tokens: {cache_stats['total_tokens']}")
    print_current(f"  History tokens: {cache_stats['history_tokens']}")
    print_current(f"  New tokens: {cache_stats['new_tokens']}")
    print_current(f"  Cache hit potential: {cache_stats['cache_hit_potential']:.1%}")
    print_current(f"  Estimated cache tokens: {cache_stats['estimated_cache_tokens']}")


if __name__ == "__main__":
    test_llm_statistics() 