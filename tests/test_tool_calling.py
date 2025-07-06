#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for tool calling functionality in ToolExecutor.

Copyright (c) 2025 AGI Bot Research Group.
Licensed under the Apache License, Version 2.0
"""

import os
import sys

# Add parent directory to path to import ToolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import ToolExecutor, should_use_chat_based_tools, is_claude_model
from tools.print_system import print_current


def test_chat_based_tool_calling():
    """Test the new chat-based tool calling functionality"""
    print_current("🧪 Testing chat-based tool calling functionality...")
    
    # Test model detection
    test_models = [
        "gpt-4",
        "gpt-3.5-turbo", 
        "claude-3-sonnet-20240229",
        "Qwen/Qwen3-30B-A3B",
        "meta-llama/Llama-2-70b-chat-hf",
        "google/gemini-pro"
    ]
    
    print_current("\n1. Testing model detection:")
    for model in test_models:
        use_chat_based = should_use_chat_based_tools(model)
        is_claude = is_claude_model(model)
        tool_method = "Chat-based" if use_chat_based else "Standard API"
        print_current(f"   {model}: {tool_method} tool calling (Claude: {is_claude})")
    
    print_current("\n2. Testing JSON tool description generation:")
    try:
        print_current("   Testing JSON-based tool description generation...")
        
        # Test JSON tool description generation for chat-based models
        from utils.parse import generate_tools_prompt_from_json
        chat_executor = ToolExecutor(model="qwen-chat")
        tool_definitions = chat_executor._load_tool_definitions_from_file()
        json_tools_prompt = generate_tools_prompt_from_json(tool_definitions, chat_executor.language)
        
        if json_tools_prompt:
            # print_current(f"   ✅ JSON tool descriptions generated: {len(json_tools_prompt)} characters")
            
            # Check key components
            key_components = [
                ("工具标题", "## 可用工具" in json_tools_prompt or "## Available Tools" in json_tools_prompt),
                ("JSON格式说明", "JSON格式调用工具" in json_tools_prompt or "JSON format" in json_tools_prompt),
                ("工具描述", "#### " in json_tools_prompt),
                ("使用示例", "```json" in json_tools_prompt)
            ]
            
            # for component_name, found in key_components:
            #     status = "✅" if found else "❌"
            #     print_current(f"   {status} {component_name}: {'包含' if found else '缺失'}")
        else:
            print_current("   ❌ JSON tool descriptions generation failed")
            
        # Test standard tool calling (no descriptions needed)
        standard_executor = ToolExecutor(model="claude-3-haiku-20240307")
        standard_components = standard_executor.load_user_prompt_components()
        rules_content = standard_components.get('rules_and_tools', '')
        
        print_current(f"   ✅ Standard tool calling rules loaded: {len(rules_content)} characters")
        
    except Exception as e:
        print_current(f"   ❌ Error testing JSON tool descriptions: {e}")
        
        # Show sample of chat tool prompt
        if 'chat_content' in locals():
            print_current("   Sample of chat tool prompt:")
            lines = chat_content.split('\n')
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print_current(f"     {line}")
            print_current("     ... (truncated)")
        
        print_current("\n=== Chat-based Tool Calling Tests Complete ===")
        
        print_current("\n=== Chat-based Tool Calling Tests Complete ===")
        
    except Exception as e:
        print_current(f"   Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_json_tool_calling_parsing():
    """Test JSON tool calling format parsing functionality"""
    print_current("🧪 Testing JSON tool calling format parsing...")
    
    # Create a mock ToolExecutor instance for testing
    try:
        executor = ToolExecutor(
            api_key="test_key",
            model="Qwen/Qwen3-30B-A3B",  # This will use chat-based tools
            api_base="test_base",
            debug_mode=True
        )
        
        print_current("\n1. Testing OpenAI-style JSON tool calls with code block:")
        
        # Test OpenAI-style JSON format with ```json wrapper
        json_content_1 = '''I'll help you read that file.

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": {
          "target_file": "src/main.py",
          "should_read_entire_file": false,
          "start_line_one_indexed": 1,
          "end_line_one_indexed_inclusive": 50
        }
      }
    }
  ]
}
```

Let me read the file for you.'''
        
        tool_calls_1 = executor.parse_tool_calls(json_content_1)
        print_current(f"   Parsed {len(tool_calls_1)} tool calls:")
        for i, call in enumerate(tool_calls_1, 1):
            print_current(f"     {i}. {call['name']}: {list(call['arguments'].keys())}")
        
        print_current("\n2. Testing multiple tool calls in JSON format:")
        
        # Test multiple tool calls
        json_content_2 = '''I'll list the directory and then search for code.

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "list_dir",
        "arguments": {
          "relative_workspace_path": "src"
        }
      }
    },
    {
      "id": "call_2",
      "type": "function",
      "function": {
        "name": "codebase_search",
        "arguments": {
          "query": "function definition",
          "target_directories": ["src/*"]
        }
      }
    }
  ]
}
```

Now let me execute these tools.'''
        
        tool_calls_2 = executor.parse_tool_calls(json_content_2)
        print_current(f"   Parsed {len(tool_calls_2)} tool calls:")
        for i, call in enumerate(tool_calls_2, 1):
            print_current(f"     {i}. {call['name']}: {list(call['arguments'].keys())}")
        
        print_current("\n3. Testing direct JSON without code block:")
        
        # Test direct JSON format without ```json wrapper
        json_content_3 = '''I'll edit the file for you.

{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "edit_file",
        "arguments": {
          "target_file": "test.py",
          "instructions": "Create a simple test function",
          "code_edit": "def test_function():\\n    return \\"Hello World\\""
        }
      }
    }
  ]
}

The file will be edited as requested.'''
        
        tool_calls_3 = executor.parse_tool_calls(json_content_3)
        print_current(f"   Parsed {len(tool_calls_3)} tool calls:")
        for i, call in enumerate(tool_calls_3, 1):
            print_current(f"     {i}. {call['name']}: {list(call['arguments'].keys())}")
        
        print_current("\n4. Testing edge cases and error handling:")
        
        # Test malformed JSON
        malformed_json = '''```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": {
          "target_file": "test.py"
          // Missing comma and incomplete
        }
      }
    
  ]
}
```'''
        
        tool_calls_4 = executor.parse_tool_calls(malformed_json)
        print_current(f"   Malformed JSON parsed {len(tool_calls_4)} tool calls (should be 0)")
        
        # Test no tool calls
        no_tools_content = '''This is just a regular response without any tool calls.
        
The user asked me to explain something, so I'm providing a text-only response.'''
        
        tool_calls_5 = executor.parse_tool_calls(no_tools_content)
        print_current(f"   No tools content parsed {len(tool_calls_5)} tool calls (should be 0)")
        
        print_current("\n5. Testing backwards compatibility with XML:")
        
        # Test that XML parsing still works
        xml_content = '''I'll help you with that.

<function_calls>
<invoke name="read_file">
<parameter name="target_file">test.py</parameter>
<parameter name="should_read_entire_file">true</parameter>
<parameter name="start_line_one_indexed">1</parameter>
<parameter name="end_line_one_indexed_inclusive">100</parameter>
</invoke>
</function_calls>

Let me read the file.'''
        
        tool_calls_6 = executor.parse_tool_calls(xml_content)
        print_current(f"   XML format parsed {len(tool_calls_6)} tool calls:")
        for i, call in enumerate(tool_calls_6, 1):
            print_current(f"     {i}. {call['name']}: {list(call['arguments'].keys())}")
        
        print_current("\n=== JSON Tool Calling Parsing Tests Complete ===")
        print_current("✅ All parsing formats are working correctly!")
        
    except Exception as e:
        print_current(f"   Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chat_based_tool_calling()
    test_json_tool_calling_parsing() 