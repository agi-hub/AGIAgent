#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Agent Research Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from src.tools.print_system import print_current, print_debug


def call_claude_with_chat_based_tools_non_streaming(executor, messages, system_message):
    """
    Call Claude API with chat-based tool calling in non-streaming mode.
    Tools should be described in the prompt and tool call responses are parsed from the content.

    Args:
        executor: ToolExecutor instance
        messages: Complete message history for the LLM (including user messages and history)
        system_message: The system message.

    Returns:
        Tuple of (content, tool_calls)
    """
    # Use the Anthropic Claude API for chat-based tool calling
    claude_messages = messages  # messages already contains all user messages and history
    # When thinking is enabled, temperature must be 1.0
    temperature = 1.0 if executor.enable_thinking else executor.temperature

    api_params = {
        "model": executor.model,
        "max_tokens": executor._get_max_tokens_for_model(executor.model),
        "system": system_message,
        "messages": claude_messages,
        "temperature": temperature
    }

    # Enable 'thinking' mode for reasoning-capable models
    if executor.enable_thinking:
        api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    response = executor.client.messages.create(**api_params)

    # Print token usage in non-streaming mode
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
        
        if cache_creation_tokens > 0 or cache_read_tokens > 0:
            print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
        else:
            print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}")

    content = ""
    thinking = ""

    # Extract 'thinking' and normal content from the Anthropic response
    if executor.enable_thinking:
        # Try 'thinking' property first
        thinking = getattr(response, 'thinking', None) or ""
        # Also collect from content blocks if present
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == "thinking":
                if hasattr(content_block, 'text'):
                    thinking += content_block.text
                elif hasattr(content_block, 'thinking'):
                    thinking += content_block.thinking

    # Concatenate regular text blocks
    for content_block in response.content:
        if hasattr(content_block, 'type') and content_block.type == "text":
            content += content_block.text

    if executor.enable_thinking and thinking:
        content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

    # Check for hallucination patterns (strict, not partial matches)
    hallucination_patterns = [
        "**LLM Called Following Tools in this round",
        "**Tool Execution Results:**"
    ]
    hallucination_detected = any(pattern in content for pattern in hallucination_patterns)
    if hallucination_detected:
        # Add error feedback to history
        executor._add_error_feedback_to_history(
            error_type='hallucination_detected',
            error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
        )
        # Ensure trailing newline (chat interface)
        if not content.endswith('\n'):
            content += '\n'
        return content, []

    # Parse tool calls from the (only) relevant content
    tool_calls = executor.parse_tool_calls(content)

    # Ensure output ends with a newline (chat interface)
    if not content.endswith('\n'):
        content += '\n'
    
    # Print LLM response in non-streaming mode
    if content:
        print_current("")
        print_current("💬 LLM Response:")
        print_current(content)
    
    return content, tool_calls

