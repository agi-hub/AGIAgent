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


def call_openai_with_chat_based_tools_non_streaming(executor, messages, system_message):
    """
    Call OpenAI API with chat-based tool calling in non-streaming mode.
    Tools should be described in the prompt and tool call responses are parsed from the content.

    Args:
        executor: ToolExecutor instance
        messages: Complete message history for the LLM (including user messages and history)
        system_message: The system message.

    Returns:
        Tuple of (content, tool_calls)
    """
    # Use the OpenAI API for chat-based tool calling
    api_messages = [
        {"role": "system", "content": system_message}
    ]
    api_messages.extend(messages)  # messages already contains all user messages and history

    response = executor.client.chat.completions.create(
        model=executor.model,
        messages=api_messages,
        max_tokens=executor._get_max_tokens_for_model(executor.model),
        temperature=executor.temperature,
        top_p=executor.top_p,
        stream=False
    )

    # Print token usage in non-streaming mode
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        total_tokens = getattr(usage, 'total_tokens', 0) or 0
        print_debug(f"📊 Current conversation token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")

    # Regular OpenAI flow
    message = response.choices[0].message
    content = message.content or ""

    # For OpenAI o1 models and other reasoning models: handle 'thinking' content
    if executor.enable_thinking:
        thinking = getattr(message, 'thinking', None)
        if thinking:
            content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

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

