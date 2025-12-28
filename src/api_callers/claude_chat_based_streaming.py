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

from email import message
import re
from src.tools.print_system import streaming_context, print_debug, print_current
from src.utils.parse import fix_wrong_tool_call_format


def _ensure_newline_before_invoke(content: str) -> str:
    """
    确保在消息文本和工具调用标签之间添加换行符。
    
    检测模式：非XML标签的文本内容后面紧跟着 <invoke name=
    在它们之间插入换行符，使输出更清晰。
    
    例如：
    "💬 现在开始撰写报告。<invoke name="edit_file">"
    会被转换为：
    "💬 现在开始撰写报告。\n<invoke name="edit_file">"
    
    Args:
        content: 原始内容字符串
        
    Returns:
        处理后的内容字符串
    """
    if not content:
        return content
    
    # 使用正则表达式匹配：在同一行中，非空白字符、非XML标签字符后面紧跟着 <invoke name=
    # 模式：匹配文本内容（非标签，非空白）后面直接跟着 <invoke name=，且它们在同一行
    
    # 更精确的实现：逐行处理，对于包含 <invoke name= 的行，检查前面是否有文本
    # 但要注意：流式输出中，内容可能跨行，所以需要全局处理
    
    # 使用正则表达式匹配：文本内容（不在标签内）后面紧跟着 <invoke name=
    # 匹配模式：
    # 1. 非标签字符序列（至少一个非空白字符）
    # 2. 后面直接跟着（可选空白 + <invoke name=）
    # 3. 且它们之间没有换行符（即在同一行）
    
    # 匹配非标签文本后紧跟着 <invoke name= 的情况
    # 使用负向前瞻和后顾断言确保不在标签内
    # 模式：(?<!>) - 前面不是 >
    #      ([^\s<>]+(?:\s+[^\s<>]+)*) - 文本内容（至少一个非空白字符）
    #      \s* - 可选空白
    #      (<invoke\s+name=) - <invoke name=
    
    # 但上面的模式在流式输出中可能不够准确，使用更简单的方法：
    # 查找所有 <invoke name= 的位置，检查前面是否需要换行
    
    lines = content.split('\n')
    result_lines = []
    
    for line_idx, line in enumerate(lines):
        if '<invoke name=' in line:
            # 在这一行中找到 <invoke name= 的位置
            invoke_pos = line.find('<invoke name=')
            if invoke_pos > 0:
                # 检查 <invoke 前面的内容
                before_invoke = line[:invoke_pos].rstrip()
                # 如果有文本内容且不是标签的一部分，需要分割
                if before_invoke and not before_invoke.endswith('>') and not before_invoke.endswith('</invoke'):
                    # 在同一行，需要分成两行
                    result_lines.append(before_invoke)
                    result_lines.append(line[invoke_pos:])
                    continue
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def call_claude_with_chat_based_tools_streaming(executor, messages, system_message):
    """
    Call Claude API with chat-based tool calling in streaming mode.
    Tools are described in the message and responses are parsed from content.
    
    Args:
        executor: ToolExecutor instance
        messages: Message history for the LLM
        user_message: Current user message
        system_message: System message
        
    Returns:
        Tuple of (content, tool_calls)
    """

    with streaming_context(show_start_message=False) as printer:
            # Prepare parameters for Anthropic API
            # Note: When thinking is enabled, temperature MUST be 1.0
            temperature = 1.0 if executor.enable_thinking else executor.temperature
            
            api_params = {
                "model": executor.model,
                "max_tokens": executor._get_max_tokens_for_model(executor.model),
                "system": system_message,
                "messages": messages,
                "temperature": temperature
            }

            # Enable thinking for reasoning-capable models
            if executor.enable_thinking:
                api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            # Try to create stream with thinking parameter, fallback if not supported
            # Note: messages.stream() returns a context manager, errors occur when entering the context
            try:
                with executor.client.messages.stream(**api_params) as stream:
                    content = ""
                    hallucination_detected = False
                    stream_error_message = ""
                    
                    # Buffer printing mechanism: keep at least 100 characters not printed, used to check hallucinations/multiple json blocks
                    min_buffer_size = 100
                    total_printed = 0
                    
                    # Whether to early-stop stream due to detecting the first complete tool call
                    tool_call_detected_early = False
                    
                    # Thinking tracking
                    thinking_printed_header = False
                    
                    try:
                        # Use event-based streaming to capture "thinking"
                        for event in stream:
                            event_type = getattr(event, 'type', None)
                            
                            # Handle start of new content block (e.g. "thinking" or "text")
                            if event_type == "content_block_start":
                                content_block = getattr(event, 'content_block', None)
                                if content_block:
                                    block_type = getattr(content_block, 'type', None)
                                    if block_type == "thinking" and executor.enable_thinking:
                                        printer.write("\n🧠 ")
                                        thinking_printed_header = True
                                    elif block_type == "text":
                                        if thinking_printed_header:
                                            printer.write("\n")
                                        printer.write("\n💬 ")
                            
                            # Handle message_delta event (token stats)
                            elif event_type == "message_delta":
                                try:
                                    delta = getattr(event, 'delta', None)
                                    if delta:
                                        usage = getattr(delta, 'usage', None) or getattr(event, 'usage', None)
                                        if usage:
                                            input_tokens = getattr(usage, 'input_tokens', 0) or 0
                                            output_tokens = getattr(usage, 'output_tokens', 0) or 0
                                            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                                            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
                                            
                                            if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                                print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                            else:
                                                print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}")
                                except Exception as e:
                                    print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")
                            
                            # Handle new content in current content block
                            elif event_type == "content_block_delta":
                                delta = getattr(event, 'delta', None)
                                if delta:
                                    delta_type = getattr(delta, 'type', None)
                                    
                                    # "thinking" content (thoughts, reasoning, etc)
                                    if delta_type == "thinking_delta" and executor.enable_thinking:
                                        thinking_text = getattr(delta, 'thinking', '')
                                        printer.write(thinking_text)
                                    
                                    # "text" content (main reply)
                                    elif delta_type == "text_delta":
                                        text = getattr(delta, 'text', '')
                                        content += text
                                        
                                        # Fix wrong tool call format before printing
                                        # This ensures <tool_call>tool_name> is converted to <invoke name="tool_name">
                                        # before it's printed to terminal, using the buffer mechanism
                                        content = fix_wrong_tool_call_format(content)
                                        
                                        # Check for hallucination patterns
                                        hallucination_patterns = [
                                            "**LLM Called Following Tools in this round",
                                            "**Tool Execution Results:**"
                                        ]
                                        
                                        unprinted_content = content[total_printed:]
                                        hallucination_detected_flag = False
                                        hallucination_start_in_unprinted = -1
                                        
                                        for pattern in hallucination_patterns:
                                            if pattern in unprinted_content:
                                                hallucination_start_in_unprinted = unprinted_content.find(pattern)
                                                hallucination_detected_flag = True
                                                break
                                        
                                        if hallucination_detected_flag:
                                            print_debug("\nHallucination Detected, stop chat")
                                            hallucination_detected = True
                                            # Truncate content before hallucination
                                            content = content[:total_printed + hallucination_start_in_unprinted].rstrip()
                                            break  # Break out of for event in stream loop
                                        
                                        # Check for multiple tool calls - only keep output up to first valid JSON tool call
                                        first_json_pos = content.find('```json')
                                        if first_json_pos != -1:
                                            second_json_pos = content.find('```json', first_json_pos + len('```json'))
                                            if second_json_pos != -1:
                                                content_before_second = content[:second_json_pos]
                                                open_braces = content_before_second.count('{')
                                                close_braces = content_before_second.count('}')
                                                
                                                if open_braces == close_braces:
                                                    print_debug("\n🛑 Multiple tool calls detected, stopping stream after first JSON block")
                                                    tool_call_detected_early = True
                                                    content = content_before_second.rstrip()
                                                    break  # Break out of for event in stream loop
                                        
                                        # Check for multiple XML tool calls - detect complete <invoke> tags
                                        # Find all complete <invoke>...</invoke> tags
                                        invoke_pattern = r'<invoke name="[^"]+">.*?</invoke>'
                                        invoke_matches = list(re.finditer(invoke_pattern, content, re.DOTALL))
                                        
                                        if len(invoke_matches) > 0:
                                            # At least one complete invoke tag found
                                            first_invoke_end = invoke_matches[0].end()
                                            remaining_content = content[first_invoke_end:].strip()
                                            
                                            # Check if there's another <invoke> tag starting after the first one
                                            if '<invoke name=' in remaining_content:
                                                # Check if the second invoke is complete
                                                second_invoke_match = re.search(r'<invoke name="[^"]+">.*?</invoke>', remaining_content, re.DOTALL)
                                                if second_invoke_match:
                                                    # Second invoke is also complete - allow multiple complete tool calls
                                                    # Continue streaming to receive all complete tool calls
                                                    print_debug("\n✅ Multiple complete XML tool calls detected, continuing to receive all")
                                                else:
                                                    # Second invoke is incomplete (streaming), but we have a complete first one
                                                    # Wait a bit to see if it completes, but don't stop immediately
                                                    # Only stop if we've been waiting too long (handled by buffer mechanism)
                                                    pass
                                        
                                        # If nothing forced an early exit, print except for trailing buffer
                                        unprinted_length = len(content) - total_printed
                                        if unprinted_length >= min_buffer_size:
                                            print_length = unprinted_length - min_buffer_size
                                            if print_length > 0:
                                                # 在打印前，确保在消息文本和工具调用标签之间添加换行
                                                # 需要对整个content进行处理，然后基于处理后的内容计算打印位置
                                                content_processed = _ensure_newline_before_invoke(content)
                                                
                                                # 如果处理后的内容长度发生变化，需要重新计算打印位置
                                                if len(content_processed) != len(content):
                                                    # 内容被修改了，更新content并重新计算
                                                    content = content_processed
                                                    unprinted_length = len(content) - total_printed
                                                    if unprinted_length >= min_buffer_size:
                                                        print_length = unprinted_length - min_buffer_size
                                                        if print_length > 0:
                                                            printer.write(content[total_printed:total_printed + print_length])
                                                            total_printed += print_length
                                                else:
                                                    # 内容没有变化，正常打印
                                                    printer.write(content[total_printed:total_printed + print_length])
                                                    total_printed += print_length
                    except Exception as e:
                        # Catch exceptions during streaming
                        stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                        print_debug(f"⚠️ {stream_error_message}")
                        print_current(f"⚠️ Claude API streaming error: {str(e)}")
                        # Continue with any content received so far
                    finally:
                        # Ensure the stream is closed properly, whether normally or early
                        try:
                            if hasattr(stream, 'close'):
                                stream.close()
                            if tool_call_detected_early:
                                print_debug("🔌 Stream closed early due to multiple tool calls detection")
                            if hallucination_detected:
                                print_debug("🔌 Stream closed early due to hallucination detection")
                        except Exception as close_error:
                            print_debug(f"⚠️ Error closing Anthropic stream: {close_error}")
                    
                    # Print remaining content not yet printed
                    if total_printed < len(content):
                        # 在打印前，确保在消息文本和工具调用标签之间添加换行
                        content = _ensure_newline_before_invoke(content)
                        printer.write(content[total_printed:])

                    # If a hallucination was detected, add error feedback but still check for tool calls
                    if hallucination_detected:
                        executor._add_error_feedback_to_history(
                            error_type='hallucination_detected',
                            error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
                        )
                    
                    # Parse tool calls from the output content
                    # parse_tool_calls now returns standardized format with "input" field
                    standardized_tool_calls = executor.parse_tool_calls(content)
                    
                    # Ensure output ends with a newline
                    if not content.endswith('\n'):
                        content += '\n'
                    return content, standardized_tool_calls
            except (TypeError, ValueError) as e:
                # If thinking parameter is not supported, retry without it
                if executor.enable_thinking and ("thinking" in str(e).lower() or "unexpected keyword" in str(e).lower()):
                    print_debug(f"⚠️ Thinking parameter not supported by this API, disabling thinking mode: {e}")
                    # Remove thinking parameter and retry
                    api_params.pop("thinking", None)
                    # Retry without thinking parameter
                    with executor.client.messages.stream(**api_params) as stream:
                        content = ""
                        hallucination_detected = False
                        min_buffer_size = 100
                        total_printed = 0
                        tool_call_detected_early = False
                        
                        try:
                            for event in stream:
                                event_type = getattr(event, 'type', None)
                                
                                if event_type == "content_block_start":
                                    content_block = getattr(event, 'content_block', None)
                                    if content_block:
                                        block_type = getattr(content_block, 'type', None)
                                        if block_type == "text":
                                            printer.write("\n💬 ")
                                
                                elif event_type == "message_delta":
                                    try:
                                        delta = getattr(event, 'delta', None)
                                        if delta:
                                            usage = getattr(delta, 'usage', None) or getattr(event, 'usage', None)
                                            if usage:
                                                input_tokens = getattr(usage, 'input_tokens', 0) or 0
                                                output_tokens = getattr(usage, 'output_tokens', 0) or 0
                                                cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                                                cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
                                                
                                                if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                                    print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                                else:
                                                    print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}")
                                    except Exception as e:
                                        print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")
                                
                                elif event_type == "content_block_delta":
                                    delta = getattr(event, 'delta', None)
                                    if delta:
                                        delta_type = getattr(delta, 'type', None)
                                        
                                        if delta_type == "text_delta":
                                            text = getattr(delta, 'text', '')
                                            content += text
                                            
                                            # Fix wrong tool call format before printing
                                            content = fix_wrong_tool_call_format(content)
                                            
                                            hallucination_patterns = [
                                                "**LLM Called Following Tools in this round",
                                                "**Tool Execution Results:**"
                                            ]
                                            
                                            unprinted_content = content[total_printed:]
                                            hallucination_detected_flag = False
                                            hallucination_start_in_unprinted = -1
                                            
                                            for pattern in hallucination_patterns:
                                                if pattern in unprinted_content:
                                                    hallucination_start_in_unprinted = unprinted_content.find(pattern)
                                                    hallucination_detected_flag = True
                                                    break
                                            
                                            if hallucination_detected_flag:
                                                print_debug("\nHallucination Detected, stop chat")
                                                hallucination_detected = True
                                                content = content[:total_printed + hallucination_start_in_unprinted].rstrip()
                                                break
                                            
                                            first_json_pos = content.find('```json')
                                            if first_json_pos != -1:
                                                second_json_pos = content.find('```json', first_json_pos + len('```json'))
                                                if second_json_pos != -1:
                                                    content_before_second = content[:second_json_pos]
                                                    open_braces = content_before_second.count('{')
                                                    close_braces = content_before_second.count('}')
                                                    
                                                    if open_braces == close_braces:
                                                        print_debug("\n🛑 Multiple tool calls detected, stopping stream after first JSON block")
                                                        tool_call_detected_early = True
                                                        content = content_before_second.rstrip()
                                                        break
                                            
                                            invoke_pattern = r'<invoke name="[^"]+">.*?</invoke>'
                                            invoke_matches = list(re.finditer(invoke_pattern, content, re.DOTALL))
                                            
                                            if len(invoke_matches) > 0:
                                                first_invoke_end = invoke_matches[0].end()
                                                remaining_content = content[first_invoke_end:].strip()
                                                
                                                if '<invoke name=' in remaining_content:
                                                    second_invoke_match = re.search(r'<invoke name="[^"]+">.*?</invoke>', remaining_content, re.DOTALL)
                                                    if second_invoke_match:
                                                        print_debug("\n✅ Multiple complete XML tool calls detected, continuing to receive all")
                                            
                                            unprinted_length = len(content) - total_printed
                                            if unprinted_length >= min_buffer_size:
                                                print_length = unprinted_length - min_buffer_size
                                                if print_length > 0:
                                                    # 在打印前，确保在消息文本和工具调用标签之间添加换行
                                                    content_processed = _ensure_newline_before_invoke(content)
                                                    
                                                    # 如果处理后的内容长度发生变化，需要重新计算打印位置
                                                    if len(content_processed) != len(content):
                                                        content = content_processed
                                                        unprinted_length = len(content) - total_printed
                                                        if unprinted_length >= min_buffer_size:
                                                            print_length = unprinted_length - min_buffer_size
                                                            if print_length > 0:
                                                                printer.write(content[total_printed:total_printed + print_length])
                                                                total_printed += print_length
                                                    else:
                                                        printer.write(content[total_printed:total_printed + print_length])
                                                        total_printed += print_length
                        except Exception as e:
                            stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                            print_debug(f"⚠️ {stream_error_message}")
                            print_current(f"⚠️ Claude API streaming error: {str(e)}")
                        finally:
                            try:
                                if hasattr(stream, 'close'):
                                    stream.close()
                                if tool_call_detected_early:
                                    print_debug("🔌 Stream closed early due to multiple tool calls detection")
                                if hallucination_detected:
                                    print_debug("🔌 Stream closed early due to hallucination detection")
                            except Exception as close_error:
                                print_debug(f"⚠️ Error closing Anthropic stream: {close_error}")
                        
                        if total_printed < len(content):
                            # 在打印前，确保在消息文本和工具调用标签之间添加换行
                            content = _ensure_newline_before_invoke(content)
                            printer.write(content[total_printed:])
                        
                        if hallucination_detected:
                            executor._add_error_feedback_to_history(
                                error_type='hallucination_detected',
                                error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
                            )
                        
                        standardized_tool_calls = executor.parse_tool_calls(content)
                        
                        if not content.endswith('\n'):
                            content += '\n'
                        return content, standardized_tool_calls
                else:
                    # Re-raise if it's a different error
                    raise

