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

    def _call_glm_with_standard_tools(self, messages, user_message, system_message):
        # Use Anthropic Claude API for chat-based tool calling
        claude_messages = [{"role": "user", "content": user_message}]
        
        if self.streaming:
            with streaming_context(show_start_message=False) as printer:
                # Prepare parameters for Anthropic API
                # Note: When thinking is enabled, temperature MUST be 1.0
                temperature = 1.0 if self.enable_thinking else self.temperature
                
                api_params = {
                    "model": self.model,
                    "max_tokens": self._get_max_tokens_for_model(self.model),
                    "system": system_message,
                    "messages": claude_messages,
                    "temperature": temperature
                }
                
                with self.client.messages.stream(**api_params) as stream:
                                content = ""
                                hallucination_detected = False
                                stream_error_occurred = False
                                stream_error_message = ""
                                
                                # 缓冲打印机制：至少缓冲100个字符
                                buffer = ""
                                min_buffer_size = 100
                                total_printed = 0
                                
                                # 标志：是否因为检测到第一个完整的工具调用而提前停止
                                tool_call_detected_early = False
                                
                                # Thinking tracking
                                thinking_printed_header = False
                                answer_started = False
                                
                                try:
                                    # Use event-based streaming to capture thinking
                                    for event in stream:
                                        event_type = getattr(event, 'type', None)
                                        
                                        # Handle content block start
                                        if event_type == "content_block_start":
                                            content_block = getattr(event, 'content_block', None)
                                            if content_block:
                                                block_type = getattr(content_block, 'type', None)
                                                if block_type == "thinking" and self.enable_thinking:
                                                    printer.write("\n🧠 ")
                                                    thinking_printed_header = True
                                                elif block_type == "text":
                                                    if thinking_printed_header:
                                                        printer.write("\n\n💬 ")
                                                    else:
                                                        printer.write("\n💬 ")
                                                    answer_started = True
                                        
                                        # Handle content block deltas
                                        elif event_type == "content_block_delta":
                                            delta = getattr(event, 'delta', None)
                                            if delta:
                                                delta_type = getattr(delta, 'type', None)
                                                
                                                # Thinking content
                                                if delta_type == "thinking_delta" and self.enable_thinking:
                                                    thinking_text = getattr(delta, 'thinking', '')
                                                    printer.write(thinking_text)
                                                
                                                # Text content
                                                elif delta_type == "text_delta":
                                                    text = getattr(delta, 'text', '')
                                                    buffer += text
                                                    content += text
                                                    
                                                    # Check for hallucination patterns
                                                    # 使用更严格的匹配：只检查完整模式，避免部分匹配误判
                                                    # 流式输出可能被截断，但只有在匹配完整模式时才判定为幻觉
                                                    hallucination_patterns = [
                                                        "**LLM Called Following Tools in this round",
                                                        "**Tool Execution Results:**"
                                                    ]
                                                    hallucination_detected_flag = False
                                                    hallucination_start = -1
                                                    
                                                    # 只检查完整模式匹配，避免部分字符串误判
                                                    for pattern in hallucination_patterns:
                                                        if pattern in content:
                                                            hallucination_start = content.find(pattern)
                                                            hallucination_detected_flag = True
                                                            break
                                                    
                                                    if hallucination_detected_flag:
                                                        print_debug("\nHallucination Detected, stop chat")
                                                        hallucination_detected = True
                                                        # 计算幻觉开始位置相对于已打印内容的位置
                                                        if hallucination_start >= total_printed:
                                                            # 幻觉字符串还在buffer中，未被打印
                                                            content = content[:hallucination_start].rstrip()
                                                            buffer = content[total_printed:] if len(content) > total_printed else ""
                                                        else:
                                                            # 幻觉字符串已经被部分打印了，只能截断剩余的
                                                            content = content[:hallucination_start].rstrip()
                                                            buffer = ""
                                                        break
                                                    
                                                    # Check for multiple tool calls
                                                    first_json_pos = content.find('```json')
                                                    if first_json_pos != -1:
                                                        second_json_pos = content.find('```json', first_json_pos + len('```json'))
                                                        if second_json_pos != -1:
                                                            print_debug("\n🛑 Multiple tool calls detected, stopping stream after first JSON block")
                                                            tool_call_detected_early = True
                                                            content = content[:second_json_pos].rstrip()
                                                            if len(buffer) > len(content) - total_printed:
                                                                buffer = content[total_printed:] if len(content) > total_printed else ""
                                                            break
                                                    
                                                    # 🔧 关键修复：在打印之前，检查content末尾是否包含幻觉模式的部分匹配
                                                    # 如果包含部分匹配，则保留该部分在buffer中不打印，等待更多字符确认
                                                    can_print_buffer = True
                                                    if len(buffer) >= min_buffer_size:
                                                        # 检查content末尾是否有幻觉模式的部分匹配
                                                        # 从content末尾往前检查，看是否匹配任何幻觉模式的前缀
                                                        max_check_length = max(len(pattern) for pattern in hallucination_patterns)
                                                        check_text = content[-max_check_length:] if len(content) > max_check_length else content
                                                        
                                                        for pattern in hallucination_patterns:
                                                            # 检查是否有部分匹配（从pattern的前缀开始）
                                                            for prefix_len in range(1, len(pattern)):
                                                                prefix = pattern[:prefix_len]
                                                                if check_text.endswith(prefix) and prefix_len >= 3:  # 至少3个字符才考虑部分匹配
                                                                    # 发现部分匹配，需要保留这部分不打印
                                                                    # 计算需要保留的字符数
                                                                    keep_in_buffer = prefix_len
                                                                    # 只打印buffer中可以安全打印的部分
                                                                    safe_print_length = len(buffer) - keep_in_buffer
                                                                    if safe_print_length > 0:
                                                                        printer.write(buffer[:safe_print_length])
                                                                        total_printed += safe_print_length
                                                                        buffer = buffer[safe_print_length:]
                                                                    can_print_buffer = False
                                                                    break
                                                            if not can_print_buffer:
                                                                break
                                                        
                                                        # 如果没有部分匹配，正常打印整个buffer
                                                        if can_print_buffer:
                                                            printer.write(buffer)
                                                            total_printed += len(buffer)
                                                            buffer = ""
                                                    
                                                    if tool_call_detected_early:
                                                        break
                                except Exception as e:
                                    # 捕获流式处理中的异常
                                    stream_error_occurred = True
                                    stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                                    print_debug(f"⚠️ {stream_error_message}")
                                    print_current(f"⚠️ Claude API streaming error: {str(e)}")
                                    # 继续处理已接收的内容
                                finally:
                                    # 确保流被正确关闭（无论是正常结束还是提前停止）
                                    try:
                                        if hasattr(stream, 'close'):
                                            stream.close()
                                        if tool_call_detected_early:
                                            print_debug("🔌 Stream closed early due to multiple tool calls detection")
                                    except Exception as close_error:
                                        print_debug(f"⚠️ Error closing Anthropic stream: {close_error}")
                                
                                # 如果发生流错误，记录并继续处理
                                if stream_error_occurred:
                                    print_current(f"⚠️ Streaming response interrupted, processed content length: {len(content)} characters")
                                    if not content:
                                        # 如果没有接收到任何内容，重新抛出异常
                                        raise Exception(f"Anthropic API streaming failed: {stream_error_message}")
                                
                                # 打印剩余缓冲区
                                if buffer:
                                    printer.write(buffer)
                                
                                
                                # If hallucination was detected, return early
                                if hallucination_detected:
                                    # 添加错误反馈到历史记录
                                    self._add_error_feedback_to_history(
                                        error_type='hallucination_detected',
                                        error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
                                    )
                                    # 添加换行（仅限chat接口）
                                    if not content.endswith('\n'):
                                        content += '\n'
                                    return content, []
                                
                                # 检查是否有工具调用（即使只有一个）
                                # 查找第一个```json块或纯JSON格式的工具调用
                                has_json_block = '```json' in content
                                # 也检查纯JSON格式（不带```json标记）
                                has_plain_json_tool_call = ('"tool_name"' in content and '"parameters"' in content) and not has_json_block
                                
                                if has_json_block or has_plain_json_tool_call:
                                    # 确保JSON块完整（如果使用```json标记）
                                    if has_json_block:
                                        # 只获取第一个JSON块之前的内容，忽略后续的JSON块
                                        content_for_parsing = self._get_content_before_second_json(content)
                                    else:
                                        # 纯JSON格式，直接使用content（但也要检查是否有多个JSON对象）
                                        # 查找第一个完整的JSON对象
                                        first_json_end = content.find('}\n```', content.find('"tool_name"'))
                                        if first_json_end != -1:
                                            content_for_parsing = content[:first_json_end + 1]
                                        else:
                                            content_for_parsing = content
                                    
                                    # Parse tool calls from the accumulated content (只解析第一个块)
                                    tool_calls = self.parse_tool_calls(content_for_parsing)
                                    
                                    # 只保留第一个工具调用（双重保险）
                                    if tool_calls and len(tool_calls) > 1:
                                        print_current(f"⚠️ Warning: Multiple tool calls detected ({len(tool_calls)}), keeping only the first one")
                                        # 添加错误反馈到历史记录
                                        self._add_error_feedback_to_history(
                                            error_type='multiple_tools_detected',
                                            error_message=f"Multiple tool calls detected ({len(tool_calls)}), only the first one was executed"
                                        )
                                        tool_calls = [tool_calls[0]]
                                    
                                    # parse_tool_calls now returns standardized format with "input" field
                                    standardized_tool_calls = tool_calls
                                    
                                    # 调试：检查转换结果
                                    if not standardized_tool_calls:
                                        print_current(f"⚠️ Warning: No tool calls parsed. Content for parsing length: {len(content_for_parsing)}")
                                        if self.debug_mode:
                                            print_current(f"Content snippet: {content_for_parsing[:500]}...")
                                    
                                    # 添加换行（仅限chat接口）
                                    if not content_for_parsing.endswith('\n'):
                                        content_for_parsing += '\n'
                                    return content_for_parsing, standardized_tool_calls
                                
                                # 没有工具调用，返回空列表
                                # 添加换行（仅限chat接口）
                                if not content.endswith('\n'):
                                    content += '\n'
                                return content, []
        else:
            # print_current("🔄 LLM is thinking:")
            # Prepare parameters for Anthropic API
            # Note: When thinking is enabled, temperature MUST be 1.0
            temperature = 1.0 if self.enable_thinking else self.temperature
            
            api_params = {
                "model": self.model,
                "max_tokens": self._get_max_tokens_for_model(self.model),
                "system": system_message,
                "messages": claude_messages,
                "temperature": temperature
            }
            
            # Enable thinking for reasoning-capable models
            if self.enable_thinking:
                api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            
            response = self.client.messages.create(**api_params)
            
            content = ""
            thinking = ""
            
            # Extract thinking and content from Anthropic response
            if self.enable_thinking:
                # Check if response has thinking attribute (for reasoning models)
                thinking = getattr(response, 'thinking', None) or ""
                
                # Also check for thinking in content blocks
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == "thinking":
                            if hasattr(content_block, 'text'):
                                thinking += content_block.text
                            elif hasattr(content_block, 'thinking'):
                                thinking += content_block.thinking
                
            # Extract text content
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
            
            # Combine thinking and content if thinking exists
            if self.enable_thinking and thinking:
                content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

            # Check for hallucination patterns in non-streaming response - strict match
            # 统一使用完整模式，避免部分匹配误判
            hallucination_patterns = [
                "**LLM Called Following Tools in this round",
                "**Tool Execution Results:**"
            ]
            hallucination_detected = any(pattern in content for pattern in hallucination_patterns)
            if hallucination_detected:
                # 添加错误反馈到历史记录
                self._add_error_feedback_to_history(
                    error_type='hallucination_detected',
                    error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
                )
                # print_current("\nHallucination Detected, stop chat")  # Reduced verbose output
                # 添加换行（仅限chat接口）
                if not content.endswith('\n'):
                    content += '\n'
                return content, []
            
            # 如果内容包含多个JSON块，只解析第一个块
            if '```json' in content:
                content_for_parsing = self._get_content_before_second_json(content)
            else:
                content_for_parsing = content
            
            # Parse tool calls from the response content (只解析第一个块)
            tool_calls = self.parse_tool_calls(content_for_parsing)
            
            # 只保留第一个工具调用（双重保险）
            if tool_calls and len(tool_calls) > 1:
                print_current(f"⚠️ Warning: Multiple tool calls detected ({len(tool_calls)}), keeping only the first one")
                # 添加错误反馈到历史记录
                self._add_error_feedback_to_history(
                    error_type='multiple_tools_detected',
                    error_message=f"Multiple tool calls detected ({len(tool_calls)}), only the first one was executed"
                )
                tool_calls = [tool_calls[0]]
            
            # parse_tool_calls now returns standardized format with "input" field
            standardized_tool_calls = tool_calls
            
            # 添加换行（仅限chat接口）
            if not content.endswith('\n'):
                content += '\n'
            return content, standardized_tool_calls

    def _call_glm_with_standard_tools(self, messages, user_message, system_message):
        """
        Call GLM with standard tool calling format.
        """
        # 重置警告跟踪，避免在新的LLM调用中携带旧的警告状态
        self._last_parse_warning_length = -1
        
        # Get standard tools for Anthropic
        tools = self._convert_tools_to_standard_format("anthropic")
        
        # Check if we have stored image data for vision API
        if hasattr(self, 'current_round_images') and self.current_round_images:
            print_current(f"🖼️ Using vision API with {len(self.current_round_images)} stored images")
            # Build vision message with stored images
            vision_user_message = self._build_vision_message(user_message if isinstance(user_message, str) else user_message.get("text", ""))
            claude_messages = [{"role": "user", "content": vision_user_message}]
            # Clear image data after using it for vision API to prevent reuse in subsequent rounds
            print_current("🧹 Clearing image data after vision API usage")
            self.current_round_images = []
        else:
            # Prepare messages for Claude - user_message can be string or content array
            claude_messages = [{"role": "user", "content": user_message}]
        

        
        # Retry logic for retryable errors
        max_retries = 3
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                if self.streaming:
                    # Simplified streaming logic - only handle message streaming, read tool calls from final messages in other parts
                    content = ""
                    tool_calls = []

                    with streaming_context(show_start_message=True) as printer:
                        hallucination_detected = False
                        stream_error_occurred = False
                        last_event_type = None
                        error_details = None
                        
                        # Thinking tracking
                        thinking_printed_header = False
                        answer_started = False

                        # Prepare parameters for Anthropic API
                        # Note: When thinking is enabled, temperature MUST be 1.0
                        temperature = 1.0 if self.enable_thinking else self.temperature
                        
                        api_params = {
                            "model": self.model,
                            "max_tokens": self._get_max_tokens_for_model(self.model),
                            "system": system_message,
                            "messages": claude_messages,
                            "tools": tools,
                            "temperature": temperature
                        }
                        
                        with self.client.messages.stream(**api_params) as stream:
                            try:
                                for event in stream:
                                    try:
                                        event_type = getattr(event, 'type', None)
                                        last_event_type = event_type

                                        # Handle content block start
                                        if event_type == "content_block_start":
                                            content_block = getattr(event, 'content_block', None)
                                            if content_block:
                                                block_type = getattr(content_block, 'type', None)
                                                if block_type == "thinking" and self.enable_thinking:
                                                    printer.write("\n🧠 ")
                                                    thinking_printed_header = True
                                                elif block_type == "text":
                                                    if thinking_printed_header:
                                                        printer.write("\n\n💬 ")
                                                    else:
                                                        printer.write("\n💬 ")
                                                    answer_started = True

                                        # Handle content streaming events
                                        elif event_type == "content_block_delta":
                                            try:
                                                delta = getattr(event, 'delta', None)

                                                if delta:
                                                    delta_type = getattr(delta, 'type', None)

                                                    # Thinking content
                                                    if delta_type == "thinking_delta" and self.enable_thinking:
                                                        thinking_text = getattr(delta, 'thinking', '')
                                                        printer.write(thinking_text)

                                                    # Text content
                                                    elif delta_type == "text_delta":
                                                        text = getattr(delta, 'text', '')
                                                        # Check for hallucination
                                                        if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                                            print_current("\nHallucination detected, stopping conversation")
                                                            hallucination_detected = True
                                                            break
                                                        printer.write(text)
                                                        content += text
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing content_block_delta: {type(e).__name__}: {str(e)}")
                                                # Continue processing other events

                                        # 处理消息统计信息
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
                                                            print_debug(f"\n📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                            except Exception as e:
                                                print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")

                                    except Exception as event_error:
                                        # Single event processing failure should not interrupt the entire stream
                                        print_debug(f"⚠️ Error processing event {last_event_type}: {type(event_error).__name__}: {str(event_error)}")
                                        # Do not use continue, let the loop continue naturally

                            except Exception as e:
                                # Check if it's a JSON parsing error, if so ignore and continue streaming inference
                                error_str = str(e)
                                if "expected value at line 1 column" in error_str and "ValueError" in str(type(e)):
                                    # JSON parsing error, ignore and continue processing other events
                                    print_debug(f"⚠️ JSON parsing error ignored for event_type={last_event_type}: {type(e).__name__}: {str(e)}")
                                    continue  # 继续处理下一个事件
                                else:
                                    # 其他类型的错误，使用增强的错误处理
                                    stream_error_occurred = True
                                    error_details = f"Streaming failed at event_type={last_event_type}: {type(e).__name__}: {str(e)}"
                                    print_debug(error_details)

                                    # 尝试回退到text_stream
                                    try:
                                        for text in stream.text_stream:
                                            if "LLM Called Following Tools in this round" in text or "**Tool Execution Results:**" in text:
                                                print_current("\nHallucination detected, stopping conversation")
                                                hallucination_detected = True
                                                # 添加错误反馈到历史记录
                                                self._add_error_feedback_to_history(
                                                    error_type='hallucination_detected',
                                                    error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results:**')"
                                                )
                                                break
                                            printer.write(text)
                                            content += text
                                    except Exception as fallback_error:
                                        print_error(f"Text streaming also failed: {fallback_error}")
                                        break

                            # 如果检测到幻觉，提前返回
                            if hallucination_detected:
                                return content, []

                        print_current("")

                        # Read tool calls directly from final message
                        if not stream_error_occurred:
                            try:
                                final_message = stream.get_final_message()

                                for content_block in final_message.content:
                                    if content_block.type == "tool_use":
                                        # 验证工具调用input
                                        tool_input = content_block.input
                                        tool_name = content_block.name

                                        # input should already be dict, but check for safety
                                        if isinstance(tool_input, str):
                                            # Fix boolean format issues
                                            tool_input = _fix_json_boolean_values(tool_input)
                                            is_valid, parsed_input, error_msg = validate_tool_call_json(tool_input, tool_name)
                                            if not is_valid:
                                                # Only show failure info for non-empty string errors
                                                if error_msg != "Empty JSON string":
                                                    print_error(f"❌ Final message tool call validation failed: {error_msg}")
                                                else:
                                                    print_debug(f"⚠️ Empty tool input for {tool_name}, skipping")
                                                continue
                                            tool_input = parsed_input

                                        tool_calls.append({
                                            "id": content_block.id,
                                            "name": tool_name,
                                            "input": tool_input
                                        })

                            except Exception as e:
                                print_error(f"Failed to get final message: {type(e).__name__}: {str(e)}")

                    # Execute tool calls
                    if tool_calls:
                        for tool_call_data in tool_calls:
                            try:
                                tool_name = tool_call_data['name']

                                # Convert to standard format
                                standard_tool_call = {
                                    "name": tool_name,
                                    "arguments": tool_call_data['input']
                                }

                                tool_result = self.execute_tool(standard_tool_call, streaming_output=True)

                                # 存储结果
                                if not hasattr(self, '_streaming_tool_results'):
                                    self._streaming_tool_results = []

                                self._streaming_tool_results.append({
                                    'tool_name': tool_name,
                                    'tool_params': tool_call_data['input'],
                                    'tool_result': tool_result
                                })

                                self._tools_executed_in_stream = True

                            except Exception as e:
                                print_error(f"❌ Tool {tool_name} execution failed: {str(e)}")

                        print_debug("✅ All tool executions completed")

                    # If an error occurred during streaming, append error details to content for feedback to the LLM
                    if stream_error_occurred and error_details is not None:
                        error_feedback = f"\n\n⚠️ **Streaming Error Feedback**: There was a problem parsing the previous response: {error_details}\nPlease regenerate a correct response based on this error message."
                        content += error_feedback

                    return content, tool_calls
                else:
                    # print_current("🔄 LLM is thinking: ")
                    # Prepare parameters for Anthropic API
                    # Note: When thinking is enabled, temperature MUST be 1.0
                    temperature = 1.0 if self.enable_thinking else self.temperature
                    
                    api_params = {
                        "model": self.model,
                        "max_tokens": self._get_max_tokens_for_model(self.model),
                        "system": system_message,
                        "messages": claude_messages,
                        "tools": tools,
                        "temperature": temperature
                    }
                    
                    # Enable thinking for reasoning-capable models
                    if self.enable_thinking:
                        api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                    
                    response = self.client.messages.create(**api_params)
                    
                    content = ""
                    tool_calls = []
                    
                    # Extract content and tool use blocks
                    for content_block in response.content:
                        if content_block.type == "text":
                            content += content_block.text
                        elif content_block.type == "tool_use":
                            tool_calls.append({
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": content_block.input
                            })
                    
                    return content, tool_calls
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                retryable_errors = [
                    'overloaded', 'rate limit', 'too many requests',
                    'service unavailable', 'timeout', 'temporary failure',
                    'server error', '429', '503', '502', '500',
                    'peer closed connection', 'incomplete chunked read'
                ]
                
                # Find which error keyword matched
                matched_error_keyword = None
                for error_keyword in retryable_errors:
                    if error_keyword in error_str:
                        matched_error_keyword = error_keyword
                        break
                
                is_retryable = matched_error_keyword is not None
                
                if is_retryable and attempt < max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = 1
                    
                    print_current(f"⚠️ GLM API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    print_current(f"🔄 Retrying in {retry_delay} seconds...")
                    
                    # Wait before retry
                    time.sleep(retry_delay)
                    continue  # Retry the loop
                    
                else:
                    # Non-retryable error or max retries exceeded
                    if is_retryable:
                        print_current(f"❌ GLM API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                        print_current(f"💡 Consider switching to a different model or trying again later")
                        print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                    else:
                        print_current(f"❌ GLM API call failed: {e}")
                    
                    raise e
    
    def _get_content_before_second_json(self, content: str) -> str:
        """
        获取第二个```json块之前的内容
        确保包含完整的第一个JSON块
        
        Args:
            content: 完整的响应内容
            
        Returns:
            str: 第二个```json块之前的内容，确保第一个JSON块完整
        """
        second_json_pos = self._find_second_json_block_start(content)
        if second_json_pos == -1:
            # 没有找到第二个```json，尝试确保第一个块完整
            return self._ensure_first_json_block_complete(content)
        
        # 截取到第二个```json之前的内容
        content_before_second = content[:second_json_pos].rstrip()
        
        # 确保第一个块是完整的
        result = self._ensure_first_json_block_complete(content_before_second)
        
        return result
    
    def _extract_json_block_robust(self, content: str, start_marker: str = '```json') -> Optional[str]:
        """
        更健壮地提取JSON块，处理嵌套的```标记和不完整的JSON块。
        特别优化了对超长JSON块（如包含大量文本的code_edit字段）的处理。
        
        Args:
            content: 要提取的内容
            start_marker: JSON块开始标记，默认为'```json'
            
        Returns:
            提取的JSON字符串，如果提取失败返回None
        """
        json_start = content.find(start_marker)
        if json_start == -1:
            return None
        
        # 从标记后开始查找JSON内容
        json_content_start = json_start + len(start_marker)
        
        # 策略1: 先尝试找到结束的```标记
        # 对于超长JSON块，使用更智能的查找策略
        json_content_end = -1
        
        # 查找JSON块的结束标记```
        # 优化：对于tool_name/parameters格式，可以利用结尾的 }\n} 模式
        # 先尝试找到最后一个```（JSON块的真正结束）
        last_triple_backtick = content.rfind('```', json_content_start)
        if last_triple_backtick > json_content_start:
            # 检查这个位置之前是否有 }\n} 模式（说明这是JSON的结束）
            # 对于超长内容，扩大检查范围
            check_range = 50 if len(content) > 10000 else 20
            before_marker = content[max(0, last_triple_backtick-check_range):last_triple_backtick]
            # 检查多种可能的结尾模式
            if ('}\n}' in before_marker or '}\n  }' in before_marker or 
                before_marker.rstrip().endswith('}') or
                content[last_triple_backtick-1:last_triple_backtick] in ['\n', '\r', ' ']):
                # 这很可能是JSON块的结束标记
                json_content_end = last_triple_backtick
            else:
                # 继续使用原来的逻辑查找
                i = current_pos = json_content_start
                while i < len(content) - 2:
                    if content[i:i+3] == '```':
                        # 检查这是否是开始标记（前面没有内容或是换行）
                        # 对于超长内容，放宽检查条件
                        if i == json_content_start or content[i-1] in ['\n', '\r', ' ']:
                            # 检查这个位置之前是否有JSON结束模式
                            check_before = content[max(0, i-50):i]
                            if ('}\n}' in check_before or '}\n  }' in check_before or 
                                check_before.rstrip().endswith('}')):
                                # 这是结束标记
                                json_content_end = i
                                break
                    i += 1
        
        # 策略2: 如果没有找到结束标记，使用括号匹配来找到完整的JSON对象
        if json_content_end == -1:
            # 没有找到结束标记，可能JSON块不完整或超长
            # 尝试找到最后一个完整的JSON对象或数组
            remaining = content[json_content_start:]
            
            # 使用括号匹配来找到完整的JSON对象
            # 对于超长内容，使用更高效的算法
            brace_count = 0
            bracket_count = 0
            in_string = False
            escape_next = False
            last_valid_pos = -1
            i = 0
            
            # 跳过开头的空白
            while i < len(remaining) and remaining[i] in ' \t\n\r':
                i += 1
            
            # 如果第一个字符是{，开始计数
            if i < len(remaining) and remaining[i] == '{':
                brace_count = 1
                i += 1
                
                # 对于超长内容，优化性能：批量处理字符
                while i < len(remaining):
                    char = remaining[i]
                    
                    if escape_next:
                        escape_next = False
                        i += 1
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        i += 1
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        i += 1
                        continue
                        
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and bracket_count == 0:
                                last_valid_pos = i + 1
                                break
                        elif char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if brace_count == 0 and bracket_count == 0:
                                last_valid_pos = i + 1
                                break
                    
                    i += 1
            
            if last_valid_pos > 0:
                return remaining[:last_valid_pos].strip()
            # 如果找不到完整的JSON，返回剩余内容（让JSON解析器尝试处理）
            # 但尝试找到最后一个可能的结束位置
            # 对于包含code_edit的超长JSON，尝试找到最后一个}
            if 'code_edit' in remaining:
                last_brace = remaining.rfind('}')
                if last_brace > 0:
                    # 检查这个位置是否合理（前面应该有匹配的{）
                    test_json = remaining[:last_brace+1].strip()
                    # 简单验证：检查是否以{开头
                    if test_json.startswith('{'):
                        return test_json
            return remaining.strip()
        
        # 提取JSON内容
        json_content = content[json_content_start:json_content_end].strip()
        return json_content
    

    def _execute_tool_immediately(self, tool_call, tool_index):
        """
        Execute a tool call immediately during streaming.
        
        Args:
            tool_call: The complete tool call object
            tool_index: The tool index for display purposes
        """
        try:
            tool_name = tool_call["function"]["name"]
            tool_params_str = tool_call["function"]["arguments"]
            
            # Parse parameters
            import json
            tool_params = json.loads(tool_params_str)
            
            print_current(f"⚡ Executing tool {tool_index} immediately: {tool_name}")
            print_current(f"   Parameters: {tool_params}")
            
            # Convert to standard format for execute_tool
            standard_tool_call = {
                "name": tool_name,
                "arguments": tool_params
            }
            
            tool_result = self.execute_tool(standard_tool_call, streaming_output=True)
            
            # Store result for later response formatting
            if not hasattr(self, '_streaming_tool_results'):
                self._streaming_tool_results = []
            
            self._streaming_tool_results.append({
                'tool_name': tool_name,
                'tool_params': tool_params,
                'tool_result': tool_result
            })
            
            # Set flag indicating tools were executed during streaming
            self._tools_executed_in_stream = True
            
            # Tool result is already displayed by streaming output, no need to duplicate
            
        except Exception as e:
            print_current(f"   ❌ Tool {tool_index} execution failed: {str(e)}")



def rebuild_json_structure(json_str: str) -> str:
    """
    Last resort method to rebuild JSON structure from malformed JSON.
    
    Args:
        json_str: Malformed JSON string
        
    Returns:
        Rebuilt JSON string
    """
    # Try to extract key-value pairs and rebuild the JSON
    # This is a very basic approach for specific cases
    
    # Look for pattern: "key": "value" or "key": value
    pairs = []
    
    # Extract string values
    string_pattern = r'"([^"]+)":\s*"([^"]*(?:\\.[^"]*)*)"'
    for match in re.finditer(string_pattern, json_str, re.DOTALL):
        key = match.group(1)
        value = match.group(2)
        # Escape the value properly
        value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        pairs.append(f'"{key}": "{value}"')
    
    # Extract non-string values (numbers, booleans, null)
    non_string_pattern = r'"([^"]+)":\s*([^",}\s]+)'
    for match in re.finditer(non_string_pattern, json_str):
        key = match.group(1)
        value = match.group(2).strip()
        # Skip if this was already captured as a string
        if not any(f'"{key}":' in pair for pair in pairs):
            pairs.append(f'"{key}": {value}')
    
    if pairs:
        rebuilt_json = '{' + ', '.join(pairs) + '}'
        try:
            # Validate the rebuilt JSON
            json.loads(rebuilt_json)
            return rebuilt_json
        except json.JSONDecodeError:
            pass
    
    # If rebuild failed, return original
    return json_str



def parse_python_params_manually(params_str: str) -> Dict[str, Any]:
    """
    Manually parse Python function parameters when JSON parsing fails.
    
    Args:
        params_str: Parameter string from Python function call
        
    Returns:
        Dictionary of parameters
    """
    params = {}
    
    # Remove the outer braces if present
    if params_str.startswith('{') and params_str.endswith('}'):
        params_str = params_str[1:-1].strip()
    
    # Split by commas, but be careful about commas inside strings
    param_parts = []
    current_part = ""
    in_quotes = False
    quote_char = None
    brace_depth = 0
    
    for char in params_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current_part += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_part += char
        elif char == '{' and not in_quotes:
            brace_depth += 1
            current_part += char
        elif char == '}' and not in_quotes:
            brace_depth -= 1
            current_part += char
        elif char == ',' and not in_quotes and brace_depth == 0:
            param_parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    if current_part.strip():
        param_parts.append(current_part.strip())
    
    # Parse each parameter
    for part in param_parts:
        # Look for key: value pattern
        if ':' in part:
            key_value = part.split(':', 1)
            if len(key_value) == 2:
                key = key_value[0].strip().strip('"\'')
                value = key_value[1].strip()
                
                # Remove quotes from value if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Convert boolean values
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # Convert numeric values
                elif value.isdigit():
                    value = int(value)
                
                params[key] = value
    
    return params


def convert_xml_parameter_value(value: str) -> Any:
    """
    Convert XML parameter value to appropriate type.
    
    This function is used in XML parsing to convert string parameter values
    extracted from XML tags to their appropriate Python types (bool, int, list, etc.).
    
    Args:
        value: String value extracted from XML to convert
        
    Returns:
        Converted value (string, int, bool, list, etc.)
    """
    # For certain parameters that may contain meaningful whitespace/formatting,
    # don't strip the value
    value_stripped = value.strip()
    
    # Handle boolean values
    if value_stripped.lower() in ('true', 'false'):
        return value_stripped.lower() == 'true'
    
    # Handle integers
    if value_stripped.isdigit():
        return int(value_stripped)
    
    # Handle negative integers
    if value_stripped.startswith('-') and value_stripped[1:].isdigit():
        return int(value_stripped)
    
    # Handle JSON arrays/objects
    if (value_stripped.startswith('[') and value_stripped.endswith(']')) or (value_stripped.startswith('{') and value_stripped.endswith('}')):
        try:
            return json.loads(value_stripped)
        except json.JSONDecodeError:
            pass
    
    # Return original value (not stripped) for string parameters to preserve formatting
    return value



def _fix_json_boolean_values(json_str: str) -> str:
    """
    Fix boolean value formatting issues in a JSON string.
    Replace :True with :true and :False with :false.

    Args:
        json_str: The original JSON string.

    Returns:
        The corrected JSON string.
    """
    json_str = re.sub(r':\s*True\b', ': true', json_str)
    json_str = re.sub(r':\s*False\b', ': false', json_str)
    return json_str




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy functions from tool_executor.py that are temporarily unused.

This file contains deprecated or legacy code that may be needed in the future.
"""

from typing import Dict, Any
from src.tools.print_system import print_current


def auto_correct_tool_parameters(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-correct wrong parameter names for various tools.
    
    This function handles parameter name corrections for backward compatibility
    and robustness. It filters out None values and empty strings, and maps
    incorrect parameter names to correct ones.
    
    Args:
        tool_name: Name of the tool being executed
        params: Dictionary of parameters passed to the tool
        
    Returns:
        Dictionary with corrected and filtered parameters
    """
    # Filter out None values and empty strings for optional parameters
    # Special handling: for edit_file, preserve empty string for code_edit parameter
    if tool_name == "edit_file":
        filtered_params = {k: v for k, v in params.items() if v is not None and not (v == "" and k != "code_edit")}
    else:
        filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
    
    # Special handling for read_file to map end_line_one_indexed to end_line_one_indexed_inclusive
    if tool_name == "read_file" and "end_line_one_indexed" in filtered_params:
        # Map end_line_one_indexed to end_line_one_indexed_inclusive
        filtered_params["end_line_one_indexed_inclusive"] = filtered_params.pop("end_line_one_indexed")
        #print_current("Mapped end_line_one_indexed parameter to end_line_one_indexed_inclusive")
    
    # Robustness handling: auto-correct wrong parameter names for edit_file and read_file
    if tool_name in ["edit_file", "read_file"]:
        # Map relative_workspace_path to target_file
        if "relative_workspace_path" in filtered_params:
            filtered_params["target_file"] = filtered_params.pop("relative_workspace_path")
            print_current(f"🔧 Auto-corrected parameter: relative_workspace_path -> target_file for {tool_name}")
        # Map file_path to target_file
        if "file_path" in filtered_params:
            filtered_params["target_file"] = filtered_params.pop("file_path")
            print_current(f"🔧 Auto-corrected parameter: file_path -> target_file for {tool_name}")
        # Map filename to target_file (for edit_file)
        if "filename" in filtered_params:
            filtered_params["target_file"] = filtered_params.pop("filename")
            print_current(f"🔧 Auto-corrected parameter: filename -> target_file for {tool_name}")
    
    # Robustness handling for edit_file: auto-correct content to code_edit
    if tool_name == "edit_file" and "content" in filtered_params:
        # Map content to code_edit
        filtered_params["code_edit"] = filtered_params.pop("content")
        print_current(f"🔧 Auto-corrected parameter: content -> code_edit for {tool_name}")
    
    # Robustness handling for workspace_search: auto-correct search_term to query
    if tool_name == "workspace_search" and "search_term" in filtered_params:
        # Map search_term to query
        filtered_params["query"] = filtered_params.pop("search_term")
        print_current(f"🔧 Auto-corrected parameter: search_term -> query for {tool_name}")
    
    return filtered_params


def _display_llm_statistics(self, messages: List[Dict[str, Any]], response_content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Display LLM input/output statistics including token count and character count.
        
        Args:
            messages: Input messages sent to LLM
            response_content: Response content from LLM
            tool_calls: Tool calls from LLM response (optional)
        """
        try:
            # Calculate input statistics
            input_text = ""
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                input_text += f"[{role}] {content}\n"
            
            # Detect if input contains images (base64 data)
            import re
            has_images = bool(re.search(r'[A-Za-z0-9+/]{100,}={0,2}', input_text))
            
            # Estimate token counts for response content, including image tokens
            input_tokens_est = estimate_token_count(input_text, has_images=has_images, model=self.model)
            output_tokens_est = estimate_token_count(response_content, has_images=False, model=self.model)
            
            # Estimate token counts for tool calls if present
            tool_calls_tokens = 0
            if tool_calls:
                tool_calls_text = self._format_tool_calls_for_token_estimation(tool_calls)
                tool_calls_tokens = estimate_token_count(tool_calls_text)
            

        except Exception as e:
            print_current(f"⚠️ Statistics calculation failed: {e}")

def _format_tool_calls_for_token_estimation(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Format tool calls into text for token estimation.
        
        Args:
            tool_calls: List of tool calls
            
        Returns:
            Formatted text representation of tool calls
        """
        if not tool_calls:
            return ""
        
        formatted_parts = []
        for tool_call in tool_calls:
            # Handle different tool call formats
            if isinstance(tool_call, dict):
                # Extract tool name
                tool_name = ""
                if "name" in tool_call:
                    tool_name = tool_call["name"]
                elif "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", "")
                
                # Extract parameters/arguments
                params = {}
                if "arguments" in tool_call:
                    params = tool_call["arguments"]
                elif "input" in tool_call:
                    params = tool_call["input"]
                elif "function" in tool_call and isinstance(tool_call["function"], dict):
                    if "arguments" in tool_call["function"]:
                        try:
                            import json
                            params = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except:
                            params = tool_call["function"]["arguments"]
                
                # Format tool call as text
                tool_text = f"Tool: {tool_name}\n"
                if params:
                    import json
                    try:
                        params_text = json.dumps(params, ensure_ascii=False)
                        tool_text += f"Parameters: {params_text}\n"
                    except:
                        tool_text += f"Parameters: {str(params)}\n"
                
                formatted_parts.append(tool_text)
        
        return "\n".join(formatted_parts)
    
    # Cache analysis functions moved to utils/cacheeff.py
  
def _format_tool_results_with_vision(self, tool_results: List[Dict[str, Any]], vision_images: List[Dict[str, Any]]) -> Any:
        """
        Format tool results that contain vision data for LLM.
        Returns the proper format for vision-capable models.
        
        Args:
            tool_results: List of tool execution results
            vision_images: List of vision image data
            
        Returns:
            Properly formatted content for vision models (content array format)
        """
        truncation_length = get_truncation_length()
        
        # Build text content first
        message_parts = ["Tool execution results:\n"]
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get('tool_name', 'unknown')
            tool_params = result.get('tool_params', {})
            tool_result = result.get('tool_result', '')
            
            # Format the tool result section
            message_parts.append(f"## Tool {i}: {tool_name}")
            
            # Add parameters if meaningful
            if tool_params:
                key_params = []
                # Check if tool_params is a dictionary before calling .items()
                if isinstance(tool_params, dict):
                    for key, value in tool_params.items():
                        if key in ['target_file', 'query', 'command', 'relative_workspace_path', 'search_term', 'instructions']:
                            # Show full parameter values without truncation
                            key_params.append(f"{key}={value}")
                    if key_params:
                        message_parts.append(f"**Parameters:** {', '.join(key_params)}")
                else:
                    # If tool_params is not a dict (e.g., string), just show it as is
                    message_parts.append(f"**Parameters:** {tool_params}")
            
            # Format the result
            message_parts.append("**Result:**")
            if isinstance(tool_result, dict):
                if tool_result.get('success') is not None:
                    # Structured result format
                    status = "✅ Success" if tool_result.get('success') else "❌ Failed"
                    message_parts.append(status)
                    
                    for key, value in tool_result.items():
                        # For image data in get_sensor_data, show metadata but reference image below
                        if (tool_name == 'get_sensor_data' and key == 'data' and 
                            any(img['tool_index'] == i for img in vision_images)):
                            message_parts.append(f"- {key}: [IMAGE DATA - See image below]")
                            print_current(f"📸 Image data formatted for vision API, tool {i}")
                        elif key not in ['status', 'command', 'working_directory']:
                            # Show full content without truncation
                            message_parts.append(f"- {key}: {value}")
            
            # Add separator between tools
            if i < len(tool_results):
                message_parts.append("")  # Empty line for separation
        
        # Build content array with text and images
        text_content = '\n'.join(message_parts)
        
        # Create content array format for vision models
        content_parts = []
        
        # Add text part
        content_parts.append({
            "type": "text",
            "text": text_content
        })
        
        # Add image parts
        for img_data in vision_images:
            if self.is_claude:
                # Claude format
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_data['mime_type'],
                        "data": img_data['data']
                    }
                })
            else:
                # OpenAI format
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_data['mime_type']};base64,{img_data['data']}"
                    }
                })
        
        print_current(f"🖼️ Formatted {len(vision_images)} images for vision API ({self.model})")
        return content_parts

def _perform_vision_analysis(self, vision_content: List[Dict[str, Any]], original_content: str) -> str:
        """
        Perform immediate vision analysis using the vision-capable model.
        
        Args:
            vision_content: Content array with text and images for vision API
            original_content: Original LLM response content
            
        Returns:
            Vision analysis result as string
        """
        try:
            # Prepare system prompt for vision analysis
            vision_system_prompt = "You are an AI assistant with vision capabilities. Analyze the images provided and give detailed descriptions of what you see."
            
            # Prepare messages for vision analysis
            vision_messages = [
                {"role": "system", "content": vision_system_prompt},
                {"role": "user", "content": vision_content}
            ]
            
            print_current("🔍 Performing vision analysis...")
            
            # Call LLM with vision data
            if self.is_claude:
                # Prepare parameters for Anthropic API
                # Note: When thinking is enabled, temperature MUST be 1.0
                temperature = 1.0 if self.enable_thinking else self.temperature
                
                api_params = {
                    "model": self.model,
                    "max_tokens": self._get_max_tokens_for_model(self.model),
                    "system": vision_system_prompt,
                    "messages": [{"role": "user", "content": vision_content}],
                    "temperature": temperature
                }
                
                # Enable thinking for reasoning-capable models
                if self.enable_thinking:
                    api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                
                response = self.client.messages.create(**api_params)
                
                vision_analysis = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        vision_analysis += content_block.text
                        
            else:
                # OpenAI format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=vision_messages,
                    max_tokens=self._get_max_tokens_for_model(self.model),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False
                )

                # Extract content and thinking field from OpenAI response
                message = response.choices[0].message
                vision_analysis = message.content or ""

                # Handle thinking field for OpenAI o1 models and other reasoning models
                if self.enable_thinking:
                    thinking = getattr(message, 'thinking', None)
                    if thinking:
                        # Combine thinking and content with clear separation
                        vision_analysis = f"## Thinking Process\n\n{thinking}\n\n## Analysis Result\n\n{vision_analysis}"
            
            print_current(f"✅ Vision analysis completed: {len(vision_analysis)} characters")
            return f"## Vision Analysis Results:\n\n{vision_analysis}"
            
        except Exception as e:
            print_current(f"❌ Vision analysis failed: {e}")
            # Fall back to text description
            text_content = ""
            for item in vision_content:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                    break
            return f"## Tool Results (Vision analysis failed):\n\n{text_content}"