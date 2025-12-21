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

from src.tools.print_system import streaming_context, print_debug, print_current


def call_openai_with_chat_based_tools_streaming(executor, messages, system_message):
    """
    Call OpenAI API with chat-based tool calling in streaming mode.
    Tools are described in the message and responses are parsed from content.
    
    Args:
        executor: ToolExecutor instance
        messages: Complete message history for the LLM (including user messages and history)
        system_message: System message
        
    Returns:
        Tuple of (content, tool_calls)
    """
    # Use OpenAI API for chat-based tool calling
    api_messages = [
        {"role": "system", "content": system_message}
    ]
    api_messages.extend(messages)  # messages already contains all user messages and history

    with streaming_context(show_start_message=False) as printer:
            # For OpenAI o1 models and reasoning models, handle thinking if enabled
            # Note: OpenAI streaming may not fully support thinking field, but we handle it if present
            thinking = ""
            thinking_printed_header = False
            
            printer.write(f"\n💬 ")
            
            response = executor.client.chat.completions.create(
                model=executor.model,
                messages=api_messages,
                max_tokens=executor._get_max_tokens_for_model(executor.model),
                temperature=executor.temperature,
                top_p=executor.top_p,
                stream=True
            )

            content = ""
            hallucination_detected = False
            stream_error_occurred = False
            stream_error_message = ""
            
            # Buffer printing mechanism: keep at least 100 characters not printed, used to check hallucinations/multiple json blocks
            min_buffer_size = 100
            total_printed = 0
            
            # Whether to early-stop stream due to detecting the first complete tool call
            tool_call_detected_early = False
            
            try:
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        # Handle thinking content if present (for o1 models)
                        if executor.enable_thinking:
                            thinking_delta = getattr(delta, 'thinking', None)
                            if thinking_delta:
                                if not thinking_printed_header:
                                    printer.write("\n🧠 ")
                                    thinking_printed_header = True
                                printer.write(thinking_delta)
                                thinking += thinking_delta
                        
                        # Handle regular content
                        if delta.content is not None:
                            text = delta.content
                            # If we printed thinking header, add newline before content
                            if thinking_printed_header and not content:
                                printer.write("\n")
                            content += text
                            
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
                                break
                            
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
                                        break
                            
                            # If nothing forced an early exit, print except for trailing buffer
                            unprinted_length = len(content) - total_printed
                            if unprinted_length >= min_buffer_size:
                                print_length = unprinted_length - min_buffer_size
                                if print_length > 0:
                                    printer.write(content[total_printed:total_printed + print_length])
                                    total_printed += print_length
            except Exception as e:
                # Catch exceptions during streaming
                stream_error_occurred = True
                stream_error_message = f"Streaming error: {type(e).__name__}: {str(e)}"
                print_debug(f"⚠️ {stream_error_message}")
                print_current(f"⚠️ OpenAI API streaming error: {str(e)}")
                # Continue with any content received so far
            finally:
                # Ensure the stream is closed properly, whether normally or early
                try:
                    if hasattr(response, 'close'):
                        response.close()
                    if tool_call_detected_early:
                        print_debug("🔌 Stream closed early due to multiple tool calls detection")
                    if hallucination_detected:
                        print_debug("🔌 Stream closed early due to hallucination detection")
                except Exception as close_error:
                    print_debug(f"⚠️ Error closing OpenAI stream: {close_error}")
            
            # Print remaining content not yet printed
            if total_printed < len(content):
                printer.write(content[total_printed:])
            
            # Combine thinking and content if thinking exists (for o1 models)
            if executor.enable_thinking and thinking:
                content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"
            
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

