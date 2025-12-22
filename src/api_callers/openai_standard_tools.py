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

import json
import time
from src.tools.print_system import streaming_context, print_debug, print_current, print_error
from src.utils.parse import validate_tool_call_json


def call_openai_with_standard_tools(executor, messages, system_message):
    """
    Call OpenAI with standard tool calling format.
    
    Args:
        executor: ToolExecutor instance
        messages: Complete message history for the LLM (including user messages and history)
        system_message: System message
    """
    # Get standard tools for OpenAI
    tools = executor._convert_tools_to_standard_format("openai")
    
    # Check if we have stored image data for vision API
    if hasattr(executor, 'current_round_images') and executor.current_round_images:
        print_current(f"🖼️ Using vision API with {len(executor.current_round_images)} stored images")
        # Extract first user message content from messages for vision API
        first_user_message_content = ""
        if messages and len(messages) > 0 and messages[0].get("role") == "user":
            content = messages[0].get("content", "")
            if isinstance(content, str):
                first_user_message_content = content
            elif isinstance(content, dict) and "text" in content:
                first_user_message_content = content.get("text", "")
        # Build vision message with stored images
        vision_user_message = executor._build_vision_message(first_user_message_content)
        api_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": vision_user_message}
        ]
        # Add remaining messages after the first one
        if len(messages) > 1:
            api_messages.extend(messages[1:])
        # Clear image data after using it for vision API to prevent reuse in subsequent rounds
        print_current("🧹 Clearing image data after vision API usage")
        executor.current_round_images = []
    else:
        # Prepare messages - messages already contains all user messages and history
        api_messages = [
            {"role": "system", "content": system_message}
        ]
        api_messages.extend(messages)
    
    # Retry logic for retryable errors
    max_retries = 3
    for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
        try:
            if executor.streaming:
                # Streaming logic - stop as soon as the complete tool call block has been received
                content = ""
                tool_calls = []
                tool_calls_buffer = {}  # Used to collect incremental tool call information
                
                with streaming_context(show_start_message=False) as printer:
                    # Show LLM is starting to speak emoji
                    printer.write(f"\n💬 ")
                    hallucination_detected = False

                response = executor.client.chat.completions.create(
                    model=executor.model,
                    messages=api_messages,
                    tools=tools,
                    max_tokens=executor._get_max_tokens_for_model(executor.model),
                    temperature=executor.temperature,
                    top_p=executor.top_p,
                    stream=True
                    )

                try:
                    tool_calls_completed = False
                    empty_chunks_after_tool_calls = 0
                    max_empty_chunks = 3  # Allow up to 3 empty chunks after tool call completion to catch trailing text
                    last_chunk = None  # 保存最后一个chunk以获取usage信息
                    
                    for chunk in response:
                        last_chunk = chunk  # 保存每个chunk，最后一个chunk包含usage信息
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            finish_reason = chunk.choices[0].finish_reason
                            
                            # Stream output of text content
                            if delta.content is not None:
                                printer.write(delta.content)
                                content += delta.content
                                # Check for hallucination patterns in streaming content
                                content_lower = content.lower()
                                if ("tool execution results" in content_lower or 
                                    "**llm called following tools" in content_lower or
                                    "**tool execution results" in content_lower):
                                    hallucination_detected = True
                                    print_debug("\nHallucination Detected in streaming, stopping...")
                                # If tool calls have finished, but still getting content, reset empty chunk counter
                                if tool_calls_completed:
                                    empty_chunks_after_tool_calls = 0
                            
                            # Incrementally collect tool call deltas
                            if delta.tool_calls:
                                for tool_call_delta in delta.tool_calls:
                                    idx = tool_call_delta.index
                                    if idx not in tool_calls_buffer:
                                        tool_calls_buffer[idx] = {
                                            "id": "",
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": ""
                                            }
                                        }
                                    
                                    # Accumulate tool call information
                                    if tool_call_delta.id:
                                        tool_calls_buffer[idx]["id"] = tool_call_delta.id
                                    if tool_call_delta.function:
                                        if tool_call_delta.function.name:
                                            tool_calls_buffer[idx]["function"]["name"] = tool_call_delta.function.name
                                        if tool_call_delta.function.arguments:
                                            tool_calls_buffer[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                            
                            # Check finish_reason
                            if finish_reason is not None:
                                if finish_reason == "tool_calls":
                                    # Tool calls completed, but there may be following text, continue processing
                                    tool_calls_completed = True
                                    print_debug("🔧 Tool calls completed, continuing to receive any following text...")
                                else:
                                    # Other end reasons (like "stop"), finish normally
                                    print_debug(f"✅ Streaming response finished: {finish_reason}")
                                    break
                            else:
                                # If no finish_reason, check if we've received empty chunks after tool calls completed
                                if tool_calls_completed:
                                    # Check if this chunk is empty (no content and no tool calls)
                                    has_content = delta.content is not None and len(delta.content.strip()) > 0
                                    has_tool_calls = delta.tool_calls is not None and len(delta.tool_calls) > 0
                                    
                                    if not has_content and not has_tool_calls:
                                        empty_chunks_after_tool_calls += 1
                                        # If several consecutive empty chunks after tool calls, assume stream is done
                                        if empty_chunks_after_tool_calls >= max_empty_chunks:
                                            print_debug(f"🔚 Received {max_empty_chunks} empty chunks after tool calls, stopping reception")
                                            break
                                    else:
                                        # Got content, reset counter
                                        empty_chunks_after_tool_calls = 0
                finally:
                    # Explicitly close streaming connection, notify server to stop generating
                    # This ensures server side knows client has stopped receiving
                    if hasattr(response, 'close'):
                        try:
                            response.close()
                            print_debug("🔌 Explicitly closed streaming connection")
                        except Exception as e:
                            print_debug(f"⚠️ Error closing streaming connection: {e}")
                    
                    print_current("")
                
                # Get token usage from the last chunk of streaming response
                # OpenAI streaming response contains usage info in the last chunk
                try:
                    if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                        usage = last_chunk.usage
                        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                        total_tokens = getattr(usage, 'total_tokens', 0) or 0
                        print_debug(f"📊 Current conversation token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
                except Exception as e:
                    print_debug(f"⚠️ Unable to get streaming response token info: {type(e).__name__}: {str(e)}")
                
                # Convert tool calls buffer into list
                if tool_calls_buffer:
                    for idx in sorted(tool_calls_buffer.keys()):
                        tool_calls.append(tool_calls_buffer[idx])
                
                # Execute tool calls
                if tool_calls:
                    for i, tool_call in enumerate(tool_calls):
                        try:
                            tool_name = tool_call["function"]["name"]
                            tool_params_str = tool_call["function"]["arguments"]
                            
                            # Use enhanced JSON validation and parsing
                            # Fix boolean value format if needed
                            # tool_params_str = _fix_json_boolean_values(tool_params_str)
                            is_valid, tool_params, error_msg = validate_tool_call_json(tool_params_str, tool_name)
                            
                            if not is_valid:
                                print_error(f"❌ Tool {i + 1} ({tool_name}) JSON parsing failed:")
                                print_error(f"   {error_msg}")
                                print_debug(f"   Raw arguments: {tool_params_str[:200]}...")
                                # Add error feedback to history
                                executor._add_error_feedback_to_history(
                                    error_type='json_parse_error',
                                    error_message=f"Tool {i + 1} ({tool_name}) JSON parsing failed: {error_msg}"
                                )
                                continue
                            
                            # Print tool name and parameters before execution in JSON format
                            tool_call_json = {
                                "tool_name": tool_name,
                                "tool_index": i + 1,
                                "parameters": tool_params if tool_params else {}
                            }
                            print_current("```json")
                            print_current(json.dumps(tool_call_json, ensure_ascii=False, indent=2))
                            print_current("```")
                            
                            # Convert to standard format
                            standard_tool_call = {
                                "name": tool_name,
                                "arguments": tool_params
                            }
                            
                            tool_result = executor.execute_tool(standard_tool_call, streaming_output=True)
                            
                            # Store result
                            if not hasattr(executor, '_streaming_tool_results'):
                                executor._streaming_tool_results = []
                            
                            executor._streaming_tool_results.append({
                                'tool_name': tool_name,
                                'tool_params': tool_params,
                                'tool_result': tool_result
                            })
                            
                            executor._tools_executed_in_stream = True
                            
                        except Exception as e:
                            print_error(f"❌ Tool {i + 1} execution failed: {type(e).__name__}: {str(e)}")
                            print_debug(f"   Tool: {tool_call.get('function', {}).get('name', 'unknown')}")
                    
                    print_debug("✅ All tool executions completed")

                # If hallucination was detected, return early with empty tool calls
                if hallucination_detected:
                    # Add error feedback to history
                    executor._add_error_feedback_to_history(
                        error_type='hallucination_detected',
                        error_message="Hallucination pattern detected in response (e.g., '**LLM Called Following Tools in this round' or '**Tool Execution Results')"
                    )
                    return content, []

                # print_current("\n✅ Streaming completed")
                return content, tool_calls
            else:
                # print_current("🔄 Starting batch generation with standard tools...")
                response = executor.client.chat.completions.create(
                    model=executor.model,
                    messages=api_messages,
                    tools=tools,
                    max_tokens=executor._get_max_tokens_for_model(executor.model),
                    temperature=executor.temperature,
                    top_p=executor.top_p,
                    stream=False
                )

                # Check if response is a Stream object (should not happen with stream=False)
                if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                    # If we got a Stream object, consume it to get the actual response
                    print_current("⚠️ Warning: Received Stream object despite stream=False. Converting to regular response...")
                    content = ""
                    tool_calls = []
                    for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content is not None:
                                content += delta.content
                            # Collect tool calls from chunks if present
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    tool_calls.append(tc)
                    
                    # Print LLM response in non-streaming mode
                    if content:
                        print_current("")
                        print_current("💬 LLM Response:")
                        print_current(content)
                    
                    return content, tool_calls

                # Extract content and thinking field from OpenAI response
                message = response.choices[0].message
                content = message.content or ""

                # Handle thinking field for OpenAI o1 models and other reasoning models
                if executor.enable_thinking:
                    thinking = getattr(message, 'thinking', None)
                    if thinking:
                        # Combine thinking and content with clear separation
                        content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

                # Check for hallucination patterns in non-streaming response - case insensitive
                content_lower = content.lower()
                hallucination_patterns = [
                    "**llm called following tools",
                    "**tool execution results",
                    "tool execution results:",
                    "tool execution results"
                ]
                
                hallucination_detected = False
                hallucination_start = len(content)
                for pattern in hallucination_patterns:
                    if pattern in content_lower:
                        # Find the actual position in original content (case-sensitive search for exact match)
                        pattern_variants = [
                            pattern,
                            pattern.capitalize(),
                            pattern.upper(),
                            "**" + pattern if not pattern.startswith("**") else pattern
                        ]
                        for variant in pattern_variants:
                            pos = content.find(variant)
                            if pos != -1:
                                hallucination_detected = True
                                hallucination_start = min(hallucination_start, pos)
                                break
                
                if hallucination_detected:
                    print_debug("\nHallucination Detected, stop chat")
                    # Truncate content at hallucination location to avoid printing hallucination string
                    if hallucination_start > 0:
                        content = content[:hallucination_start].rstrip()
                    else:
                        content = ""
                    return content, []

                raw_tool_calls = response.choices[0].message.tool_calls or []
                
                # Convert OpenAI tool_calls objects to dictionary format
                tool_calls = []
                for i, tool_call in enumerate(raw_tool_calls):
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    # Print tool name and parameters before returning in JSON format
                    try:
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    except:
                        tool_args = {}
                    
                    tool_call_json = {
                        "tool_name": tool_name,
                        "tool_index": i + 1,
                        "parameters": tool_args if isinstance(tool_args, dict) else {}
                    }
                    print_current("```json")
                    print_current(json.dumps(tool_call_json, ensure_ascii=False, indent=2))
                    print_current("```")
                    
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args_str
                        }
                    })
                
                # Print token usage in non-streaming mode
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                    total_tokens = getattr(usage, 'total_tokens', 0) or 0
                    print_debug(f"📊 Current conversation token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")

                # Print LLM response in non-streaming mode
                if content:
                    print_current("")
                    print_current("💬 LLM Response:")
                    print_current(content)
                
                # print_current("✅ Generation completed")
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
                retry_delay = min(2 ** attempt, 10)  # 1, 2, 4 seconds, max 10
                
                print_current(f"⚠️ OpenAI API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print_current(f"💡 Consider switching to a different model or trying again later")
                print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                print_current(f"🔄 Retrying in {retry_delay} seconds...")
                
                # Wait before retry
                time.sleep(retry_delay)
                continue  # Retry the loop
                
            else:
                # Non-retryable error or max retries exceeded
                if is_retryable:
                    print_current(f"❌ OpenAI API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                else:
                    print_current(f"❌ OpenAI API call failed: {e}")
                
                raise e

