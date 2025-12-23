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


def call_claude_with_standard_tools(executor, messages, system_message):
    """
    Call Claude with standard tool calling format.
    
    Args:
        executor: ToolExecutor instance
        messages: Complete message history for the LLM (including user messages and history)
        system_message: System message
    """
    # Get standard tools for Anthropic
    tools = executor._convert_tools_to_standard_format("anthropic")
    
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
        claude_messages = [{"role": "user", "content": vision_user_message}]
        # Add remaining messages after the first one
        if len(messages) > 1:
            claude_messages.extend(messages[1:])
        # Clear image data after using it for vision API to prevent reuse in subsequent rounds
        print_current("🧹 Clearing image data after vision API usage")
        executor.current_round_images = []
    else:
        # Prepare messages for Claude - messages already contains all user messages and history
        claude_messages = messages

    # Retry logic for retryable errors
    max_retries = 3
    for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
        try:
            if executor.streaming:
                # Improved streaming logic - handle all event types including tool calls
                content = ""
                tool_calls = []
                
                # Tool call buffers - used to progressively build up the tool call JSON
                tool_call_buffers = {}  # {block_index: {"id": str, "name": str, "input_json": str}}
                current_block_index = None
                current_block_type = None

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
                    temperature = 1.0 if executor.enable_thinking else executor.temperature
                    
                    api_params = {
                        "model": executor.model,
                        "max_tokens": executor._get_max_tokens_for_model(executor.model),
                        "system": system_message,
                        "messages": claude_messages,
                        "tools": tools,
                        "temperature": temperature
                    }
                    
                    # Enable thinking for reasoning-capable models
                    if executor.enable_thinking:
                        api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                    
                    # Try to create stream with thinking parameter, fallback if not supported
                    try:
                        stream = executor.client.messages.stream(**api_params)
                    except (TypeError, ValueError) as e:
                        # If thinking parameter is not supported, retry without it
                        if executor.enable_thinking and ("thinking" in str(e).lower() or "unexpected keyword" in str(e).lower()):
                            print_debug(f"⚠️ Thinking parameter not supported by this API, disabling thinking mode: {e}")
                            # Remove thinking parameter and retry
                            api_params.pop("thinking", None)
                            stream = executor.client.messages.stream(**api_params)
                        else:
                            # Re-raise if it's a different error
                            raise
                    
                    with stream:
                        try:
                            for event in stream:
                                # Before each event, check if hallucination has been detected
                                if hallucination_detected:
                                    print_current("\n🛑 Stopping stream due to hallucination detection")
                                    break
                                    
                                try:
                                    event_type = getattr(event, 'type', None)
                                    last_event_type = event_type
                                    
                                    # Record raw event data for debugging
                                    try:
                                        # Try to get all attributes of the event
                                        event_dict = {}
                                        if hasattr(event, '__dict__'):
                                            event_dict = event.__dict__
                                        elif hasattr(event, 'model_dump'):
                                            event_dict = event.model_dump()
                                    except:
                                        pass

                                    # Handle content_block_start event
                                    if event_type == "content_block_start":
                                        try:
                                            
                                            # Safely get index attribute
                                            try:
                                                block_index = getattr(event, 'index', None)
                                            except Exception as idx_err:
                                                print_error(f"   Error getting index: {type(idx_err).__name__}: {str(idx_err)}")
                                                block_index = None
                                            
                                            # Safely get content_block attribute
                                            try:
                                                content_block = getattr(event, 'content_block', None)

                                                # Try serializing content_block to see what's inside
                                                if content_block:
                                                    try:
                                                        if hasattr(content_block, '__dict__'):
                                                            pass
                                                    except Exception as dump_err:
                                                        pass
                                            except Exception as cb_err:
                                                print_error(f"   Error getting content_block: {type(cb_err).__name__}: {str(cb_err)}")
                                                import traceback
                                                content_block = None
                                            
                                            if content_block:
                                                try:
                                                    block_type = getattr(content_block, 'type', None)
                                                    current_block_index = block_index
                                                    current_block_type = block_type
                                                    
                                                    if block_type == "thinking" and executor.enable_thinking:
                                                        # Thinking block started
                                                        printer.write("\n🧠 ")
                                                        thinking_printed_header = True
                                                    elif block_type == "text":
                                                        # Text block started
                                                        if thinking_printed_header:
                                                            printer.write("\n\n💬 ")
                                                        else:
                                                            printer.write("\n💬 ")
                                                        answer_started = True
                                                    elif block_type == "tool_use":
                                                        # Start a new tool call
                                                        tool_id = getattr(content_block, 'id', '')
                                                        tool_name = getattr(content_block, 'name', '')
                                                        
                                                        tool_call_buffers[block_index] = {
                                                            "id": tool_id,
                                                            "name": tool_name,
                                                            "input_json": ""
                                                        }
                                                except Exception as type_err:
                                                    print_error(f"   Error processing block_type: {type(type_err).__name__}: {str(type_err)}")
                                        except Exception as e:
                                            print_error(f"⚠️ Error processing content_block_start: {type(e).__name__}: {str(e)}")
                                            import traceback
                                            print_error(f"   Full traceback:\n{traceback.format_exc()}")
                                            # Continue processing other events without interrupting the stream

                                    # Handle content_block_delta event
                                    elif event_type == "content_block_delta":
                                        try:
                                            delta = getattr(event, 'delta', None)
                                            block_index = getattr(event, 'index', None)
                                            
                                            if delta:
                                                delta_type = getattr(delta, 'type', None)
                                                
                                                if delta_type == "thinking_delta" and executor.enable_thinking:
                                                    # Streaming of thinking content
                                                    thinking_text = getattr(delta, 'thinking', '')
                                                    printer.write(thinking_text)
                                                
                                                elif delta_type == "text_delta":
                                                    # Streaming of text content
                                                    text = getattr(delta, 'text', '')
                                                    # Detect hallucination patterns - case insensitive
                                                    text_lower = text.lower()
                                                    if ("llm called following tools" in text_lower or 
                                                        "tool execution results" in text_lower):
                                                        print_current("\nHallucination detected, stopping conversation")
                                                        hallucination_detected = True
                                                        # Add error feedback to history
                                                        executor._add_error_feedback_to_history(
                                                            error_type='hallucination_detected',
                                                            error_message="Hallucination pattern detected in response (e.g., 'Tool execution results:', '**Tool Execution Results', etc.)"
                                                        )
                                                        break
                                                    printer.write(text)
                                                    content += text
                                                
                                                elif delta_type == "input_json_delta":
                                                    partial_json = getattr(delta, 'partial_json', '')

                                                    if block_index in tool_call_buffers and partial_json:
                                                        current_json = tool_call_buffers[block_index]["input_json"]
                                                        
                                                        tool_call_buffers[block_index]["input_json"] += partial_json
                                        except Exception as e:
                                            print_debug(f"⚠️ Error processing content_block_delta: {type(e).__name__}: {str(e)}")

                                    # Handle content_block_stop event
                                    elif event_type == "content_block_stop":
                                        try:
                                            block_index = getattr(event, 'index', None)
                                            
                                            if block_index in tool_call_buffers:
                                                # Tool call block stopped, validate and save
                                                buffer = tool_call_buffers[block_index]
                                                tool_name = buffer["name"]
                                                json_str = buffer["input_json"]
                                                
                                                # Validate JSON integrity
                                                # Fix boolean value format if needed
                                                # json_str = _fix_json_boolean_values(json_str)
                                                is_valid, parsed_input, error_msg = validate_tool_call_json(json_str, tool_name)
                                                
                                                if is_valid:
                                                    tool_calls.append({
                                                        "id": buffer["id"],
                                                        "name": tool_name,
                                                        "input": parsed_input
                                                    })
                                                    print_debug(f"✅ Tool call validated: {tool_name}")
                                                else:
                                                    print_error(f"❌ Tool call JSON validation failed for {tool_name}:")
                                                    print_error(f"   {error_msg}")
                                                    print_debug(f"   Raw JSON: {json_str[:200]}...")
                                                    # Add error feedback to history
                                                    executor._add_error_feedback_to_history(
                                                        error_type='json_parse_error',
                                                        error_message=f"Tool call JSON validation failed for {tool_name}: {error_msg}"
                                                    )
                                        except Exception as e:
                                            print_debug(f"⚠️ Error processing content_block_stop: {type(e).__name__}: {str(e)}")

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

                                                    # 总是打印 token 使用情况
                                                    if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                                        print_debug(f"\n📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                                                    else:
                                                        print_debug(f"\n📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}")
                                        except Exception as e:
                                            print_debug(f"⚠️ Error processing message_delta: {type(e).__name__}: {str(e)}")
                                
                                except Exception as event_error:
                                    # Single event processing failure should not interrupt the entire stream
                                    print_debug(f"⚠️ Error processing event {last_event_type}: {type(event_error).__name__}: {str(event_error)}")
                                    # Do not use continue, let the loop continue naturally

                        except Exception as e:
                            # Improved error handling
                            stream_error_occurred = True
                            error_details = f"Streaming failed at event_type={last_event_type}: {type(e).__name__}: {str(e)}"
                            print_error(error_details)
                            # If there is some partial tool call data, try to save it
                            if tool_call_buffers:
                                pass

                            # Try falling back to text_stream
                            try:
                                for text in stream.text_stream:
                                    # Detect hallucination patterns - case insensitive
                                    text_lower = text.lower()
                                    if ("llm called following tools" in text_lower or 
                                        "tool execution results" in text_lower):
                                        print_current("\nHallucination detected, stopping conversation")
                                        hallucination_detected = True
                                        # Add error feedback to history
                                        executor._add_error_feedback_to_history(
                                            error_type='hallucination_detected',
                                            error_message="Hallucination pattern detected in response (e.g., 'Tool execution results:', '**Tool Execution Results', etc.)"
                                        )
                                        break
                                    printer.write(text)
                                    content += text
                            except Exception as fallback_error:
                                print_error(f"Text streaming also failed: {fallback_error}")
                                break

                        # If hallucination was detected, return early
                        if hallucination_detected:
                            return content, []

                    print_current("")

                    # Get token usage from final_message
                    try:
                        final_message = stream.get_final_message()
                        if hasattr(final_message, 'usage') and final_message.usage:
                            usage = final_message.usage
                            input_tokens = getattr(usage, 'input_tokens', 0) or 0
                            output_tokens = getattr(usage, 'output_tokens', 0) or 0
                            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0
                            
                            if cache_creation_tokens > 0 or cache_read_tokens > 0:
                                print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}, Cache Creation: {cache_creation_tokens}, Cache Read: {cache_read_tokens}")
                            else:
                                print_debug(f"📊 Current conversation token usage - Input: {input_tokens}, Output: {output_tokens}")
                    except Exception as e:
                        print_debug(f"⚠️ Unable to get final_message token info: {type(e).__name__}: {str(e)}")

                    # If there were no complete tool calls collected from streaming, try extracting from the final message
                    if not tool_calls and not stream_error_occurred:
                        try:
                            final_message = stream.get_final_message()
                            
                            for content_block in final_message.content:
                                if content_block.type == "tool_use":
                                    # Validate tool call input
                                    tool_input = content_block.input
                                    tool_name = content_block.name
                                    
                                    # The input should already be a dict, but check just in case
                                    if isinstance(tool_input, str):
                                        #tool_input = _fix_json_boolean_values(tool_input)
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

                            if tool_calls:
                                pass
                        except Exception as e:
                            print_error(f"Failed to get final message: {type(e).__name__}: {str(e)}")

                # Execute tool calls
                if tool_calls:
                    for i, tool_call_data in enumerate(tool_calls):
                        try:
                            tool_name = tool_call_data['name']
                            tool_params = tool_call_data['input']

                            # Print tool name and parameters before execution in JSON format
                            tool_call_json = {
                                "tool_name": tool_name,
                                "tool_index": i + 1,
                                "parameters": tool_params if isinstance(tool_params, dict) else {}
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
                                'tool_params': tool_call_data['input'],
                                'tool_result': tool_result
                            })

                            executor._tools_executed_in_stream = True

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
                temperature = 1.0 if executor.enable_thinking else executor.temperature
                
                api_params = {
                    "model": executor.model,
                    "max_tokens": executor._get_max_tokens_for_model(executor.model),
                    "system": system_message,
                    "messages": claude_messages,
                    "tools": tools,
                    "temperature": temperature
                }
                
                # Enable thinking for reasoning-capable models
                if executor.enable_thinking:
                    api_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                
                # Try to create response with thinking parameter, fallback if not supported
                try:
                    response = executor.client.messages.create(**api_params)
                except (TypeError, ValueError) as e:
                    # If thinking parameter is not supported, retry without it
                    if executor.enable_thinking and ("thinking" in str(e).lower() or "unexpected keyword" in str(e).lower()):
                        print_debug(f"⚠️ Thinking parameter not supported by this API, disabling thinking mode: {e}")
                        # Remove thinking parameter and retry
                        api_params.pop("thinking", None)
                        response = executor.client.messages.create(**api_params)
                    else:
                        # Re-raise if it's a different error
                        raise
                
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
                tool_calls = []
                
                # Extract thinking and content from Anthropic response
                if executor.enable_thinking:
                    # Check if response has thinking attribute (for reasoning models)
                    thinking = getattr(response, 'thinking', None) or ""
                
                # Extract content and tool use blocks
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == "thinking":
                            # Extract thinking content from thinking blocks
                            if executor.enable_thinking:
                                if hasattr(content_block, 'text'):
                                    thinking += content_block.text
                                elif hasattr(content_block, 'thinking'):
                                    thinking += content_block.thinking
                        elif content_block.type == "text":
                            content += content_block.text
                        elif content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
                            
                            # Print tool name and parameters in JSON format
                            tool_call_json = {
                                "tool_name": tool_name,
                                "tool_index": len(tool_calls) + 1,
                                "parameters": tool_input if isinstance(tool_input, dict) else {}
                            }
                            print_current("```json")
                            print_current(json.dumps(tool_call_json, ensure_ascii=False, indent=2))
                            print_current("```")
                            
                            tool_calls.append({
                                "id": content_block.id,
                                "name": tool_name,
                                "input": tool_input
                            })
                
                # Combine thinking and content if thinking exists
                if executor.enable_thinking and thinking:
                    content = f"## Thinking Process\n\n{thinking}\n\n## Final Answer\n\n{content}"

                # Check for hallucination patterns in non-streaming response - case insensitive
                content_lower = content.lower()
                hallucination_patterns = [
                    "llm called following tools",
                    "**tool execution results",
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

                # Print LLM response in non-streaming mode
                if content:
                    print_current("")
                    print_current("💬 LLM Response:")
                    print_current(content)

                return content, tool_calls

        except Exception as e:
            error_str = str(e).lower()

            # Check if this is a retryable error
            retryable_errors = [
                'overloaded', 'rate limit', 'too many requests',
                'service unavailable', 'timeout', 'temporary failure',
                'server error', '429', '503', '502', '500'
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

                print_current(f"⚠️ Claude API {matched_error_keyword} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print_current(f"💡 Consider switching to a different model or trying again later")
                print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                print_current(f"🔄 Retrying in {retry_delay} seconds...")

                # Wait before retry
                time.sleep(retry_delay)
                continue  # Retry the loop
                
            else:
                # Non-retryable error or max retries exceeded
                if is_retryable:
                    print_current(f"❌ Claude API {matched_error_keyword} error: Maximum retries ({max_retries}) exceeded")
                    print_current(f"💡 Consider switching to a different model or trying again later")
                    print_current(f"🔄 You can change the model in config.txt and restart AGIAgent")
                else:
                    print_current(f"❌ Claude API call failed: {e}")
                
                raise e

