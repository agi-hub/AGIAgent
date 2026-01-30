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

"""
Debug recorder for LLM call tracking and debugging
"""

import os
import json
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.tools.print_system import print_current
from src.tools.agent_context import get_current_agent_id


class DebugRecorder:
    """Debug recorder for tracking LLM calls and execution details"""
    
    def __init__(self, debug_mode: bool = False, llm_logs_dir: Optional[str] = None, 
                 model: Optional[str] = None, image_data_remove_from_history: Optional[Any] = None):
        """
        Initialize debug recorder
        
        Args:
            debug_mode: Whether to enable debug mode
            llm_logs_dir: Directory for LLM call logs
            model: Model name being used
            image_data_remove_from_history: Image data optimizer instance (optional)
        """
        self.debug_mode = debug_mode
        self.llm_call_records: List[Dict[str, Any]] = []
        self.llm_logs_dir = llm_logs_dir
        self.model = model
        self.image_data_remove_from_history = image_data_remove_from_history
        self.llm_call_counter = 0  # LLM call counter
        
    
    def record_llm_call(self, task_id: str, task_name: str, round_num: int, 
                       prompt: str, llm_output: str, tool_name: str = "", 
                       tool_params: str = "", tool_result: str = "", 
                       task_completed: bool = False, history_length: int = 0,
                       error_msg: str = ""):
        """
        Record detailed information of one LLM call for debugging
        
        Args:
            task_id: Subtask ID
            task_name: Task name
            round_num: Iteration round
            prompt: LLM input
            llm_output: LLM output
            tool_name: Called tool name
            tool_params: Tool parameters (JSON format string)
            tool_result: Tool execution result
            task_completed: Whether task completion flag is detected
            history_length: History record length
            error_msg: Error message
        """
        if not self.debug_mode:
            return
            
        try:
            # Prepare record data for in-memory storage
            timestamp = datetime.now().isoformat()
            
            record = {
                'timestamp': timestamp,
                'task_id': task_id,
                'task_name': task_name,
                'round_num': round_num,
                'prompt': prompt,  # Limit length for memory efficiency
                'llm_output': llm_output,
                'tool_name': tool_name,
                'tool_params': tool_params,
                'tool_result': tool_result,
                'task_completed': task_completed,
                'history_length': history_length,
                'error_msg': error_msg
            }
            
            # Store in memory for potential future use
            self.llm_call_records.append(record)
            
        except Exception as e:
            print_current(f"❌ Failed to record LLM call: {e}")
    
    def save_llm_call_debug_log(self, messages: List[Dict[str, Any]], content: str, tool_call_round: int = 0, tool_calls_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save detailed debug log for LLM call.
        
        Args:
            messages: Complete messages sent to LLM
            content: LLM response content
            tool_call_round: Current tool call round number
            tool_calls_info: Additional tool call information for better logging
        """
        try:
            # Increment call counter
            self.llm_call_counter += 1
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            
            current_agent_id = get_current_agent_id()
            if current_agent_id:
                log_filename = f"llm_call_{current_agent_id}_{self.llm_call_counter:03d}_{timestamp}.json"
            else:
                log_filename = f"llm_call_{self.llm_call_counter:03d}_{timestamp}.json"
            
            # Only create log path if logs directory is available
            if self.llm_logs_dir:
                log_path = os.path.join(self.llm_logs_dir, log_filename)
            else:
                log_path = None
            
            # 🔧 Apply message optimization to remove base64 data from logs
            optimized_messages = self._optimize_messages_for_logging(messages)
            
            # Prepare debug data - including detailed tool call information
            debug_data = {
                "call_info": {
                    "call_number": self.llm_call_counter,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "tool_call_round": tool_call_round  # Track which tool call round this is
                },
                "messages": optimized_messages,
                "response_content": content
            }
            
            # Add tool call information if available
            if tool_calls_info:
                debug_data["tool_calls_info"] = tool_calls_info
                
                # Add detailed breakdown for better debugging
                if "parsed_tool_calls" in tool_calls_info:
                    debug_data["call_info"]["tool_calls_count"] = len(tool_calls_info["parsed_tool_calls"])
                    debug_data["call_info"]["tool_names"] = [tc.get("name", "unknown") for tc in tool_calls_info["parsed_tool_calls"]]
                
                if "tool_results" in tool_calls_info:
                    debug_data["call_info"]["tool_results_count"] = len(tool_calls_info["tool_results"])
                    
                if "formatted_tool_results" in tool_calls_info:
                    debug_data["call_info"]["formatted_results_length"] = len(tool_calls_info["formatted_tool_results"])
            
            # Save to JSON file only if log_path is available
            if log_path:
                with open(log_path, 'w', encoding='utf-8') as f:
                    # Convert escaped newlines to actual newlines in response_content
                    if "response_content" in debug_data:
                        debug_data["response_content"] = debug_data["response_content"].replace('\\n', '\n')

                    # Convert escaped newlines in messages content
                    if "messages" in debug_data:
                        for message in debug_data["messages"]:
                            if "content" in message:
                                message["content"] = message["content"].replace('\\n', '\n')

                    # Convert escaped newlines in tool_calls_info
                    if "tool_calls_info" in debug_data:
                        tool_calls_info = debug_data["tool_calls_info"]
                        if "formatted_tool_results" in tool_calls_info:
                            tool_calls_info["formatted_tool_results"] = tool_calls_info["formatted_tool_results"].replace('\\n', '\n')

                    json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            
        except Exception as e:
            print_current(f"⚠️ Debug log save failed: {e}")

    def _optimize_messages_for_logging(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize messages by replacing base64 data with references for logging purposes.
        
        Args:
            messages: Original messages list
            
        Returns:
            Optimized messages list with base64 data replaced by references
        """
        if not hasattr(self, 'image_data_remove_from_history') or not self.image_data_remove_from_history:
            return messages
        
        optimized_messages = []
        
        for message in messages:
            optimized_message = message.copy()
            
            # Check and optimize content field
            if 'content' in message and isinstance(message['content'], str):
                optimized_content = self._optimize_text_for_logging(message['content'])
                optimized_message['content'] = optimized_content
            
            optimized_messages.append(optimized_message)
        
        return optimized_messages
    
    def _optimize_text_for_logging(self, text: str) -> str:
        """
        Optimize text content by replacing base64 data with lightweight references for logging.
        
        Args:
            text: Original text content
            
        Returns:
            Optimized text with base64 data replaced by references
        """
        if not text or not isinstance(text, str):
            return text
        
        # Detect base64 image data patterns
        base64_pattern = r'[A-Za-z0-9+/]{500,}={0,2}'
        matches = list(re.finditer(base64_pattern, text))
        
        if not matches:
            return text
        
        optimized_text = text
        offset = 0
        
        for match in matches:
            base64_data = match.group()
            
            # Calculate image hash for reference
            image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:16]
            
            # Extract file path info if present
            file_marker_pattern = r'\[FILE_(?:SOURCE|SAVED):([^\]]+)\]'
            file_match = re.search(file_marker_pattern, base64_data)
            file_info = f"|{file_match.group(1)}" if file_match else ""
            
            # Estimate size
            estimated_size_kb = len(base64_data) * 3 // 4 // 1024
            
            # Create compact reference
            reference_text = f"[IMAGE_DATA_REF:{image_hash}|{estimated_size_kb}KB{file_info}]"
            
            # Calculate position in adjusted text
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            
            # Replace base64 data with reference
            optimized_text = (optimized_text[:start_pos] + 
                            reference_text + 
                            optimized_text[end_pos:])
            
            # Update offset
            offset += len(reference_text) - len(base64_data)
        
        return optimized_text
    