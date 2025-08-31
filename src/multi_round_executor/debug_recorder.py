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

import json
from datetime import datetime
from typing import List, Dict, Any, Optional


class DebugRecorder:
    """Debug recorder for tracking LLM calls and execution details"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize debug recorder
        
        Args:
            debug_mode: Whether to enable debug mode
        """
        self.debug_mode = debug_mode
        self.llm_call_records: List[Dict[str, Any]] = []
        
    
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
            print(f"❌ Failed to record LLM call: {e}")
    