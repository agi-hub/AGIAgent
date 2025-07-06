#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 AGI Bot Research Group.

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

from .config import DEBUG_PROMPT_LIMIT, DEBUG_OUTPUT_LIMIT, DEBUG_RESULT_LIMIT


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
                'prompt': prompt[:DEBUG_PROMPT_LIMIT],  # Limit length for memory efficiency
                'llm_output': llm_output[:DEBUG_OUTPUT_LIMIT],
                'tool_name': tool_name,
                'tool_params': tool_params,
                'tool_result': tool_result[:DEBUG_RESULT_LIMIT],
                'task_completed': task_completed,
                'history_length': history_length,
                'error_msg': error_msg
            }
            
            # Store in memory for potential future use
            self.llm_call_records.append(record)
            
        except Exception as e:
            print(f"❌ Failed to record LLM call: {e}")
    
    def get_records(self) -> List[Dict[str, Any]]:
        """
        Get all recorded LLM call records
        
        Returns:
            List of LLM call records
        """
        return self.llm_call_records.copy()
    
    def get_records_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get LLM call records for a specific task
        
        Args:
            task_id: Task ID to filter by
            
        Returns:
            List of LLM call records for the task
        """
        return [record for record in self.llm_call_records 
                if record.get('task_id') == task_id]
    
    def clear_records(self):
        """Clear all recorded LLM call records"""
        self.llm_call_records.clear()
        if self.debug_mode:
            print("🐛 DEBUG: Cleared all LLM call records")
    
    def export_records_to_json(self, output_file: str):
        """
        Export all records to JSON file
        
        Args:
            output_file: Output JSON file path
        """
        if not self.debug_mode:
            return
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_call_records, f, ensure_ascii=False, indent=2)
            print(f"🐛 DEBUG: Exported {len(self.llm_call_records)} records to {output_file}")
        except Exception as e:
            print(f"❌ Failed to export debug records: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get debug statistics
        
        Returns:
            Dictionary containing debug statistics
        """
        if not self.debug_mode:
            return {}
        
        total_calls = len(self.llm_call_records)
        error_calls = len([r for r in self.llm_call_records if r.get('error_msg')])
        completed_calls = len([r for r in self.llm_call_records if r.get('task_completed')])
        
        # Group by task
        tasks = {}
        for record in self.llm_call_records:
            task_id = record.get('task_id', 'unknown')
            if task_id not in tasks:
                tasks[task_id] = 0
            tasks[task_id] += 1
        
        return {
            'total_calls': total_calls,
            'error_calls': error_calls,
            'completed_calls': completed_calls,
            'tasks_count': len(tasks),
            'calls_per_task': tasks
        }