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
Task completion checker for detecting task completion flags
"""

from typing import List, Dict, Any, Optional

from tools.print_system import print_debug


class TaskChecker:
    """Task completion checker for analyzing LLM responses"""
    
    @staticmethod
    def check_task_completion(result: str) -> bool:
        """
        Check if the large model response contains task completion flag
        Only triggers when a line starts with TASK_COMPLETED or **TASK_COMPLETED
        Only checks LLM's original response content, ignoring tool execution results
        (e.g., terminal output) to avoid false positives from sub-agents.
        
        Args:
            result: Large model response text (may include tool execution results)
            
        Returns:
            Whether task completion flag is detected in LLM's original response
        """
        completion_info = TaskChecker.extract_completion_info(result)
        return completion_info is not None
    
    @staticmethod
    def extract_completion_info(result: str) -> Optional[str]:
        """
        Extract task completion message from LLM response
        
        Args:
            result: Large model response text (may include tool execution results)
            
        Returns:
            Completion message if found, None otherwise
        """
        # Extract only LLM's original response content, ignoring tool execution results
        # Tool execution results are separated by "--- Tool Execution Results ---"
        # We should only check content before this marker
        tool_results_marker = "--- Tool Execution Results ---"
        if tool_results_marker in result:
            # Only check content before tool execution results
            llm_content = result.split(tool_results_marker)[0]
        else:
            # No tool execution results, check entire result
            llm_content = result
        
        # Split LLM content into lines for line-by-line checking
        lines = llm_content.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if line starts with TASK_COMPLETED (any format)
            if stripped_line.startswith("TASK_COMPLETED"):
                # Extract completion description
                try:
                    if stripped_line.startswith("TASK_COMPLETED:"):
                        completion_desc = stripped_line[len("TASK_COMPLETED:"):].strip()
                        return completion_desc if completion_desc else ""
                    else:
                        # TASK_COMPLETED without colon
                        return ""
                except Exception:
                    return ""  # Even if parsing fails, consider task completed
            
            # Check if line starts with **TASK_COMPLETED
            elif stripped_line.startswith("**TASK_COMPLETED"):
                # Extract completion description
                try:
                    if stripped_line.startswith("**TASK_COMPLETED**"):
                        completion_part = stripped_line[len("**TASK_COMPLETED**"):].strip()
                        if completion_part.startswith(":"):
                            completion_part = completion_part[1:].strip()
                        return completion_part if completion_part else ""
                    elif stripped_line.startswith("**TASK_COMPLETED:"):
                        completion_part = stripped_line[len("**TASK_COMPLETED:"):].strip()
                        return completion_part if completion_part else ""
                    else:
                        return ""
                except Exception:
                    return ""  # Even if parsing fails, consider task completed
        
        return None
  
    
    