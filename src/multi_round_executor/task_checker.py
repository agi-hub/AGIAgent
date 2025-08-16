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
Task completion checker for detecting task completion flags
"""

from typing import List, Dict, Any

from tools.print_system import print_debug


class TaskChecker:
    """Task completion checker for analyzing LLM responses"""
    
    @staticmethod
    def check_task_completion(result: str) -> bool:
        """
        Check if the large model response contains task completion flag
        Only triggers when a line starts with TASK_COMPLETED or **TASK_COMPLETED
        
        Args:
            result: Large model response text
            
        Returns:
            Whether task completion flag is detected
        """
        # Split result into lines for line-by-line checking
        lines = result.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if line starts with TASK_COMPLETED (any format)
            if stripped_line.startswith("TASK_COMPLETED"):
                # Extract completion description
                try:
                    if stripped_line.startswith("TASK_COMPLETED:"):
                        completion_desc = stripped_line[len("TASK_COMPLETED:"):].strip()
                    return True
                except Exception as e:
                    return True  # Even if parsing fails, consider task completed
            
            # Check if line starts with **TASK_COMPLETED
            elif stripped_line.startswith("**TASK_COMPLETED"):
                # Extract completion description
                try:
                    if stripped_line.startswith("**TASK_COMPLETED**"):
                        completion_part = stripped_line[len("**TASK_COMPLETED**"):].strip()
                        if completion_part.startswith(":"):
                            completion_part = completion_part[1:].strip()
                        completion_desc = completion_part
                    else:
                        pass
                    return True
                except Exception as e:
                    return True  # Even if parsing fails, consider task completed
        
        return False
  
    
    