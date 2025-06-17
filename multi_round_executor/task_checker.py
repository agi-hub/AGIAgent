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

from .config import TASK_COMPLETION_KEYWORDS


class TaskChecker:
    """Task completion checker for analyzing LLM responses"""
    
    @staticmethod
    def check_task_completion(result: str) -> bool:
        """
        Check if the large model response contains task completion flag
        
        Args:
            result: Large model response text
            
        Returns:
            Whether task completion flag is detected
        """
        # Check if task completion flag is included
        if "TASK_COMPLETED:" in result:
            # Extract completion description
            try:
                completion_part = result.split("TASK_COMPLETED:")[1].strip()
                completion_desc = completion_part.split('\n')[0].strip()
                print(f"🎉 Task completion flag detected: {completion_desc}")
                return True
            except Exception as e:
                print(f"⚠️ Error parsing task completion flag: {e}")
                return True  # Even if parsing fails, consider task completed
        
        return False
    
    @staticmethod
    def analyze_task_success(history: List[Dict[str, Any]]) -> bool:
        """
        Analyze task execution history to determine success
        
        Args:
            history: Task execution history
            
        Returns:
            True if task appears to be successful
        """
        if not history:
            return False
        
        # Check for success indicators in results
        success_count = 0
        total_results = 0
        
        for record in history:
            if 'result' in record and not record.get('error'):
                total_results += 1
                result = record['result'].lower()
                
                # Check for success keywords
                if any(keyword in result for keyword in TASK_COMPLETION_KEYWORDS):
                    success_count += 1
        
        # Consider successful if more than half of results contain success indicators
        return total_results > 0 and (success_count / total_results) > 0.5
    
    @staticmethod
    def extract_key_achievements(history: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key achievements from task history
        
        Args:
            history: Task execution history
            
        Returns:
            List of key achievements
        """
        achievements = []
        
        for record in history:
            if 'result' in record and not record.get('error'):
                result = record['result']
                
                # Look for achievement indicators
                for keyword in TASK_COMPLETION_KEYWORDS:
                    if keyword in result.lower():
                        # Extract the sentence containing the keyword
                        sentences = result.split('.')
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                achievements.append(sentence.strip()[:200])
                                break
                        break
        
        # Remove duplicates and limit quantity
        return list(dict.fromkeys(achievements))[:5]
    
    @staticmethod
    def extract_created_files(history: List[Dict[str, Any]]) -> List[str]:
        """
        Extract created files from task history
        
        Args:
            history: Task execution history
            
        Returns:
            List of created files
        """
        files = []
        
        for record in history:
            if 'result' in record and not record.get('error'):
                result = record['result']
                
                # Look for file creation indicators
                if 'edit_file' in result or 'create' in result.lower() or '创建' in result:
                    import re
                    # Extract file patterns
                    file_matches = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{1,4}', result)
                    files.extend(file_matches)
                
                # Look for File: patterns
                if 'File:' in result:
                    import re
                    file_matches = re.findall(r'File: ([^\n]+)', result)
                    files.extend(file_matches)
        
        # Remove duplicates and limit quantity
        return list(dict.fromkeys(files))[:10]
    
    @staticmethod
    def extract_error_messages(history: List[Dict[str, Any]]) -> List[str]:
        """
        Extract error messages from task history
        
        Args:
            history: Task execution history
            
        Returns:
            List of error messages
        """
        errors = []
        
        for record in history:
            if record.get('error'):
                errors.append(record['error'][:200])
            elif 'result' in record:
                result = record['result'].lower()
                # Look for error indicators
                if any(keyword in result for keyword in ['error', 'fail', 'exception', '错误', '失败', '异常']):
                    errors.append(record['result'][:200])
        
        # Remove duplicates and limit quantity
        return list(dict.fromkeys(errors))[:5]