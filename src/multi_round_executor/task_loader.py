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
Task loader for parsing todo.csv files
"""

import csv
import os
from typing import List, Dict, Any
from tools.print_system import print_system, print_current


class TaskLoader:
    """Task loader for parsing and validating todo.csv files"""
    
    @staticmethod
    def load_todo_csv(csv_file: str) -> List[Dict[str, Any]]:
        """
        Load todo.csv file
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Task list
        """
        tasks = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                # Read all lines
                lines = file.readlines()
                
                # Check if first line is header
                first_line = lines[0].strip() if lines else ""
                has_header = TaskLoader._detect_header(first_line)
                
                if has_header:
                    # Has header, use DictReader
                    file.seek(0)  # Reset file pointer
                    reader = csv.DictReader(file)
                    for row in reader:
                        tasks.append(TaskLoader._parse_task_row(row))
                else:
                    # No header, manual parsing
                    print_current("⚠️  CSV file missing header, using default column names for parsing")
                    for line_num, line in enumerate(lines, 1):
                        if line.strip():
                            parts = [part.strip() for part in line.strip().split(',')]
                            if len(parts) >= 3:
                                tasks.append(TaskLoader._parse_task_parts(parts, line_num))
                
                print_current(f"📋 Successfully loaded {len(tasks)} tasks")
                for i, task in enumerate(tasks):
                    print_current(f"  Task{task['Task ID']}: {task['Task Name']} - {task['Task Description'][:50]}...")
                    
        except Exception as e:
            print_current(f"Error loading CSV file: {e}")
            return []
        
        return tasks
    
    @staticmethod
    def _detect_header(first_line: str) -> bool:
        """
        Detect if the first line is a header
        
        Args:
            first_line: First line content
            
        Returns:
            True if header is detected
        """
        header_indicators = [
            'Task ID', '任务编号', 'task', 'Task Name', '任务名称',
            'Task Description', '任务详细描述', 'Description', '描述'
        ]
        
        return any(indicator in first_line for indicator in header_indicators)
    
    @staticmethod
    def _parse_task_row(row: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse a task row from CSV DictReader
        
        Args:
            row: CSV row dictionary
            
        Returns:
            Standardized task dictionary
        """
        return {
            'Task ID': row.get('Task ID', row.get('任务编号', '')),
            'Task Name': row.get('Task Name', row.get('任务名称', '')),
            'Task Description': row.get('Task Description', row.get('任务详细描述', '')),
            'Execution Status': row.get('Execution Status', row.get('执行状态', '0')),
            'Dependencies': row.get('Dependencies', row.get('依赖任务', ''))
        }
    
    @staticmethod
    def _parse_task_parts(parts: List[str], line_num: int) -> Dict[str, Any]:
        """
        Parse task from comma-separated parts
        
        Args:
            parts: List of comma-separated parts
            line_num: Line number for generating default IDs
            
        Returns:
            Standardized task dictionary
        """
        return {
            'Task ID': parts[0] if len(parts) > 0 else str(line_num),
            'Task Name': parts[1] if len(parts) > 1 else f"Task{line_num}",
            'Task Description': parts[2] if len(parts) > 2 else "",
            'Execution Status': parts[3] if len(parts) > 3 else '0',
            'Dependencies': parts[4] if len(parts) > 4 else ''
        }
    
    @staticmethod
    def read_csv_content(csv_file: str) -> str:
        """
        Read raw CSV file content
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Raw CSV content
        """
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Unable to read CSV file: {e}")
    
    @staticmethod
    def validate_csv_file(csv_file: str) -> bool:
        """
        Validate if CSV file exists and is not empty
        
        Args:
            csv_file: CSV file path
            
        Returns:
            True if file is valid
        """
        if not os.path.exists(csv_file):
            return False
            
        try:
            content = TaskLoader.read_csv_content(csv_file)
            return bool(content.strip())
        except Exception:
            return False