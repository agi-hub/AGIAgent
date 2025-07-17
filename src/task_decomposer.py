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
Task Decomposer - Decompose complex tasks into executable subtasks
"""

import os
import re
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from tool_executor import ToolExecutor
from config_loader import get_api_key, get_api_base, get_model, get_max_tokens, get_streaming, get_language, get_truncation_length

class TaskDecomposer:
    def __init__(self, api_key: str = None, 
                 model: str = None, 
                 api_base: str = None):
        """
        Initialize task decomposer
        
        Args:
            api_key: API key
            model: Model name
            api_base: API base URL
        """
        # Load API key from config/config.txt if not provided
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key not found. Please provide api_key parameter or set it in config/config.txt")
        
        # Load model from config/config.txt if not provided
        if model is None:
            model = get_model()
            if model is None:
                raise ValueError("Model not found. Please provide model parameter or set it in config/config.txt")
        
        # Load API base from config/config.txt if not provided
        if api_base is None:
            api_base = get_api_base()
            if api_base is None:
                raise ValueError("API base URL not found. Please provide api_base parameter or set it in config/config.txt")
        
        self.executor = ToolExecutor(api_key, model, api_base)
        
    def decompose_task(self, user_requirement: str, output_file: str = "todo.csv", workspace_dir: str = None) -> str:
        """
        Decompose user requirements into subtasks and create todo.csv file
        
        Args:
            user_requirement: User requirement description
            output_file: Output CSV file name
            workspace_dir: Workspace directory path
            
        Returns:
            Task decomposition result information
        """
        
        # Build specialized system prompt
        system_prompt = self._create_task_decomposition_prompt()
        
        # Build user prompt
        user_prompt = f"User requirement: {user_requirement}"
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM API directly
            print(f"🤖 Calling LLM for task decomposition: {user_requirement[:get_truncation_length()]}...")
            
            if self.executor.is_claude:
                # Use Anthropic Claude API - streaming call
                # Claude needs to separate system and other messages
                system_message = ""
                claude_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        claude_messages.append(msg)
                
                print("🔄 Starting task decomposition generation...")
                response = self.executor.client.messages.create(
                    model=self.executor.model,
                    max_tokens=self.executor._get_max_tokens_for_model(self.executor.model),
                    system=system_message,
                    messages=claude_messages,
                    temperature=0.7
                )
                
                # Get complete response content
                content = response.content[0].text
                print("\n🤖 Task decomposition completed")
                print("✅ Generation completed")
                
            else:
                # Use OpenAI API - batch call
                print("🔄 Starting task decomposition generation...")
                response = self.executor.client.chat.completions.create(
                    model=self.executor.model,
                    messages=messages,
                    max_tokens=self.executor._get_max_tokens_for_model(self.executor.model),
                    temperature=0.7,
                    top_p=0.8
                )
                
                # Get complete response content
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                else:
                    content = ""
                    print("⚠️ Warning: OpenAI API returned empty choices list")
                print("\n🤖 Task decomposition completed")
                print("✅ Generation completed")
            
            # No longer repeat display response content since it's already streamed
            print(f"\n📝 Decomposition content length: {len(content)} characters")
            
            # Extract task list from response and create CSV file
            tasks = self._extract_tasks_from_response(content)
            
            
            # Create CSV file
            self._create_csv_file(tasks, output_file)
            
            return f"Task decomposition completed successfully, decomposed into {len(tasks)} subtasks, saved to {output_file}"
            
        except Exception as e:
            error_msg = f"Task decomposition failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _extract_tasks_from_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Extract task list from LLM response
        
        Args:
            response_content: LLM response content
            
        Returns:
            Task list
        """
        tasks = []
        
        # First try to parse structured task information from response
        lines = response_content.split('\n')
        current_task = {}
        task_id = 1
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Look for task title pattern (e.g.: 1. Task name or **1. Task name**)
            task_pattern = r'^(?:\*\*)?(\d+)\.?\s*(.+?)(?:\*\*)?$'
            match = re.match(task_pattern, line)
            
            if match:
                # If there was a previous task, save it first
                if current_task and 'name' in current_task:
                    tasks.append(current_task)
                
                # Start new task
                task_num = int(match.group(1))
                task_name = match.group(2).strip()
                current_task = {
                    'id': task_num,
                    'name': task_name,
                    'description': task_name,  # Default description is task name
                    'status': 0,
                    'dependencies': ''
                }
                continue
            
            # If current line contains description information, update task description
            if current_task and ('Description' in line or 'description' in line or 'Detail' in line or 'detail' in line):
                if ':' in line or '：' in line:
                    desc_part = line.split(':')[-1].split('：')[-1].strip()
                    if desc_part:
                        current_task['description'] = desc_part
                continue
            
            # If current line contains dependency information
            if current_task and ('Dependency' in line or 'dependency' in line or 'Depends' in line or 'depends' in line):
                if ':' in line or '：' in line:
                    dep_part = line.split(':')[-1].split('：')[-1].strip()
                    if dep_part and dep_part != 'None' and dep_part != 'No dependencies':
                        current_task['dependencies'] = dep_part
                continue
        
        # Add the last task
        if current_task and 'name' in current_task:
            tasks.append(current_task)
        
        # If no structured tasks were parsed, generate default task list
        if not tasks:
            print("⚠️ Failed to parse structured tasks, generating default task decomposition")
            tasks = self._generate_default_tasks(response_content)
        
        return tasks
    
    def _generate_default_tasks(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Generate default task list when structured tasks cannot be parsed
        
        Args:
            response_content: LLM response content
            
        Returns:
            Default task list
        """
        # Simple keyword matching to generate tasks
        default_tasks = [
            {
                'id': 1,
                'name': 'Requirement Analysis',
                'description': 'Analyze and understand user requirements, clarify project goals',
                'status': 0,
                'dependencies': ''
            },
            {
                'id': 2,
                'name': 'Technology Selection',
                'description': 'Select appropriate technology stack and tools',
                'status': 0,
                'dependencies': '1'
            },
            {
                'id': 3,
                'name': 'Project Design',
                'description': 'Design project architecture and module structure',
                'status': 0,
                'dependencies': '2'
            },
            {
                'id': 4,
                'name': 'Core Feature Implementation',
                'description': 'Implement core functional modules of the project',
                'status': 0,
                'dependencies': '3'
            },
            {
                'id': 5,
                'name': 'Testing and Validation',
                'description': 'Test and validate implemented features',
                'status': 0,
                'dependencies': '4'
            }
        ]
        
        return default_tasks
    
    def _create_csv_file(self, tasks: List[Dict[str, Any]], csv_path: str) -> None:
        """
        Create CSV file
        
        Args:
            tasks: Task list
            csv_path: CSV file path
        """
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Task ID', 'Task Name', 'Task Description', 'Status', 'Dependencies']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write task data
            for task in tasks:
                writer.writerow({
                    'Task ID': task.get('id', ''),
                    'Task Name': task.get('name', ''),
                    'Task Description': task.get('description', ''),
                    'Status': task.get('status', 0),
                    'Dependencies': task.get('dependencies', '')
                })
        
        print(f"📝 CSV file created: {csv_path}")

    def _create_task_decomposition_prompt(self) -> str:
        """
        Create system prompt for task decomposition
        
        Returns:
            System prompt string
        """
        return """You are a professional task decomposition expert, skilled at breaking down complex user requirements into specific executable subtasks.

Your task is to:
1. Analyze the requirements provided by the user
2. Decompose requirements into specific, executable subtasks
3. Provide clear descriptions and dependencies for each subtask

Please output the task decomposition results in the following format:

**Task Decomposition Results:**

1. Task Name 1
   Description: Specific execution content and objectives
   Dependencies: None (or dependent task numbers)

2. Task Name 2
   Description: Specific execution content and objectives
   Dependencies: 1 (if dependent on task 1)

3. Task Name 3
   Description: Specific execution content and objectives
   Dependencies: 1/2 (if dependent on tasks 1 and 2)

Task decomposition principles:
1. Subtasks should be specific and executable, avoiding being too abstract
2. Each subtask should have clear inputs and outputs
3. There should be reasonable dependencies between subtasks
4. Task granularity should be moderate, not too coarse or too fine
5. Ensure that executing all subtasks achieves the final goal
6. Consider actual development processes and arrange in reasonable order

Please strictly follow the above format for output so I can correctly parse and process the task information.
"""

def main():
    """
    Demonstrate the usage of task decomposer
    """
    print("=== Task Decomposer Demo ===\n")
    
    decomposer = TaskDecomposer()
    
    # Example user requirement
    example_requirement = """
    I want to develop a Python Web application with the following features:
    1. User registration and login functionality
    2. Users can publish articles
    3. Other users can comment on articles
    4. An admin backend to manage users and articles
    5. Frontend uses modern responsive design
    """
    
    print("User requirement:")
    print(example_requirement)
    print("\nStarting task decomposition...")
    
    # Execute task decomposition
    result = decomposer.decompose_task(example_requirement)
    
    print("\nTask decomposition result:")
    print(result)
    
    # Check if CSV file was created
    if os.path.exists("todo.csv"):
        print("\n✅ todo.csv file successfully created")
        with open("todo.csv", 'r', encoding='utf-8') as f:
            content = f.read()
            print("File content preview:")
            print(content[:get_truncation_length()] + "..." if len(content) > get_truncation_length() else content)
    else:
        print("\n❌ todo.csv file not created")

if __name__ == "__main__":
    main() 