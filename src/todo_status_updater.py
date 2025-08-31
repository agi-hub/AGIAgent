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
Todo Status Updater - Automatically update task status in todo.md file
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TodoStatusUpdater:
    """Todo status updater for managing task completion status"""
    
    def __init__(self, todo_file: str = "todo.md", create_backup: bool = False):
        """
        Initialize todo status updater
        
        Args:
            todo_file: Path to todo.md file
            create_backup: Whether to create backup files (default: False)
        """
        self.todo_file = todo_file
        self.backup_file = f"{todo_file}.backup"
        self.create_backup = create_backup
        
        # Set log file in the same directory as todo.md
        todo_dir = os.path.dirname(os.path.abspath(todo_file))
        self.log_file = os.path.join(todo_dir, "todo_status_log.json")
        
    def read_todo_file(self) -> Optional[str]:
        """
        Read todo.md file content
        
        Returns:
            File content string or None if file doesn't exist
        """
        try:
            if not os.path.exists(self.todo_file):
                print(f"⚠️ Todo file not found: {self.todo_file}")
                return None
                
            with open(self.todo_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading todo file: {e}")
            return None
    
    def backup_todo_file(self) -> bool:
        """
        Create backup of todo file
        
        Returns:
            Success status
        """
        try:
            content = self.read_todo_file()
            if content is None:
                return False
                
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def parse_tasks(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from todo.md content
        
        Args:
            content: Todo file content
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        lines = content.split('\n')
        current_task = None
        
        for line in lines:
            # Match task header pattern: ## 📝 Task 1: Task Name
            task_header_match = re.match(r'^## ([📝✅🔄❌⏸️]) Task (\d+): (.+)$', line.strip())
            if task_header_match:
                # Save previous task if exists
                if current_task:
                    tasks.append(current_task)
                
                status_icon = task_header_match.group(1)
                task_id = int(task_header_match.group(2))
                task_name = task_header_match.group(3)
                
                # Determine status from icon
                status = self._icon_to_status(status_icon)
                
                current_task = {
                    'id': task_id,
                    'name': task_name,
                    'status': status,
                    'original_line': line,
                    'checkbox_lines': []
                }
                continue
            
            # Match checkbox lines
            if current_task and line.strip().startswith('- ['):
                checkbox_match = re.match(r'^- \[([x ])\] \*\*(.+?)\*\*: (.+)$', line.strip())
                if checkbox_match:
                    checked = checkbox_match.group(1) == 'x'
                    field_name = checkbox_match.group(2)
                    field_value = checkbox_match.group(3)
                    
                    current_task['checkbox_lines'].append({
                        'checked': checked,
                        'field': field_name,
                        'value': field_value,
                        'original_line': line
                    })
        
        # Add last task
        if current_task:
            tasks.append(current_task)
            
        return tasks
    
    def _icon_to_status(self, icon: str) -> TaskStatus:
        """Convert status icon to TaskStatus enum"""
        icon_map = {
            '📝': TaskStatus.PENDING,
            '🔄': TaskStatus.IN_PROGRESS,
            '✅': TaskStatus.COMPLETED,
            '❌': TaskStatus.BLOCKED,
            '⏸️': TaskStatus.SKIPPED
        }
        return icon_map.get(icon, TaskStatus.PENDING)
    
    def _status_to_icon(self, status: TaskStatus) -> str:
        """Convert TaskStatus enum to status icon"""
        status_map = {
            TaskStatus.PENDING: '📝',
            TaskStatus.IN_PROGRESS: '🔄', 
            TaskStatus.COMPLETED: '✅',
            TaskStatus.BLOCKED: '❌',
            TaskStatus.SKIPPED: '⏸️'
        }
        return status_map.get(status, '📝')
    
    def update_task_status(self, task_id: int, new_status: TaskStatus, 
                          description: str = None) -> bool:
        """
        Update specific task status
        
        Args:
            task_id: Task ID to update
            new_status: New task status
            description: Optional status description
            
        Returns:
            Success status
        """
        try:
            # Create backup first (only if enabled)
            if self.create_backup:
                if not self.backup_todo_file():
                    print("⚠️ Warning: Could not create backup")
            # else:
            #     print("🔧 Backup disabled - no backup file created")
            
            content = self.read_todo_file()
            if content is None:
                return False
            
            # Parse tasks
            tasks = self.parse_tasks(content)
            target_task = None
            
            for task in tasks:
                if task['id'] == task_id:
                    target_task = task
                    break
            
            if not target_task:
                print(f"❌ Task {task_id} not found")
                return False
            
            # Update content
            updated_content = self._update_content_for_task(
                content, target_task, new_status, description
            )
            
            # Write updated content
            with open(self.todo_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            status_name = new_status.value.replace('_', ' ').title()
            print(f"✅ Task {task_id} status updated to: {status_name}")
            
            # Log status change
            self._log_status_change(task_id, target_task['name'], new_status, description)
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating task status: {e}")
            return False
    
    def _update_content_for_task(self, content: str, task: Dict[str, Any], 
                                new_status: TaskStatus, description: str = None) -> str:
        """Update content for specific task"""
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            # Update task header icon
            if f"Task {task['id']}:" in line and line.startswith('##'):
                old_icon = self._icon_to_status(task['status'])
                new_icon = self._status_to_icon(new_status)
                updated_line = re.sub(r'^## [📝✅🔄❌⏸️]', f'## {new_icon}', line)
                updated_lines.append(updated_line)
                
            # Update checkbox for Status field
            elif ('**Status**:' in line and 
                  line.strip().startswith('- [') and 
                  self._is_in_task_section(lines, updated_lines, task['id'])):
                
                checkbox_char = 'x' if new_status == TaskStatus.COMPLETED else ' '
                status_text = new_status.value.replace('_', ' ').title()
                if description:
                    status_text += f" - {description}"
                
                updated_line = f"- [{checkbox_char}] **Status**: {status_text}"
                updated_lines.append(updated_line)
                
            else:
                updated_lines.append(line)
        
        return '\n'.join(updated_lines)
    
    def _is_in_task_section(self, all_lines: List[str], processed_lines: List[str], 
                           task_id: int) -> bool:
        """Check if current line is in specific task section"""
        # Simple heuristic: check if we recently processed the task header
        for line in reversed(processed_lines[-10:]):  # Check last 10 lines
            if f"Task {task_id}:" in line and line.startswith('##'):
                return True
        return False
    
    def mark_task_completed(self, task_id: int, completion_note: str = None) -> bool:
        """
        Mark task as completed
        
        Args:
            task_id: Task ID to mark as completed
            completion_note: Optional completion note
            
        Returns:
            Success status
        """
        return self.update_task_status(task_id, TaskStatus.COMPLETED, completion_note)
    
    def mark_task_in_progress(self, task_id: int, progress_note: str = None) -> bool:
        """
        Mark task as in progress
        
        Args:
            task_id: Task ID to mark as in progress
            progress_note: Optional progress note
            
        Returns:
            Success status
        """
        return self.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress_note)
    
    def mark_task_blocked(self, task_id: int, block_reason: str = None) -> bool:
        """
        Mark task as blocked
        
        Args:
            task_id: Task ID to mark as blocked
            block_reason: Optional block reason
            
        Returns:
            Success status
        """
        return self.update_task_status(task_id, TaskStatus.BLOCKED, block_reason)
    
    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get task completion summary
        
        Returns:
            Summary dictionary with counts and progress
        """
        content = self.read_todo_file()
        if content is None:
            return {"error": "Could not read todo file"}
        
        tasks = self.parse_tasks(content)
        
        summary = {
            "total_tasks": len(tasks),
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "blocked": 0,
            "skipped": 0,
            "completion_percentage": 0,
            "tasks": []
        }
        
        for task in tasks:
            status = task['status']
            summary[status.value] += 1
            
            summary["tasks"].append({
                "id": task['id'],
                "name": task['name'],
                "status": status.value
            })
        
        if summary["total_tasks"] > 0:
            summary["completion_percentage"] = round(
                (summary["completed"] / summary["total_tasks"]) * 100, 1
            )
        
        return summary
    
    def _log_status_change(self, task_id: int, task_name: str, 
                          new_status: TaskStatus, description: str = None):
        """Log status change for tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "task_name": task_name,
            "new_status": new_status.value,
            "description": description
        }
        
        log_file = self.log_file
        
        try:
            # Read existing log
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            else:
                log_data = {"status_changes": []}
            
            # Add new entry
            log_data["status_changes"].append(log_entry)
            
            # Keep only last 100 entries
            log_data["status_changes"] = log_data["status_changes"][-100:]
            
            # Write log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not write status log: {e}")
    
    def display_progress(self):
        """Display current task progress"""
        summary = self.get_task_summary()
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n📊 Task Progress Summary:")
        print(f"   📝 Total Tasks: {summary['total_tasks']}")
        print(f"   ✅ Completed: {summary['completed']}")
        print(f"   🔄 In Progress: {summary['in_progress']}")
        print(f"   📝 Pending: {summary['pending']}")
        print(f"   ❌ Blocked: {summary['blocked']}")
        print(f"   ⏸️ Skipped: {summary['skipped']}")
        print(f"   📈 Progress: {summary['completion_percentage']}%")
        
        # Progress bar
        if summary['total_tasks'] > 0:
            completed_ratio = summary['completed'] / summary['total_tasks']
            bar_length = 20
            filled_length = int(bar_length * completed_ratio)
            bar = '▰' * filled_length + '▱' * (bar_length - filled_length)
            print(f"   📊 Progress Bar: {bar} {summary['completion_percentage']}%")

def main():
    """Demo usage of TodoStatusUpdater"""
    print("=== Todo Status Updater Demo ===\n")
    
    updater = TodoStatusUpdater()
    
    # Display current progress
    updater.display_progress()
    
    # Example: Mark task 1 as in progress
    print("\n📝 Example: Marking task 1 as in progress...")
    updater.mark_task_in_progress(1, "Started requirement analysis")
    
    # Example: Mark task 2 as completed  
    print("\n✅ Example: Marking task 2 as completed...")
    updater.mark_task_completed(2, "Technology stack selected")
    
    # Display updated progress
    print("\n📊 Updated Progress:")
    updater.display_progress()

if __name__ == "__main__":
    main() 